#!/usr/bin/env python3
"""Evaluate a fixed-covariance Info-LDM random-step nuisance checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from implicit_nuisance.utils.random_step_nuisance_eval import (
    evaluate_initial_distribution,
    evaluate_random_step_nuisance,
)


def _load_config(path: Path) -> Any:
    from omegaconf import OmegaConf

    if path.suffix.lower() == ".json":
        with path.open() as file:
            cfg = OmegaConf.create(json.load(file))
    else:
        cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    return cfg


def _load_model(checkpoint: Path, cfg: Any, device: str) -> Any:
    import torch

    from implicit_nuisance.methods import METHODS

    if cfg.method not in METHODS:
        raise ValueError(f"unknown method {cfg.method!r}; choose one of {tuple(METHODS)}")
    model = METHODS[cfg.method](cfg)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0
        payload = torch.load(checkpoint, map_location="cpu")
    state_dict = payload.get("state_dict", payload)
    incompatible = model.load_state_dict(state_dict, strict=False)

    allowed_missing = {"encoder.prediction_variance"}
    disallowed_missing = set(incompatible.missing_keys) - allowed_missing
    allowed_unexpected_prefixes = ("encoder.variance_predictor.",)
    disallowed_unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith(allowed_unexpected_prefixes)
    ]
    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            "checkpoint and config do not describe the same model; "
            f"missing keys: {sorted(disallowed_missing)}, "
            f"unexpected keys: {sorted(disallowed_unexpected)}"
        )
    if incompatible.missing_keys:
        print(
            "warning: legacy checkpoint has no fixed-covariance buffer; using "
            "the configured value for: " + ", ".join(incompatible.missing_keys),
            file=sys.stderr,
        )
    if incompatible.unexpected_keys:
        print(
            "warning: ignoring removed legacy covariance-predictor keys: "
            + ", ".join(incompatible.unexpected_keys),
            file=sys.stderr,
        )
    return model.to(device).eval()


def _sequence_subset(dataset: Any, maximum: int, seed: int) -> Any:
    from torch.utils.data import Subset

    if maximum < 2:
        raise ValueError("sample limits must be at least two")
    sequence_length = int(getattr(dataset, "sequence_length", 0))
    if sequence_length < 2:
        raise ValueError("random-step evaluation requires sequence_length >= 2")
    number = min(len(dataset), math.ceil(maximum / sequence_length))
    indices = np.random.default_rng(seed).choice(len(dataset), size=number, replace=False)
    return Subset(dataset, np.sort(indices))


def _flatten_sequence(array: np.ndarray) -> np.ndarray:
    if array.ndim < 3:
        raise ValueError(f"expected a batched sequence array, got {array.shape}")
    return array.reshape(-1, int(np.prod(array.shape[2:])))


def _source_transition_rows(array: Any, sequence_length: int) -> np.ndarray:
    """Flatten source values aligned with successor time steps 1 through T-1."""

    value = array.detach().cpu().numpy()
    if value.ndim != 3:
        raise ValueError(f"transition arrays must have shape [B,T,D], got {value.shape}")
    if value.shape[1] == sequence_length:
        value = value[:, 1:]
    elif value.shape[1] != sequence_length - 1:
        raise ValueError(
            f"transition length must be {sequence_length} or {sequence_length - 1}, "
            f"got {value.shape[1]}"
        )
    return value.reshape(-1, value.shape[-1])


def _model_transition_rows(array: Any, sequence_length: int) -> np.ndarray:
    """Flatten outputs aligned with the ``T - 1`` successor transitions."""

    value = array.detach().cpu().numpy()
    if value.ndim != 3:
        raise ValueError(f"transition arrays must have shape [B,T,D], got {value.shape}")
    if value.shape[1] == sequence_length:
        value = value[:, :-1]
    elif value.shape[1] != sequence_length - 1:
        raise ValueError(
            f"model transition length must be {sequence_length} or "
            f"{sequence_length - 1}, got {value.shape[1]}"
        )
    return value.reshape(-1, value.shape[-1])


def _fixed_diagonal(name: str, values: np.ndarray, *, standard_deviation: bool) -> np.ndarray:
    if values.ndim != 2 or len(values) < 1:
        raise ValueError(f"{name} must contain transition rows")
    diagonal = values[0]
    if np.any(diagonal <= 0):
        raise ValueError(f"{name} must be strictly positive")
    if not np.allclose(values, diagonal, rtol=0.0, atol=1e-7):
        raise ValueError(f"{name} is not fixed across histories")
    return diagonal**2 if standard_deviation else diagonal


def _collect(
    model: Any,
    dataset: Any,
    *,
    maximum: int,
    selection_seed: int,
    representation_key: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    subset = _sequence_subset(dataset, maximum, selection_seed)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )
    batches: dict[str, list[np.ndarray]] = {
        "z": [],
        "signal": [],
        "nuisance": [],
        "observations": [],
    }
    transitions: dict[str, list[np.ndarray]] = {
        "model_samples": [],
        "model_means": [],
        "model_variances": [],
        "source_samples": [],
        "source_means": [],
        "source_stds": [],
        "initial_signal": [],
    }

    with torch.inference_mode():
        for observations, labels in loader:
            observations_device = observations.to(device, non_blocking=True)
            latents = model.encoder(observations_device)
            if representation_key not in latents or latents[representation_key] is None:
                raise KeyError(
                    f"representation {representation_key!r} is unavailable; "
                    f"choose one of {tuple(latents)}"
                )
            missing_labels = {
                "signal",
                "nuisance",
                "transition_mean",
                "transition_std",
            } - set(labels)
            if missing_labels:
                raise KeyError(
                    "random-step nuisance labels are missing "
                    + ", ".join(sorted(missing_labels))
                )

            sequence_length = observations.shape[1]
            representation = latents[representation_key].detach().cpu().numpy()
            signal = labels["signal"].detach().cpu().numpy()
            nuisance = labels["nuisance"].detach().cpu().numpy()
            observation_array = observations.detach().cpu().numpy()
            for name, value in (
                ("z", representation),
                ("signal", signal),
                ("nuisance", nuisance),
                ("observations", observation_array),
            ):
                if value.shape[:2] != observations.shape[:2]:
                    raise ValueError(
                        f"{name} sequence shape {value.shape[:2]} does not match "
                        f"observations {tuple(observations.shape[:2])}"
                    )
                batches[name].append(_flatten_sequence(value))

            required_latents = (
                "inferences",
                "predictions",
                "predictions_variances",
            )
            missing_latents = [key for key in required_latents if key not in latents]
            if missing_latents:
                raise KeyError(f"model transition outputs are missing {missing_latents}")
            transitions["model_samples"].append(
                _source_transition_rows(latents["inferences"], sequence_length)
            )
            transitions["model_means"].append(
                _model_transition_rows(latents["predictions"], sequence_length)
            )
            transitions["model_variances"].append(
                _model_transition_rows(latents["predictions_variances"], sequence_length)
            )
            transitions["source_samples"].append(
                _source_transition_rows(labels["signal"], sequence_length)
            )
            transitions["source_means"].append(
                _source_transition_rows(labels["transition_mean"], sequence_length)
            )
            transitions["source_stds"].append(
                _source_transition_rows(labels["transition_std"], sequence_length)
            )
            transitions["initial_signal"].append(signal[:, 0])

    if not batches["z"]:
        raise ValueError("dataset yielded no evaluation samples")
    arrays = {name: np.concatenate(values, axis=0) for name, values in batches.items()}
    transition_arrays = {
        name: np.concatenate(values, axis=0) for name, values in transitions.items()
    }

    rng = np.random.default_rng(selection_seed + 1009)
    selected_rows = rng.permutation(len(arrays["z"]))[:maximum]
    arrays = {name: value[selected_rows] for name, value in arrays.items()}

    model_covariance = _fixed_diagonal(
        "model prediction variance",
        transition_arrays["model_variances"],
        standard_deviation=False,
    )
    source_covariance = _fixed_diagonal(
        "source transition standard deviation",
        transition_arrays["source_stds"],
        standard_deviation=True,
    )
    transition_inputs = {
        "model": {
            "samples": transition_arrays["model_samples"],
            "means": transition_arrays["model_means"],
            "covariance": model_covariance,
        },
        "source": {
            "samples": transition_arrays["source_samples"],
            "means": transition_arrays["source_means"],
            "covariance": source_covariance,
        },
        "initial_signal": transition_arrays["initial_signal"],
    }
    return arrays, transition_inputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--representation-key", default="inferences")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", help="checkpoint inference device, e.g. cpu or cuda")
    parser.add_argument("--max-train-samples", type=int, default=10000)
    parser.add_argument("--max-test-samples", type=int, default=10000)
    parser.add_argument("--kernel-sample-max", type=int, default=1024)
    parser.add_argument("--kernel-alpha", type=float, default=1e-3)
    parser.add_argument("--kernel-gamma", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    return parser.parse_args()


def main() -> None:
    import torch

    from implicit_nuisance.data import DATASETS

    args = _parse_args()
    cfg = _load_config(args.config)
    if cfg.data.dataset != "randstep_nuisance":
        raise ValueError(
            "this evaluator requires data.dataset='randstep_nuisance', got "
            f"{cfg.data.dataset!r}"
        )
    if cfg.data.dataset not in DATASETS:
        raise ValueError(
            f"unknown dataset {cfg.data.dataset!r}; choose one of {tuple(DATASETS)}"
        )
    state_dimension = int(cfg.method_kwargs.state_dim)
    signal_dimension = int(cfg.data.settings.signal_dim)
    nuisance_dimension = int(cfg.data.settings.nuisance_dim)
    if state_dimension != signal_dimension:
        raise ValueError(
            "the nuisance-removal experiment requires equal signal/code dimensions, "
            f"but state_dim={state_dimension} and signal_dim={signal_dimension}"
        )
    if nuisance_dimension < 1:
        raise ValueError(
            f"nuisance_dim must be at least one, got {nuisance_dimension}"
        )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = device
    model = _load_model(args.checkpoint, cfg, device)
    dataset_type = DATASETS[cfg.data.dataset]
    train_dataset = dataset_type(cfg, split="train")
    test_dataset = dataset_type(cfg, split="test")
    train_arrays, train_transitions = _collect(
        model,
        train_dataset,
        maximum=args.max_train_samples,
        selection_seed=args.seed,
        representation_key=args.representation_key,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    test_arrays, test_transitions = _collect(
        model,
        test_dataset,
        maximum=args.max_test_samples,
        selection_seed=args.seed + 1,
        representation_key=args.representation_key,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    # Calibration is reported only on the held-out test split.  The train
    # transition arrays are collected solely to verify the fixed mechanism.
    model_transition = test_transitions["model"]
    source_transition = test_transitions["source"]
    for transition_name in ("model", "source"):
        if not np.allclose(
            train_transitions[transition_name]["covariance"],
            test_transitions[transition_name]["covariance"],
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(f"train/test {transition_name} covariance differs")

    metrics = evaluate_random_step_nuisance(
        train_arrays["z"],
        train_arrays["signal"],
        train_arrays["nuisance"],
        train_arrays["observations"],
        test_arrays["z"],
        test_arrays["signal"],
        test_arrays["nuisance"],
        test_arrays["observations"],
        model_transition=model_transition,
        source_transition=source_transition,
        kernel_sample_max=args.kernel_sample_max,
        kernel_alpha=args.kernel_alpha,
        kernel_gamma=args.kernel_gamma,
        random_state=args.seed,
    )
    initial_distribution = test_dataset.initial_distribution
    metrics["initial_signal"] = evaluate_initial_distribution(
        test_transitions["initial_signal"],
        initial_distribution,
    )
    metrics["metadata"]["nuisance_dim_greater_than_signal_dim"] = bool(
        nuisance_dimension > signal_dimension
    )
    metrics["gaussian_conditional"]["metadata"] = {
        "split": "test",
        "successor_steps_only": True,
        "initial_distribution": initial_distribution,
        "state_dim": state_dimension,
        "signal_dim": signal_dimension,
    }
    rendered = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
