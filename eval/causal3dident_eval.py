#!/usr/bin/env python3
"""Run held-out Causal3DIdent representation probes.

The most reproducible interface consumes two NumPy archives::

    python eval/causal3dident_eval.py \
        --train-npz representations_train.npz \
        --test-npz representations_test.npz \
        --output metrics.json

Each archive must contain ``z``, ``factors``, and ``class`` arrays.  A
checkpoint/config mode is also provided for the local InfoLDM model and
Causal3DIdent dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from implicit_nuisance.utils.causal3dident_eval import (
    CAUSAL3DIDENT_FACTOR_NAMES,
    DEFAULT_SIGNAL_INDICES,
    evaluate_gaussian_conditional,
    evaluate_representations,
)


def _comma_separated_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("at least one index is required")
    return result


def _comma_separated_strings(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one name is required")
    return result


def _load_archive(
    path: Path,
    *,
    z_key: str,
    factors_key: str,
    class_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in (z_key, factors_key, class_key) if key not in archive]
        if missing:
            raise KeyError(f"{path} is missing arrays: {', '.join(missing)}")
        return (
            np.asarray(archive[z_key]),
            np.asarray(archive[factors_key]),
            np.asarray(archive[class_key]),
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


def _load_local_model(checkpoint: Path, cfg: Any, device: str) -> Any:
    import torch

    from implicit_nuisance.methods import METHODS

    if cfg.method not in METHODS:
        raise ValueError(f"unknown method {cfg.method!r}; choose one of {tuple(METHODS)}")
    model = METHODS[cfg.method](cfg)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only argument
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
            "warning: legacy checkpoint is missing fixed-covariance keys; "
            "using the configured value for: "
            + ", ".join(incompatible.missing_keys),
            file=sys.stderr,
        )
    if incompatible.unexpected_keys:
        print(
            "warning: ignoring removed legacy covariance-predictor keys: "
            + ", ".join(incompatible.unexpected_keys),
            file=sys.stderr,
        )
    return model.to(device).eval()


def _collect_representations(
    model: Any,
    dataset: Any,
    *,
    representation_key: str,
    batch_size: int,
    num_workers: int,
    device: str,
    maximum: int | None,
    sequence_index: int | None = None,
    collect_transitions: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )
    representations = []
    factors = []
    classes = []
    transition_batches: dict[str, list[np.ndarray]] = {}
    count = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            latents = model.encoder(images)
            if not isinstance(latents, dict) or representation_key not in latents:
                available = tuple(latents) if isinstance(latents, dict) else type(latents).__name__
                raise KeyError(
                    f"representation {representation_key!r} not found; available: {available}"
                )
            representation = latents[representation_key]
            if representation is None:
                raise ValueError(f"representation {representation_key!r} is None")
            representation = representation.detach().cpu().numpy()
            factor_batch = labels["factors"].detach().cpu().numpy()
            class_batch = labels["class"].detach().cpu().numpy()

            if collect_transitions:
                required_latents = (
                    "inferences",
                    "predictions",
                    "predictions_variances",
                )
                missing_latents = [key for key in required_latents if key not in latents]
                required_labels = (
                    "signal_normal_score",
                    "transition_mean",
                    "transition_target",
                    "transition_std",
                    "knn_distance",
                )
                missing_labels = [key for key in required_labels if key not in labels]
                if missing_latents or missing_labels:
                    raise KeyError(
                        "transition diagnostics require latents "
                        f"{required_latents} and labels {required_labels}; missing "
                        f"{missing_latents + missing_labels}"
                    )
                transition_values = {
                    "model_sample": latents["inferences"][:, 1:],
                    "model_mean": latents["predictions"][:, :-1],
                    "model_variance": latents["predictions_variances"][:, :-1],
                    "data_sample": labels["signal_normal_score"][:, 1:],
                    "data_mean": labels["transition_mean"][:, 1:],
                    "data_target": labels["transition_target"][:, 1:],
                    "data_std": labels["transition_std"][:, 1:],
                    "knn_distance": labels["knn_distance"][:, 1:],
                }
                for name, value in transition_values.items():
                    array = value.detach().cpu().numpy()
                    if array.ndim >= 3:
                        array = array.reshape(-1, array.shape[-1])
                    else:
                        array = array.reshape(-1)
                    transition_batches.setdefault(name, []).append(array)

            if sequence_index is None:
                representation = representation.reshape(-1, representation.shape[-1])
                factor_batch = factor_batch.reshape(-1, factor_batch.shape[-1])
                class_batch = class_batch.reshape(-1)
            else:
                if not -representation.shape[1] <= sequence_index < representation.shape[1]:
                    raise IndexError(
                        f"sequence_index {sequence_index} is invalid for sequence length "
                        f"{representation.shape[1]}"
                    )
                representation = representation[:, sequence_index]
                factor_batch = factor_batch[:, sequence_index]
                class_batch = class_batch[:, sequence_index]
            if not (len(representation) == len(factor_batch) == len(class_batch)):
                raise ValueError(
                    "representation and label sequence dimensions do not align: "
                    f"{representation.shape}, {factor_batch.shape}, {class_batch.shape}"
                )

            if maximum is not None:
                remaining = maximum - count
                if remaining <= 0:
                    break
                representation = representation[:remaining]
                factor_batch = factor_batch[:remaining]
                class_batch = class_batch[:remaining]
            representations.append(representation)
            factors.append(factor_batch)
            classes.append(class_batch)
            count += len(representation)
            if maximum is not None and count >= maximum:
                break

    if not representations:
        raise ValueError("dataset yielded no representations")
    transition_arrays = {
        name: np.concatenate(batches, axis=0)
        for name, batches in transition_batches.items()
    }
    return (
        np.concatenate(representations, axis=0),
        np.concatenate(factors, axis=0),
        np.concatenate(classes, axis=0),
        transition_arrays,
    )


def _extract_from_checkpoint(
    args: argparse.Namespace,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    import torch
    from torch.utils.data import Subset

    from implicit_nuisance.data import DATASETS

    if args.config is None:
        raise ValueError("--config is required with --checkpoint")
    cfg = _load_config(args.config)
    if args.data_root is not None:
        cfg.data.settings.root = str(args.data_root)
    configured_signal_indices = tuple(
        int(index)
        for index in cfg.data.settings.get(
            "signal_indices", DEFAULT_SIGNAL_INDICES
        )
    )
    state_dimension = int(cfg.method_kwargs.state_dim)
    if cfg.data.dataset not in DATASETS:
        raise ValueError(
            f"unknown dataset {cfg.data.dataset!r}; choose one of {tuple(DATASETS)}"
        )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_local_model(args.checkpoint, cfg, device)
    dataset_type = DATASETS[cfg.data.dataset]
    test_dataset = dataset_type(cfg, split="test")
    requested = args.max_train_samples + args.max_test_samples
    if requested > len(test_dataset):
        raise ValueError(
            "disjoint held-out probe partitions require "
            f"{requested} test sequences, but the dataset has {len(test_dataset)}; "
            "lower --max-train-samples or --max-test-samples"
        )
    permutation = np.random.default_rng(args.seed).permutation(len(test_dataset))
    train_dataset = Subset(test_dataset, permutation[: args.max_train_samples])
    evaluation_dataset = Subset(
        test_dataset,
        permutation[
            args.max_train_samples : args.max_train_samples + args.max_test_samples
        ],
    )
    common = {
        "representation_key": args.representation_key,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": device,
        "maximum": None,
        # The anchor view enumerates the official test split exactly.  The
        # generated later view is used only for conditional diagnostics.
        "sequence_index": 0,
        "collect_transitions": True,
    }
    *train_arrays, train_transitions = _collect_representations(
        model, train_dataset, **common
    )
    *test_arrays, test_transitions = _collect_representations(
        model, evaluation_dataset, **common
    )
    transitions = {
        name: np.concatenate((train_transitions[name], test_transitions[name]), axis=0)
        for name in train_transitions
    }

    model_variances = transitions["model_variance"]
    model_diagonal = model_variances[0]
    if not np.allclose(model_variances, model_diagonal, rtol=0.0, atol=1e-7):
        raise ValueError("model prediction covariance is not fixed across histories")
    data_stds = transitions["data_std"]
    data_diagonal = data_stds[0] ** 2
    if not np.allclose(data_stds, data_stds[0], rtol=0.0, atol=1e-7):
        raise ValueError("dataset transition covariance is not fixed across histories")

    match_error = np.linalg.norm(
        transitions["data_sample"] - transitions["data_target"], axis=1
    )
    gaussian_metrics = {
        "metadata": {
            "signal_indices": list(configured_signal_indices),
            "state_dim": state_dimension,
            "equal_signal_code_dimensions": (
                state_dimension == len(configured_signal_indices)
            ),
            "probe_source": "disjoint_partitions_of_official_test_split",
        },
        "model": evaluate_gaussian_conditional(
            transitions["model_sample"],
            transitions["model_mean"],
            model_diagonal,
        ),
        "data_target_before_lookup": evaluate_gaussian_conditional(
            transitions["data_target"],
            transitions["data_mean"],
            data_diagonal,
        ),
        "data_realized_after_lookup": evaluate_gaussian_conditional(
            transitions["data_sample"],
            transitions["data_mean"],
            data_diagonal,
        ),
        "nearest_neighbor_lookup": {
            "mean_error": float(np.mean(match_error)),
            "p50_error": float(np.quantile(match_error, 0.50)),
            "p95_error": float(np.quantile(match_error, 0.95)),
            "max_error": float(np.max(match_error)),
            "reported_distance_agrees": bool(
                np.allclose(match_error, transitions["knn_distance"], atol=1e-5)
            ),
        },
    }
    return (*train_arrays, *test_arrays), gaussian_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--train-npz",
        type=Path,
        help="training archive containing z, factors, and class arrays",
    )
    source.add_argument("--checkpoint", type=Path, help="local Lightning checkpoint")
    parser.add_argument(
        "--test-npz",
        type=Path,
        help="held-out archive (required with --train-npz)",
    )
    parser.add_argument("--config", type=Path, help="JSON or YAML config for checkpoint mode")
    parser.add_argument("--data-root", type=Path, help="override cfg.data.settings.root")
    parser.add_argument("--z-key", default="z")
    parser.add_argument("--factors-key", default="factors")
    parser.add_argument("--class-key", default="class")
    parser.add_argument("--representation-key", default="inferences")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", help="checkpoint inference device, e.g. cpu or cuda")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=10000,
        help="held-out test sequences used to fit post-hoc probes",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=10000,
        help="disjoint held-out test sequences used to score post-hoc probes",
    )
    parser.add_argument(
        "--signal-indices",
        type=_comma_separated_ints,
        help=(
            "comma-separated raw factor columns transformed into Gaussian signal q; "
            "checkpoint mode derives these from its saved config"
        ),
    )
    parser.add_argument(
        "--signal-transform",
        choices=("normal_score", "identity"),
        default="normal_score",
    )
    parser.add_argument(
        "--factor-names",
        type=_comma_separated_strings,
        default=CAUSAL3DIDENT_FACTOR_NAMES,
        help="comma-separated factor names",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    gaussian_metrics = None
    if args.train_npz is not None:
        if args.test_npz is None:
            raise ValueError("--test-npz is required with --train-npz")
        train_arrays = _load_archive(
            args.train_npz,
            z_key=args.z_key,
            factors_key=args.factors_key,
            class_key=args.class_key,
        )
        test_arrays = _load_archive(
            args.test_npz,
            z_key=args.z_key,
            factors_key=args.factors_key,
            class_key=args.class_key,
        )
        arrays = (*train_arrays, *test_arrays)
    else:
        if args.test_npz is not None:
            raise ValueError("--test-npz cannot be combined with --checkpoint")
        arrays, gaussian_metrics = _extract_from_checkpoint(args)

    if gaussian_metrics is None:
        signal_indices = args.signal_indices or DEFAULT_SIGNAL_INDICES
    else:
        configured_signal_indices = tuple(
            gaussian_metrics["metadata"]["signal_indices"]
        )
        if (
            args.signal_indices is not None
            and tuple(args.signal_indices) != configured_signal_indices
        ):
            raise ValueError(
                "--signal-indices conflicts with the checkpoint config: "
                f"{tuple(args.signal_indices)} != {configured_signal_indices}"
            )
        signal_indices = configured_signal_indices

    train_z, train_factors, train_class, test_z, test_factors, test_class = arrays
    metrics = evaluate_representations(
        train_z,
        train_factors,
        train_class,
        test_z,
        test_factors,
        test_class,
        signal_indices=signal_indices,
        signal_transform=args.signal_transform,
        factor_names=args.factor_names,
    )
    if gaussian_metrics is not None:
        metrics["gaussian_conditional"] = gaussian_metrics
    rendered = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
