#!/usr/bin/env python3
"""Evaluate an action-conditioned physical Hopper checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from implicit_nuisance.utils.causal3dident_eval import (
    evaluate_gaussian_conditional,
)
from implicit_nuisance.utils.dm_control_hopper_eval import (
    evaluate_hopper_learned_decoders,
    evaluate_hopper_representations,
    plot_decoding_r2,
    plot_nonlinear_decoding_r2,
)


_DATASET = "offline_dm_control_hopper_nuisance"


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
    disallowed_missing = set(incompatible.missing_keys) - {
        "encoder.prediction_variance"
    }
    disallowed_unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith("encoder.variance_predictor.")
    ]
    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            "checkpoint and config do not describe the same model; "
            f"missing keys: {sorted(disallowed_missing)}, "
            f"unexpected keys: {sorted(disallowed_unexpected)}"
        )
    if incompatible.missing_keys:
        print(
            "warning: using the configured fixed covariance for the legacy "
            "checkpoint",
            file=sys.stderr,
        )
    return model.to(device).eval()


def _is_data_root(path: Path) -> bool:
    manifest = path / "metadata.json"
    if not manifest.is_file():
        return False
    try:
        metadata = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    required = (
        "images.npy",
        "position.npy",
        "velocity.npy",
        "touch.npy",
        "nuisance.npy",
        "actions.npy",
        "action_noise.npy",
        "executed_actions.npy",
    )
    return (
        metadata.get("format_version") == 4
        and metadata.get("dataset") == "dm_control_hopper_nuisance"
        and all(
            (path / split / filename).is_file()
            for split in ("train", "test")
            for filename in required
        )
    )


def _find_data_root(explicit: Path | None, configured: Any) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    environment_root = os.environ.get("DM_CONTROL_HOPPER_ROOT")
    if environment_root:
        candidates.append(Path(environment_root).expanduser())
    if configured:
        configured_path = Path(str(configured)).expanduser()
        candidates.append(
            configured_path
            if configured_path.is_absolute()
            else project_root / configured_path
        )
    candidates.extend(
        (
            project_root / "datasets" / "dm_control_hopper",
            project_root.parent / "datasets" / "dm_control_hopper",
        )
    )
    for candidate in dict.fromkeys(candidates):
        if _is_data_root(candidate):
            return candidate.resolve()
    checked = "\n".join(f"  - {path}" for path in dict.fromkeys(candidates))
    raise FileNotFoundError(
        "Could not find the Hopper dataset. Pass --data-root or set "
        "DM_CONTROL_HOPPER_ROOT. Checked:\n" + checked
    )


def _sequence_subset(dataset: Any, maximum: int, seed: int) -> Any:
    from torch.utils.data import Subset

    if maximum < 2:
        raise ValueError("sample limits must be at least two")
    sequence_length = int(dataset.images.shape[1])
    number = min(len(dataset), math.ceil(maximum / sequence_length))
    indices = np.random.default_rng(seed).choice(len(dataset), number, replace=False)
    return Subset(dataset, np.sort(indices))


def _flatten(value: Any) -> np.ndarray:
    array = value.detach().cpu().numpy()
    return array.reshape(-1, array.shape[-1])


def _fixed_diagonal(values: np.ndarray) -> np.ndarray:
    diagonal = values[0]
    if np.any(diagonal <= 0):
        raise ValueError("prediction variance must be positive")
    if not np.allclose(values, diagonal, rtol=0.0, atol=1e-7):
        raise ValueError("prediction variance is not fixed across histories")
    return diagonal


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
    transitions: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        _sequence_subset(dataset, maximum, selection_seed),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )
    frames: dict[str, list[np.ndarray]] = defaultdict(list)
    transition_rows: dict[str, list[np.ndarray]] = defaultdict(list)
    learned_decoders = {
        "position": "signal",
        "velocity": "velocity_hidden",
        "nuisance": "nuisance",
    }
    with torch.inference_mode():
        for images, labels in loader:
            actions = labels["actions"].to(device, non_blocking=True)
            latents = model.encoder(
                images.to(device, non_blocking=True), actions
            )
            if representation_key not in latents:
                raise KeyError(
                    f"representation {representation_key!r} is unavailable; "
                    f"choose one of {tuple(latents)}"
                )
            for name, value in (
                ("z", latents[representation_key]),
                ("position", labels["position"]),
                ("velocity", labels["velocity"]),
                ("touch", labels["touch"]),
                ("nuisance", labels["nuisance"]),
            ):
                frames[name].append(_flatten(value))
            for target_name, decoder_name in learned_decoders.items():
                if decoder_name not in model.decoder.decoders:
                    continue
                decoder_settings = model.decoder.decoder_settings[decoder_name]
                decoder_input = latents[decoder_settings["input_variable"]]
                batch_size_value, sequence_length = decoder_input.shape[:2]
                decoder_input = decoder_input.view(
                    batch_size_value, sequence_length, -1
                )
                prediction = model.decoder.decoders[decoder_name](decoder_input)
                frames[f"predicted_{target_name}"].append(_flatten(prediction))

            if not transitions:
                continue
            sequence_length = images.shape[1]
            if sequence_length < 2:
                raise ValueError("controlled-transition evaluation requires T >= 2")
            estimates = latents["estimates"][:, :-1]
            command_actions = actions[:, :-1]
            transition_action_shape = command_actions.shape[2:]
            flat_actions = command_actions.reshape(-1, *transition_action_shape)
            permuted_actions = torch.roll(
                flat_actions, shifts=1, dims=0
            ).reshape_as(command_actions)
            encoded_permuted_actions = model.encoder.encode_actions(
                permuted_actions, estimates
            )
            permuted_means = model.encoder.predictor(
                torch.cat((estimates, encoded_permuted_actions), dim=-1)
            )
            transition_rows["samples"].append(_flatten(latents["inferences"][:, 1:]))
            transition_rows["means"].append(_flatten(latents["predictions"][:, :-1]))
            transition_rows["permuted_means"].append(_flatten(permuted_means))
            transition_rows["variances"].append(
                _flatten(latents["predictions_variances"][:, :-1])
            )
            transition_rows["commands"].append(_flatten(labels["actions"][:, :-1]))
            transition_rows["executed"].append(
                _flatten(labels["executed_actions"][:, :-1])
            )
            transition_rows["action_noise"].append(
                _flatten(labels["action_noise"][:, :-1])
            )

    if not frames:
        raise ValueError("dataset yielded no samples")
    arrays = {name: np.concatenate(values) for name, values in frames.items()}
    keep = np.random.default_rng(selection_seed + 1009).permutation(
        len(arrays["z"])
    )[:maximum]
    arrays = {name: values[keep] for name, values in arrays.items()}
    if not transitions:
        return arrays, None
    collected_transitions = {
        name: np.concatenate(values) for name, values in transition_rows.items()
    }
    collected_transitions["covariance"] = _fixed_diagonal(
        collected_transitions.pop("variances")
    )
    return arrays, collected_transitions


def _to_rgb(images: Any, normalization: Any) -> np.ndarray:
    import torch

    value = images.detach().cpu().float()
    normalized = str(normalization).lower()
    if normalized == "imagenet":
        mean = torch.tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = torch.tensor((0.229, 0.224, 0.225))[None, :, None, None]
        value = value * std + mean
    elif normalized in {"minus_one_one", "-1_1"}:
        value = value * 0.5 + 0.5
    return value.clamp(0, 1).permute(0, 2, 3, 1).numpy()


def _plot_reconstructions(
    model: Any,
    dataset: Any,
    normalization: Any,
    output: Path,
    *,
    sequence_index: int,
    times: tuple[int, ...],
    device: str,
) -> None:
    import matplotlib.pyplot as plt
    import torch

    images, labels = dataset[sequence_index]
    valid_times = tuple(time for time in times if 0 <= time < len(images))
    if not valid_times:
        raise ValueError("none of the requested reconstruction times are in the sequence")
    with torch.inference_mode():
        result = model(
            images[None].to(device), labels["actions"][None].to(device)
        )
    if "image" not in result["target_estimates"]:
        raise KeyError("the checkpoint has no image decoder")
    original = _to_rgb(images[list(valid_times)], normalization)
    reconstructed = _to_rgb(
        result["target_estimates"]["image"][0, list(valid_times)], normalization
    )
    figure, axes = plt.subplots(
        2,
        len(valid_times),
        figsize=(2.8 * len(valid_times), 5.0),
        squeeze=False,
    )
    for row, (label, row_images) in enumerate(
        (("Input", original), ("Reconstruction", reconstructed))
    ):
        for column, (time, image) in enumerate(zip(valid_times, row_images, strict=True)):
            axes[row, column].imshow(image)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(f"t = {time}")
            if column == 0:
                axes[row, column].text(
                    -0.04,
                    0.5,
                    label,
                    transform=axes[row, column].transAxes,
                    ha="right",
                    va="center",
                )
    figure.suptitle("Physical Hopper reconstruction")
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--representation-key", default="estimates")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", help="checkpoint inference device, e.g. cpu or cuda")
    parser.add_argument("--max-train-samples", type=int, default=10000)
    parser.add_argument("--max-test-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    parser.add_argument("--plot-dir", type=Path, help="write R2 and reconstruction PNGs here")
    parser.add_argument("--reconstruction-sequence", type=int, default=0)
    parser.add_argument(
        "--reconstruction-times", type=int, nargs="*", default=(0, 4, 8, 16, 31)
    )
    return parser.parse_args()


def evaluate_checkpoint(
    checkpoint: Path,
    cfg: Any,
    data_root: Path,
    *,
    representation_key: str = "estimates",
    batch_size: int = 16,
    num_workers: int = 4,
    device: str = "cpu",
    max_train_samples: int = 10000,
    max_test_samples: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate one checkpoint; used by both the CLI and notebook."""

    from implicit_nuisance.data import DATASETS

    supported_methods = ("infoLDMAction", "nextObservation")
    if cfg.method not in supported_methods:
        raise ValueError(
            f"this evaluator requires method in {supported_methods}"
        )
    if cfg.data.dataset != _DATASET:
        raise ValueError(f"this evaluator requires data.dataset={_DATASET!r}")
    cfg.data.settings.root = str(data_root)
    cfg.device = device
    model = _load_model(checkpoint, cfg, device)
    dataset_type = DATASETS[cfg.data.dataset]
    train_dataset = dataset_type(cfg, split="train")
    test_dataset = dataset_type(cfg, split="test")
    train, _ = _collect(
        model,
        train_dataset,
        maximum=max_train_samples,
        selection_seed=seed,
        representation_key=representation_key,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        transitions=False,
    )
    test, transition = _collect(
        model,
        test_dataset,
        maximum=max_test_samples,
        selection_seed=seed + 1,
        representation_key=representation_key,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        transitions=True,
    )
    assert transition is not None
    metrics = evaluate_hopper_representations(
        train["z"],
        train["position"],
        train["velocity"],
        train["touch"],
        train["nuisance"],
        test["z"],
        test["position"],
        test["velocity"],
        test["touch"],
        test["nuisance"],
    )
    prediction_keys = (
        "predicted_position",
        "predicted_velocity",
        "predicted_nuisance",
    )
    missing_predictions = [key for key in prediction_keys if key not in test]
    if missing_predictions:
        raise KeyError(
            "checkpoint is missing learned Hopper decoders: "
            + ", ".join(missing_predictions)
        )
    metrics["nonlinear_decoding"] = evaluate_hopper_learned_decoders(
        test["position"],
        test["predicted_position"],
        test["velocity"],
        test["predicted_velocity"],
        test["nuisance"],
        test["predicted_nuisance"],
    )
    correct = evaluate_gaussian_conditional(
        transition["samples"], transition["means"], transition["covariance"]
    )
    permuted = evaluate_gaussian_conditional(
        transition["samples"],
        transition["permuted_means"],
        transition["covariance"],
    )
    metrics["controlled_transition"] = {
        "correct_action": correct,
        "permuted_action": permuted,
        "permuted_minus_correct_mean_squared_error": float(
            permuted["mean_squared_error"] - correct["mean_squared_error"]
        ),
        "permuted_minus_correct_mean_gaussian_nll_per_dimension": float(
            permuted["mean_gaussian_nll_per_dimension"]
            - correct["mean_gaussian_nll_per_dimension"]
        ),
    }
    action_noise = transition["action_noise"]
    metrics["action_diagnostics"] = {
        "command_mean": transition["commands"].mean(axis=0).tolist(),
        "command_std": transition["commands"].std(axis=0).tolist(),
        "hidden_noise_mean": action_noise.mean(axis=0).tolist(),
        "hidden_noise_std": action_noise.std(axis=0).tolist(),
        "executed_action_clipped_fraction": float(
            np.mean(np.abs(transition["executed"]) >= 1.0 - 1e-7)
        ),
    }
    metrics["metadata"].update(
        {
            "checkpoint": str(checkpoint.resolve()),
            "data_root": str(data_root),
            "split": "train_probe_fit_test_probe_score",
            "linear_probe_representation": representation_key,
            "signal_dim": 7,
            "action_dim": 4,
            "action_encoder_hidden_dim": int(
                cfg.method_kwargs.action_encoder_hidden_dim
            ),
            "frame_skip": int(test_dataset.frame_skip),
            "nuisance_dim": 12,
        }
    )
    return metrics


def main() -> None:
    import torch

    from implicit_nuisance.data import DATASETS

    args = _parse_args()
    cfg = _load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = _find_data_root(args.data_root, cfg.data.settings.root)
    metrics = evaluate_checkpoint(
        args.checkpoint,
        cfg,
        data_root,
        representation_key=args.representation_key,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
    )
    metrics["metadata"]["config"] = str(args.config.resolve())
    if args.plot_dir is not None:
        cfg.data.settings.root = str(data_root)
        cfg.device = device
        model = _load_model(args.checkpoint, cfg, device)
        test_dataset = DATASETS[cfg.data.dataset](cfg, split="test")
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        decoding_path = args.plot_dir / "decoding_r2.png"
        nonlinear_decoding_path = args.plot_dir / "nonlinear_decoding_r2.png"
        reconstruction_path = args.plot_dir / "reconstructions.png"
        plot_decoding_r2(metrics, decoding_path)
        plot_nonlinear_decoding_r2(metrics, nonlinear_decoding_path)
        _plot_reconstructions(
            model,
            test_dataset,
            cfg.data.settings.normalization,
            reconstruction_path,
            sequence_index=args.reconstruction_sequence,
            times=tuple(args.reconstruction_times),
            device=device,
        )
        metrics["plots"] = {
            "decoding_r2": str(decoding_path.resolve()),
            "nonlinear_decoding_r2": str(nonlinear_decoding_path.resolve()),
            "reconstructions": str(reconstruction_path.resolve()),
        }
    rendered = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
