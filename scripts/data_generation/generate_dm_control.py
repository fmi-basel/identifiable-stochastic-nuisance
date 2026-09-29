#!/usr/bin/env python3
"""Pregenerate stochastic DM-Control reacher data as mmap arrays."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from implicit_nuisance.data.dm_control import DMControlReacherNuisance


_STORED_LABELS = (
    "signal",
    "nuisance",
    "background_seed",
    "transition_mean",
    "transition_std",
    "transition_noise",
)


def _config_from_args(args: argparse.Namespace):
    transition_std = (
        math.sqrt(1.0 - args.rho**2)
        if args.transition_std is None
        else args.transition_std
    )
    return OmegaConf.create(
        {
            "data": {
                "dataset": "dm_control_reacher_nuisance",
                "settings": {
                    "environment": "reacher",
                    "num_train_sequences": args.train_sequences,
                    "num_test_sequences": args.test_sequences,
                    "sequence_length": args.sequence_length,
                    "image_size": args.image_size,
                    "normalization": "none",
                    "initial_distribution": args.initial_distribution,
                    "rho": args.rho,
                    "transition_std": transition_std,
                    "rotation_strength": args.rotation_strength,
                    "angle_scale": args.angle_scale,
                    "mechanism_seed": args.mechanism_seed,
                    "seed": args.seed,
                    "stochastic_transitions": False,
                    "continuations_per_initial_state": args.continuations,
                },
            }
        }
    )


def generate_dataset(cfg, output: Path, show_progress: bool = True) -> None:
    """Generate both splits, writing the manifest only after completion."""

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output path is not empty: {output}. Choose a new path or remove it explicitly."
        )
    output.mkdir(parents=True, exist_ok=True)

    metadata = None
    for split in ("train", "test"):
        dataset = DMControlReacherNuisance(cfg, split=split)
        split_root = output / split
        split_root.mkdir()
        height, width = dataset.image_size
        sequence_shape = (len(dataset), dataset.sequence_length)
        arrays = {
            "images": np.lib.format.open_memmap(
                split_root / "images.npy",
                mode="w+",
                dtype=np.uint8,
                shape=(*sequence_shape, height, width, 3),
            ),
            "signal": np.lib.format.open_memmap(
                split_root / "signal.npy",
                mode="w+",
                dtype=np.float32,
                shape=(*sequence_shape, dataset.signal_dim),
            ),
            "nuisance": np.lib.format.open_memmap(
                split_root / "nuisance.npy",
                mode="w+",
                dtype=np.float32,
                shape=(*sequence_shape, dataset.nuisance_dim),
            ),
            "background_seed": np.lib.format.open_memmap(
                split_root / "background_seed.npy",
                mode="w+",
                dtype=np.int64,
                shape=sequence_shape,
            ),
            "transition_mean": np.lib.format.open_memmap(
                split_root / "transition_mean.npy",
                mode="w+",
                dtype=np.float32,
                shape=(*sequence_shape, dataset.signal_dim),
            ),
            "transition_std": np.lib.format.open_memmap(
                split_root / "transition_std.npy",
                mode="w+",
                dtype=np.float32,
                shape=(*sequence_shape, dataset.signal_dim),
            ),
            "transition_noise": np.lib.format.open_memmap(
                split_root / "transition_noise.npy",
                mode="w+",
                dtype=np.float32,
                shape=(*sequence_shape, dataset.signal_dim),
            ),
        }

        iterator = tqdm(
            range(len(dataset)),
            desc=f"Generating {split}",
            unit="sequence",
            disable=not show_progress,
        )
        for index in iterator:
            frames, labels = dataset[index]
            images = (
                frames.permute(0, 2, 3, 1)
                .mul(255.0)
                .round()
                .clamp(0.0, 255.0)
                .cpu()
                .numpy()
            )
            arrays["images"][index] = images.astype(np.uint8, copy=False)
            for name in _STORED_LABELS:
                arrays[name][index] = labels[name].cpu().numpy()

        for array in arrays.values():
            array.flush()

        current_metadata = {
            "format_version": 1,
            "dataset": dataset._DATASET_NAME,
            "environment": dataset._ENVIRONMENT,
            "signal_dim": dataset.signal_dim,
            "nuisance_dim": dataset.nuisance_dim,
            "sequence_length": dataset.sequence_length,
            "image_size": list(dataset.image_size),
            "rho": dataset.rho,
            "transition_std": dataset.transition_std,
            "rotation_strength": dataset.rotation_strength,
            "initial_distribution": dataset.initial_distribution,
            "angle_scale": dataset.angle_scale,
            "mechanism_seed": dataset.mechanism_seed,
            "seed": dataset.seed,
            "continuations_per_initial_state": dataset.continuations_per_initial_state,
            "splits": {},
        }
        if metadata is None:
            metadata = current_metadata
        metadata["splits"][split] = {"num_sequences": len(dataset)}

    assert metadata is not None
    with (output / "metadata.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(metadata, manifest_file, indent=2)
        manifest_file.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-sequences", type=int, default=100000)
    parser.add_argument("--test-sequences", type=int, default=10000)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--continuations", type=int, default=4)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--transition-std", type=float)
    parser.add_argument("--rotation-strength", type=float, default=1.0)
    parser.add_argument(
        "--initial-distribution", choices=("gaussian", "uniform"), default="gaussian"
    )
    parser.add_argument("--angle-scale", type=float, default=2.6)
    parser.add_argument("--mechanism-seed", type=int, default=1729)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_dataset(_config_from_args(args), args.output)


if __name__ == "__main__":
    main()
