#!/usr/bin/env python3
"""Pregenerate action-labelled physical Hopper rollouts as mmap arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from implicit_nuisance.data.dm_control_hopper_rollout import (
    DMControlHopperNuisance,
)
from implicit_nuisance.data.offline_dm_control_hopper_rollout import (
    OfflineDMControlHopperNuisance,
)


_STORED_LABELS = tuple(OfflineDMControlHopperNuisance._EXPECTED_TAILS)


def _config_from_args(args: argparse.Namespace):
    return OmegaConf.create(
        {
            "data": {
                "dataset": "dm_control_hopper_nuisance",
                "settings": {
                    "num_train_sequences": args.train_sequences,
                    "num_test_sequences": args.test_sequences,
                    "sequence_length": args.sequence_length,
                    "image_size": args.image_size,
                    "normalization": "none",
                    "controller_path": str(args.controller_path),
                    "controller_device": args.controller_device,
                    "burn_in_steps": args.burn_in_steps,
                    "frame_skip": args.frame_skip,
                    "exploration_std": args.exploration_std,
                    "dynamics_noise_std": args.dynamics_noise_std,
                    "mechanism_seed": args.mechanism_seed,
                    "seed": args.seed,
                    "continuations_per_initial_state": args.continuations,
                },
            }
        }
    )


def generate_dataset(cfg, output: Path, show_progress: bool = True) -> None:
    """Generate train/test splits and publish the manifest after completion."""

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output path is not empty: {output}. Choose a new path or remove "
            "it explicitly."
        )
    output.mkdir(parents=True, exist_ok=True)

    metadata = None
    for split in ("train", "test"):
        dataset = DMControlHopperNuisance(cfg, split=split)
        split_root = output / split
        split_root.mkdir()
        height, width = dataset.image_size
        sequence_shape = (len(dataset), dataset.sequence_length)
        label_tails = dict(OfflineDMControlHopperNuisance._EXPECTED_TAILS)
        for name in OfflineDMControlHopperNuisance._ACTION_LABELS:
            label_tails[name] = (dataset.frame_skip, dataset.action_dim)
        arrays = {
            "images": np.lib.format.open_memmap(
                split_root / "images.npy",
                mode="w+",
                dtype=np.uint8,
                shape=(*sequence_shape, height, width, 3),
            )
        }
        for name, tail in label_tails.items():
            dtype = np.int64 if name == "background_seed" else np.float32
            arrays[name] = np.lib.format.open_memmap(
                split_root / f"{name}.npy",
                mode="w+",
                dtype=dtype,
                shape=(*sequence_shape, *tail),
            )

        iterator = tqdm(
            range(len(dataset)),
            desc=f"Generating physical Hopper {split}",
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
                .astype(np.uint8, copy=False)
            )
            arrays["images"][index] = images
            for name in _STORED_LABELS:
                arrays[name][index] = labels[name].cpu().numpy()

        for array in arrays.values():
            array.flush()

        current_metadata = {
            "format_version": 4,
            "dataset": dataset._DATASET_NAME,
            "environment": dataset._ENVIRONMENT,
            "task": dataset._TASK_NAME,
            "signal_dim": dataset.signal_dim,
            "signal_layout": {"position": [0, 7]},
            "position_names": [
                "root_height",
                "torso_pitch_sin",
                "torso_pitch_cos",
                "waist",
                "hip",
                "knee",
                "ankle",
            ],
            "physics_state_dim": 14,
            "controller_state_dim": dataset.controller_state_dim,
            "controller_observation_order": [
                "raw_position", "touch", "velocity"
            ],
            "nuisance_dim": dataset.nuisance_dim,
            "action_dim": dataset.action_dim,
            "action_alignment": (
                "actions[t, :] are the ordered commands for the frame t -> "
                "frame t+1 transition"
            ),
            "conditioned_action": "actions",
            "executed_action": "executed_actions",
            "stored_labels": list(_STORED_LABELS),
            "sequence_length": dataset.sequence_length,
            "image_size": list(dataset.image_size),
            "control_timestep": 0.02,
            "frame_skip": dataset.frame_skip,
            "render_timestep": 0.02 * dataset.frame_skip,
            "controller": "LAOM CleanRL PPO hopper-hop expert",
            "controller_checkpoint": str(dataset.controller_path),
            "burn_in_steps": dataset.burn_in_steps,
            "exploration_std": dataset.exploration_std,
            "dynamics_noise_std": dataset.dynamics_noise_std,
            "mechanism_seed": dataset.mechanism_seed,
            "seed": dataset.seed,
            "continuations_per_initial_state": (
                dataset.continuations_per_initial_state
            ),
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
    parser.add_argument("--controller-path", type=Path, required=True)
    parser.add_argument("--controller-device", default="cpu")
    parser.add_argument("--train-sequences", type=int, default=3124)
    parser.add_argument("--test-sequences", type=int, default=312)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--burn-in-steps", type=int, default=25)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--exploration-std", type=float, default=0.10)
    parser.add_argument("--dynamics-noise-std", type=float, default=0.03)
    parser.add_argument("--continuations", type=int, default=4)
    parser.add_argument("--mechanism-seed", type=int, default=1729)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_dataset(_config_from_args(args), args.output)


if __name__ == "__main__":
    main()
