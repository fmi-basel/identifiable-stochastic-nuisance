#!/usr/bin/env python3
"""Generate a slow GIF from a noise-free DM-Control Hopper rollout."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from implicit_nuisance.data.dm_control_hopper_rollout import (
    DMControlHopperNuisance,
)


class HopperGifDataset(DMControlHopperNuisance):
    """Hopper rollout with an optional clean DM-Control render."""

    def __init__(
        self,
        *args,
        no_nuisance: bool = False,
        camera_shake_scale: float = 0.2,
        **kwargs,
    ):
        self.no_nuisance = no_nuisance
        self._CAMERA_POSITION_JITTER_SCALE = camera_shake_scale
        self._CAMERA_FOVY_MINIMUM = 40.0
        self._CAMERA_FOVY_MAXIMUM = 70.0
        super().__init__(*args, **kwargs)

    def _render_frame(
        self, nuisance: np.ndarray, background_seed: int
    ) -> torch.Tensor:
        if not self.no_nuisance:
            return super()._render_frame(nuisance, background_seed)

        assert self._environment is not None
        height, width = self.image_size
        rgb = self._environment.physics.render(
            height=height,
            width=width,
            camera_id=self._CAMERA_NAME,
        )
        frame = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        return (frame - self._normalization_mean) / self._normalization_std


def _config(args: argparse.Namespace):
    return OmegaConf.create(
        {
            "data": {
                "dataset": "dm_control_hopper_nuisance",
                "settings": {
                    "num_train_sequences": 1,
                    "num_test_sequences": 1,
                    "sequence_length": args.frames,
                    "image_size": args.image_size,
                    "normalization": "none",
                    "controller_path": str(args.controller_path),
                    "controller_device": "cpu",
                    "burn_in_steps": args.burn_in_steps,
                    "frame_skip": 1,
                    "exploration_std": 0.0,
                    "dynamics_noise_std": 0.0,
                    "mechanism_seed": args.mechanism_seed,
                    "seed": args.seed,
                    "continuations_per_initial_state": 1,
                },
            }
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("eval/plots/hopper.gif"))
    parser.add_argument(
        "--controller-path",
        type=Path,
        default=Path(
            "../laom/scripts/data_collection/checkpoints/hopper-hop-expert"
        ),
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--burn-in-steps", type=int, default=64)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument(
        "--camera-shake-scale",
        type=float,
        default=0.2,
    )
    parser.add_argument("--mechanism-seed", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--no-nuisance",
        action="store_true",
        help="render the original DM-Control scene without visual nuisance",
    )
    args = parser.parse_args()
    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    if not np.isfinite(args.camera_shake_scale) or args.camera_shake_scale < 0.0:
        parser.error("--camera-shake-scale must be finite and non-negative")
    return args


def main() -> None:
    args = _parse_args()
    dataset = HopperGifDataset(
        _config(args),
        split="test",
        no_nuisance=args.no_nuisance,
        camera_shake_scale=args.camera_shake_scale,
    )
    frames, labels = dataset[0]

    if not np.all(labels["action_noise"].numpy() == 0.0):
        raise RuntimeError("Expected zero actuator noise in the generated rollout")

    images = (
        frames.permute(0, 2, 3, 1)
        .mul(255.0)
        .round()
        .clamp(0.0, 255.0)
        .numpy()
        .astype(np.uint8)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame_duration_ms = round(1000.0 / args.fps)
    imageio.mimsave(args.output, images, duration=frame_duration_ms, loop=0)
    print(
        f"Saved {len(images)} frames to {args.output} at {args.fps:g} fps "
        f"with actuator noise 0 and visual nuisance "
        f"{'disabled' if args.no_nuisance else 'enabled'}."
    )


if __name__ == "__main__":
    main()
