from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import omegaconf
import torch
from torch.utils.data import Dataset

from implicit_nuisance.data.offline_dm_control import (
    OfflineDMControlReacherNuisance,
)
from implicit_nuisance.utils.misc import omegaconf_select


class OfflineDMControlHopperNuisance(Dataset):
    """Memory-mapped reader for physical, action-labelled Hopper rollouts."""

    _VALID_SPLITS = ("train", "test")
    _DATASET_NAME = "dm_control_hopper_nuisance"
    _ACTION_LABELS = (
        "policy_actions",
        "actions",
        "executed_actions",
        "action_noise",
    )
    _EXPECTED_TAILS = {
        "signal": (7,),
        "position": (7,),
        "velocity": (7,),
        "touch": (2,),
        "physics_state": (14,),
        "controller_state": (15,),
        "nuisance": (12,),
        "background_seed": (),
        "policy_actions": (4,),
        "actions": (4,),
        "executed_actions": (4,),
        "action_noise": (4,),
        "rewards": (),
    }

    def __init__(self, cfg: omegaconf.DictConfig, split: str = "train"):
        super().__init__()
        cfg = self.add_and_assert_specific_cfg(cfg)
        if split not in self._VALID_SPLITS:
            raise ValueError(f"split must be one of {self._VALID_SPLITS}")

        self.root = Path(cfg.data.settings.root).expanduser()
        manifest_path = self.root / "metadata.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing pregenerated dataset manifest: {manifest_path}"
            )
        with manifest_path.open(encoding="utf-8") as manifest_file:
            self.metadata = json.load(manifest_file)
        self.format_version = int(self.metadata.get("format_version", -1))
        if self.format_version != 4:
            raise ValueError("Physical Hopper data requires format_version 4")
        if self.metadata.get("dataset") != self._DATASET_NAME:
            raise ValueError("metadata.json is not a physical Hopper dataset")
        if tuple(self.metadata.get("stored_labels", ())) != tuple(
            self._EXPECTED_TAILS
        ):
            raise ValueError("metadata.json has an unexpected stored_labels schema")

        split_root = self.root / split
        self.images = self._load_array(split_root / "images.npy")
        self.labels = {
            name: self._load_array(split_root / f"{name}.npy")
            for name in self._EXPECTED_TAILS
        }
        self.signal_dim = 7
        self.controller_state_dim = 15
        self.nuisance_dim = 12
        self.action_dim = 4
        self.frame_skip = int(self.metadata.get("frame_skip", 1))
        if self.frame_skip < 1:
            raise ValueError("metadata frame_skip must be positive")
        self._expected_tails = dict(self._EXPECTED_TAILS)
        for name in self._ACTION_LABELS:
            self._expected_tails[name] = (self.frame_skip, self.action_dim)
        self._normalization_mean, self._normalization_std = (
            OfflineDMControlReacherNuisance._parse_normalization(
                cfg.data.settings.normalization
            )
        )
        self._validate_arrays(split)

    @staticmethod
    def _load_array(path: Path) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"Missing pregenerated dataset array: {path}")
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def _validate_arrays(self, split: str) -> None:
        if self.images.ndim != 5 or self.images.shape[-1] != 3:
            raise ValueError(
                f"{split}/images.npy must have shape [N,T,H,W,3], got "
                f"{self.images.shape}"
            )
        if self.images.dtype != np.uint8:
            raise ValueError(f"{split}/images.npy must have dtype uint8")
        sequence_shape = self.images.shape[:2]
        for name, tail in self._expected_tails.items():
            expected = (*sequence_shape, *tail)
            if self.labels[name].shape != expected:
                raise ValueError(
                    f"{split}/{name}.npy must have shape {expected}, got "
                    f"{self.labels[name].shape}"
                )

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        image_array = np.array(self.images[idx], dtype=np.float32, copy=True) / 255.0
        frames = torch.from_numpy(image_array).permute(0, 3, 1, 2)
        frames = (frames - self._normalization_mean) / self._normalization_std
        labels = {
            name: torch.from_numpy(np.array(values[idx], copy=True))
            for name, values in self.labels.items()
        }
        labels["signal_normal_score"] = labels["signal"]
        labels["pos"] = labels["position"]
        labels["joint_angles"] = labels["position"][:, 3:]
        return frames, labels

    @staticmethod
    def add_and_assert_specific_cfg(
        cfg: omegaconf.DictConfig,
    ) -> omegaconf.DictConfig:
        root = omegaconf_select(cfg, "data.settings.root", None)
        if root is None:
            raise ValueError("data.settings.root is required for the offline dataset")
        cfg.data.settings.normalization = omegaconf_select(
            cfg, "data.settings.normalization", "imagenet"
        )
        return cfg


__all__ = ["OfflineDMControlHopperNuisance"]
