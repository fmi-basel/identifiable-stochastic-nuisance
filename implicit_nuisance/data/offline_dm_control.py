from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import omegaconf
import torch
from torch.utils.data import Dataset

from implicit_nuisance.utils.misc import omegaconf_select


class OfflineDMControlReacherNuisance(Dataset):
    """Memory-mapped reader for pregenerated reacher nuisance sequences."""

    _VALID_SPLITS = ("train", "test")
    _STORED_LABELS = (
        "signal",
        "nuisance",
        "background_seed",
        "transition_mean",
        "transition_std",
        "transition_noise",
    )

    def __init__(self, cfg: omegaconf.DictConfig, split: str = "train"):
        super().__init__()
        cfg = self.add_and_assert_specific_cfg(cfg)
        if split not in self._VALID_SPLITS:
            raise ValueError(f"split must be one of {self._VALID_SPLITS}")

        self.root = Path(cfg.data.settings.root).expanduser()
        manifest_path = self.root / "metadata.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing pregenerated dataset manifest: {manifest_path}")
        with manifest_path.open(encoding="utf-8") as manifest_file:
            self.metadata = json.load(manifest_file)
        if self.metadata.get("dataset") != "dm_control_reacher_nuisance":
            raise ValueError("metadata.json is not a DM-Control reacher dataset")
        self.environment = self.metadata.get("environment", "reacher")
        if self.environment != "reacher":
            raise ValueError(f"Unsupported DM-Control environment {self.environment!r}")
        expected_environment = str(cfg.data.settings.environment)
        if expected_environment != "any" and self.environment != expected_environment:
            raise ValueError(
                f"Expected {expected_environment!r} data, found {self.environment!r}"
            )

        split_root = self.root / split
        self.images = self._load_array(split_root / "images.npy")
        self.labels = {
            name: self._load_array(split_root / f"{name}.npy")
            for name in self._STORED_LABELS
        }
        self.angle_scale = float(self.metadata["angle_scale"])
        self.signal_dim = int(self.metadata.get("signal_dim", 2))
        self.nuisance_dim = int(self.metadata.get("nuisance_dim", 12))
        self._normalization_mean, self._normalization_std = self._parse_normalization(
            cfg.data.settings.normalization
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
                f"{split}/images.npy must have shape [N,T,H,W,3], got {self.images.shape}"
            )
        if self.images.dtype != np.uint8:
            raise ValueError(f"{split}/images.npy must have dtype uint8")
        sequence_shape = self.images.shape[:2]
        expected_tails = {
            "signal": (self.signal_dim,),
            "nuisance": (self.nuisance_dim,),
            "background_seed": (),
            "transition_mean": (self.signal_dim,),
            "transition_std": (self.signal_dim,),
            "transition_noise": (self.signal_dim,),
        }
        for name, tail in expected_tails.items():
            expected = (*sequence_shape, *tail)
            if self.labels[name].shape != expected:
                raise ValueError(
                    f"{split}/{name}.npy must have shape {expected}, "
                    f"got {self.labels[name].shape}"
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
        signal = labels["signal"]
        labels.update(
            {
                "signal_normal_score": signal,
                "pos": signal,
                "joint_angles": self._joint_angles(signal),
            }
        )
        return frames, labels

    def _joint_angles(self, signal: torch.Tensor) -> torch.Tensor:
        return self.angle_scale * torch.tanh(signal)

    @staticmethod
    def _parse_normalization(value: object) -> tuple[torch.Tensor, torch.Tensor]:
        if value is None or str(value).lower() in {"none", "zero_one"}:
            mean, std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        elif isinstance(value, str) and value.lower() in {"minus_one_one", "-1_1"}:
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        elif isinstance(value, str) and value.lower() == "imagenet":
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        else:
            raise ValueError("normalization must be none, minus_one_one, or imagenet")
        return (
            torch.tensor(mean, dtype=torch.float32)[None, :, None, None],
            torch.tensor(std, dtype=torch.float32)[None, :, None, None],
        )

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
        cfg.data.settings.environment = omegaconf_select(
            cfg, "data.settings.environment", "any"
        )
        return cfg
