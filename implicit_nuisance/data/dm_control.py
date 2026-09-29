from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from implicit_nuisance.utils.misc import omegaconf_select


class DMControlReacherNuisance(Dataset):
    """Stochastic reacher videos with fresh, view-private visual nuisance.

    A two-dimensional latent signal follows an exact fixed-covariance Gaussian
    transition. MuJoCo is used only to render the signal after mapping it to the
    two reacher joint angles; its nonlinear physics does not alter the source
    conditional. Appearance, target position, camera translation, lighting,
    and a procedural background are independently resampled for every frame.
    """

    _VALID_SPLITS = ("train", "test")
    _INITIAL_DISTRIBUTIONS = ("gaussian", "uniform")
    _SPLIT_SEEDS = {"train": 0x54524149, "test": 0x54455354}
    _NUISANCE_DIM = 12
    _SIGNAL_DIM = 2
    _DATASET_NAME = "dm_control_reacher_nuisance"
    _ENVIRONMENT = "reacher"
    _DOMAIN_NAME = "reacher"
    _TASK_NAME = "easy"
    _CAMERA_NAME = "fixed"
    _CAMERA_JITTER_AXES = ("x", "y")
    _CAMERA_JITTER_SCALE = 0.025
    _LIGHT_NAME = "light"
    _FOREGROUND_GEOMS = ("root", "arm", "hand", "finger", "target")

    def __init__(self, cfg: omegaconf.DictConfig, split: str = "train"):
        super().__init__()
        cfg = self.add_and_assert_specific_cfg(cfg)
        if split not in self._VALID_SPLITS:
            raise ValueError(f"split must be one of {self._VALID_SPLITS}")

        settings = cfg.data.settings
        self.split = split
        self.sequence_length = int(settings.sequence_length)
        self.num_sequences = int(settings[f"num_{split}_sequences"])
        self.rho = float(settings.rho)
        self.transition_std = float(settings.transition_std)
        self.rotation_strength = float(settings.rotation_strength)
        self.initial_distribution = str(settings.initial_distribution)
        self.angle_scale = float(settings.angle_scale)
        self.signal_dim = self._SIGNAL_DIM
        self.nuisance_dim = self._NUISANCE_DIM
        self.image_size = self._parse_image_size(settings.image_size)
        self.seed = int(settings.seed)
        self.mechanism_seed = int(settings.mechanism_seed)
        self.continuations_per_initial_state = int(
            settings.continuations_per_initial_state
        )
        self.stochastic_transitions = split == "train" and bool(
            settings.stochastic_transitions
        )
        self._normalization_mean, self._normalization_std = self._parse_normalization(
            settings.normalization
        )

        self._validate_settings()
        mechanism_rng = np.random.default_rng(self.mechanism_seed)
        random_rotation, _ = np.linalg.qr(
            mechanism_rng.normal(size=(self.signal_dim, self.signal_dim))
        )
        prediction_matrix = (
            self.rotation_strength * random_rotation
            + (1.0 - self.rotation_strength) * np.eye(self.signal_dim)
        )
        self.prediction_matrix = torch.from_numpy(
            prediction_matrix.astype(np.float32)
        )

        self._generators: dict[int, np.random.Generator] = {}
        self._physics = None
        self._base_camera_position = None
        self._base_light_position = None

    def _validate_settings(self) -> None:
        if self.sequence_length < 2:
            raise ValueError("data.settings.sequence_length must be at least 2")
        if self.num_sequences < 1:
            raise ValueError(f"data.settings.num_{self.split}_sequences must be positive")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("data.settings.rho must be strictly between -1 and 1")
        if not np.isfinite(self.transition_std) or self.transition_std <= 0.0:
            raise ValueError("data.settings.transition_std must be finite and positive")
        if not 0.0 <= self.rotation_strength <= 1.0:
            raise ValueError("data.settings.rotation_strength must lie in [0, 1]")
        if self.initial_distribution not in self._INITIAL_DISTRIBUTIONS:
            raise ValueError(
                "data.settings.initial_distribution must be one of "
                f"{self._INITIAL_DISTRIBUTIONS}"
            )
        if not np.isfinite(self.angle_scale) or self.angle_scale <= 0.0:
            raise ValueError("data.settings.angle_scale must be finite and positive")
        if self.continuations_per_initial_state < 1:
            raise ValueError(
                "data.settings.continuations_per_initial_state must be positive"
            )
        if self.num_sequences % self.continuations_per_initial_state:
            raise ValueError(
                f"data.settings.num_{self.split}_sequences must be divisible by "
                "continuations_per_initial_state"
            )

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        rng = self._rng(idx)
        anchor_rng = self._anchor_rng(idx)

        signal = np.empty(
            (self.sequence_length, self.signal_dim), dtype=np.float32
        )
        transition_mean = np.zeros_like(signal)
        transition_noise = np.zeros_like(signal)
        if self.initial_distribution == "gaussian":
            signal[0] = anchor_rng.normal(size=self.signal_dim)
        else:
            signal[0] = anchor_rng.uniform(0.0, 1.0, size=self.signal_dim)

        rotation = self.prediction_matrix.numpy()
        for time in range(1, self.sequence_length):
            mean = self.rho * (rotation @ signal[time - 1])
            noise = rng.normal(
                scale=self.transition_std, size=self.signal_dim
            ).astype(np.float32)
            transition_mean[time] = mean
            transition_noise[time] = noise
            signal[time] = mean + noise

        nuisance = np.empty(
            (self.sequence_length, self._NUISANCE_DIM), dtype=np.float32
        )
        nuisance[0] = anchor_rng.normal(size=self._NUISANCE_DIM)
        nuisance[1:] = rng.normal(
            size=(self.sequence_length - 1, self._NUISANCE_DIM)
        )
        background_seed = np.empty(self.sequence_length, dtype=np.int64)
        background_seed[0] = anchor_rng.integers(0, np.iinfo(np.int32).max)
        background_seed[1:] = rng.integers(
            0,
            np.iinfo(np.int32).max,
            size=self.sequence_length - 1,
            dtype=np.int64,
        )
        frames = torch.stack(
            [
                self._render_frame(
                    signal[time], nuisance[time], int(background_seed[time])
                )
                for time in range(self.sequence_length)
            ]
        )

        signal_tensor = torch.from_numpy(signal)
        labels = {
            "signal": signal_tensor,
            "signal_normal_score": signal_tensor,
            "pos": signal_tensor,
            "joint_angles": self._joint_angles(signal_tensor),
            "nuisance": torch.from_numpy(nuisance),
            "background_seed": torch.from_numpy(background_seed),
            "transition_mean": torch.from_numpy(transition_mean),
            "transition_std": torch.full_like(signal_tensor, self.transition_std),
            "transition_noise": torch.from_numpy(transition_noise),
        }
        return frames, labels

    def _anchor_rng(self, idx: int) -> np.random.Generator:
        anchor_index = int(idx) // self.continuations_per_initial_state
        return np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, self._SPLIT_SEEDS[self.split], anchor_index, 0xA11C]
            )
        )

    def _rng(self, idx: int) -> np.random.Generator:
        if not self.stochastic_transitions:
            return np.random.default_rng(
                np.random.SeedSequence(
                    [self.seed, self._SPLIT_SEEDS[self.split], int(idx)]
                )
            )

        worker = torch.utils.data.get_worker_info()
        worker_id = -1 if worker is None else int(worker.id)
        if worker_id not in self._generators:
            worker_seed = self.seed if worker is None else int(worker.seed)
            self._generators[worker_id] = np.random.default_rng(
                np.random.SeedSequence(
                    [worker_seed, self._SPLIT_SEEDS[self.split], 0xD0C0]
                )
            )
        return self._generators[worker_id]

    def _ensure_renderer(self) -> None:
        if self._physics is not None:
            return
        try:
            from dm_control import suite
        except ImportError as error:
            raise ImportError(
                "dm_control_reacher requires dm_control. Install the sibling "
                "checkout non-editably with `uv add ../dm_control`."
            ) from error

        environment = suite.load(
            domain_name=self._DOMAIN_NAME,
            task_name=self._TASK_NAME,
            task_kwargs={"random": self.mechanism_seed},
        )
        environment.reset()
        self._physics = environment.physics
        self._base_camera_position = np.array(
            self._physics.named.model.cam_pos[self._CAMERA_NAME], copy=True
        )
        self._base_light_position = np.array(
            self._physics.named.model.light_pos[self._LIGHT_NAME], copy=True
        )

    def _render_frame(
        self, signal: np.ndarray, nuisance: np.ndarray, background_seed: int
    ) -> torch.Tensor:
        self._ensure_renderer()
        physics = self._physics
        assert physics is not None

        self._apply_pose_and_appearance(signal, nuisance)

        physics.named.model.cam_pos[self._CAMERA_NAME] = self._base_camera_position
        physics.named.model.cam_pos[
            self._CAMERA_NAME, list(self._CAMERA_JITTER_AXES)
        ] += (
            self._CAMERA_JITTER_SCALE * np.tanh(nuisance[8:10])
        )
        physics.named.model.light_pos[self._LIGHT_NAME] = self._base_light_position
        physics.named.model.light_pos[self._LIGHT_NAME, "z"] *= float(
            np.clip(np.exp(0.15 * nuisance[10]), 0.7, 1.4)
        )
        physics.forward()

        height, width = self.image_size
        rgb = physics.render(
            height=height, width=width, camera_id=self._CAMERA_NAME
        )
        segmentation = physics.render(
            height=height,
            width=width,
            camera_id=self._CAMERA_NAME,
            segmentation=True,
        )
        foreground_ids = [
            physics.model.name2id(name, "geom") for name in self._FOREGROUND_GEOMS
        ]
        foreground = np.isin(segmentation[..., 0], foreground_ids)
        background = self._procedural_background(
            height, width, background_seed, nuisance[11]
        )
        composited = np.where(foreground[..., None], rgb, background)
        tensor = torch.from_numpy(composited.astype(np.float32) / 255.0).permute(2, 0, 1)
        return (tensor - self._normalization_mean) / self._normalization_std

    def _apply_pose_and_appearance(
        self, signal: np.ndarray, nuisance: np.ndarray
    ) -> None:
        physics = self._physics
        assert physics is not None
        physics.named.data.qpos[["shoulder", "wrist"]] = (
            self.angle_scale * np.tanh(signal)
        )
        physics.named.data.qvel[["shoulder", "wrist"]] = 0.0
        physics.named.model.geom_pos["target", ["x", "y"]] = (
            0.18 * np.tanh(nuisance[6:8])
        )

        arm_color = self._sigmoid(nuisance[0:3])
        target_color = self._sigmoid(nuisance[3:6])
        for geom_name in ("root", "arm", "hand", "finger"):
            physics.named.model.geom_rgba[geom_name, :3] = arm_color
        physics.named.model.geom_rgba["target", :3] = target_color

    def _joint_angles(self, signal: torch.Tensor) -> torch.Tensor:
        return self.angle_scale * torch.tanh(signal)

    @staticmethod
    def _procedural_background(
        height: int, width: int, seed: int, style: float
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        fields = []
        for grid_size in (4, 12, 32):
            field = torch.from_numpy(
                rng.normal(size=(1, 3, grid_size, grid_size)).astype(np.float32)
            )
            fields.append(
                F.interpolate(
                    field,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[0]
            )
        texture = fields[0] + 0.55 * fields[1] + 0.2 * fields[2]
        tint = torch.from_numpy(rng.normal(scale=0.7, size=(3, 1, 1)).astype(np.float32))
        contrast = float(np.clip(np.exp(0.2 * style), 0.65, 1.6))
        texture = torch.sigmoid(contrast * texture + tint)
        return (
            texture.mul(255.0)
            .clamp(0.0, 255.0)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .numpy()
        )

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-values))

    @staticmethod
    def _parse_image_size(value: object) -> tuple[int, int]:
        if isinstance(value, (int, np.integer)):
            image_size = (int(value), int(value))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 2:
                raise ValueError("image_size must be an integer or [height, width]")
            image_size = (int(value[0]), int(value[1]))
        else:
            raise ValueError("image_size must be an integer or [height, width]")
        if min(image_size) < 1:
            raise ValueError("image_size dimensions must be positive")
        return image_size

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
            torch.tensor(mean, dtype=torch.float32)[:, None, None],
            torch.tensor(std, dtype=torch.float32)[:, None, None],
        )

    @staticmethod
    def add_and_assert_specific_cfg(
        cfg: omegaconf.DictConfig,
    ) -> omegaconf.DictConfig:
        settings = cfg.data.settings
        settings.sequence_length = omegaconf_select(
            cfg, "data.settings.sequence_length", 8
        )
        settings.num_train_sequences = omegaconf_select(
            cfg, "data.settings.num_train_sequences", 100000
        )
        settings.num_test_sequences = omegaconf_select(
            cfg, "data.settings.num_test_sequences", 10000
        )
        settings.rho = omegaconf_select(cfg, "data.settings.rho", 0.9)
        default_std = math.sqrt(1.0 - float(settings.rho) ** 2)
        settings.transition_std = omegaconf_select(
            cfg, "data.settings.transition_std", default_std
        )
        settings.rotation_strength = omegaconf_select(
            cfg, "data.settings.rotation_strength", 1.0
        )
        settings.initial_distribution = omegaconf_select(
            cfg, "data.settings.initial_distribution", "gaussian"
        )
        settings.angle_scale = omegaconf_select(
            cfg, "data.settings.angle_scale", 2.6
        )
        settings.image_size = omegaconf_select(cfg, "data.settings.image_size", 128)
        settings.normalization = omegaconf_select(
            cfg, "data.settings.normalization", "imagenet"
        )
        settings.seed = omegaconf_select(cfg, "data.settings.seed", 1)
        settings.mechanism_seed = omegaconf_select(
            cfg, "data.settings.mechanism_seed", 1729
        )
        settings.stochastic_transitions = omegaconf_select(
            cfg, "data.settings.stochastic_transitions", True
        )
        settings.continuations_per_initial_state = omegaconf_select(
            cfg, "data.settings.continuations_per_initial_state", 1
        )
        return cfg
