from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import omegaconf
import torch
from torch.utils.data import Dataset

from implicit_nuisance.data.dm_control import DMControlReacherNuisance
from implicit_nuisance.utils.misc import omegaconf_select


class DMControlHopperNuisance(Dataset):
    """Physical Hopper rollouts controlled from state and rendered with nuisance.

    ``actions[t]`` contains the logged commands for all ``frame_skip`` control
    steps between rendered frames ``t`` and ``t + 1``.  The simulator receives
    ``executed_actions[t]``, which additionally contains hidden iid actuator
    noise.  Keeping that second perturbation out of the predictor input makes
    the controlled transition stochastic.
    """

    _VALID_SPLITS = ("train", "test")
    _SPLIT_SEEDS = {"train": 0x54524149, "test": 0x54455354}
    _DATASET_NAME = "dm_control_hopper_nuisance"
    _ENVIRONMENT = "hopper"
    _DOMAIN_NAME = "hopper"
    _TASK_NAME = "hop"
    _CAMERA_NAME = "cam0"
    _CAMERA_POSITION_JITTER_SCALE = 0.2
    _CAMERA_FOVY_MINIMUM = 40.0
    _CAMERA_FOVY_MAXIMUM = 70.0
    _LIGHT_NAME = "top"
    _FOREGROUND_GEOMS = ("torso", "nose", "pelvis", "thigh", "calf", "foot")
    _SOURCE_OBSERVATION_KEYS = ("position", "velocity", "touch")
    # LAOM trained through Gymnasium's FlattenObservation, which sorts Dict
    # space keys rather than preserving dm_control's OrderedDict insertion order.
    _CONTROLLER_OBSERVATION_KEYS = ("position", "touch", "velocity")
    signal_dim = 7
    controller_state_dim = 15
    nuisance_dim = 12
    action_dim = 4

    def __init__(self, cfg: omegaconf.DictConfig, split: str = "train"):
        super().__init__()
        cfg = self.add_and_assert_specific_cfg(cfg)
        if split not in self._VALID_SPLITS:
            raise ValueError(f"split must be one of {self._VALID_SPLITS}")

        settings = cfg.data.settings
        self.split = split
        self.sequence_length = int(settings.sequence_length)
        self.num_sequences = int(settings[f"num_{split}_sequences"])
        self.image_size = self._parse_image_size(settings.image_size)
        self.seed = int(settings.seed)
        self.mechanism_seed = int(settings.mechanism_seed)
        self.controller_path = Path(str(settings.controller_path)).expanduser()
        self.controller_device = str(settings.controller_device)
        self.burn_in_steps = int(settings.burn_in_steps)
        self.frame_skip = int(settings.frame_skip)
        self.exploration_std = float(settings.exploration_std)
        self.dynamics_noise_std = float(settings.dynamics_noise_std)
        self.continuations_per_initial_state = int(
            settings.continuations_per_initial_state
        )
        self._normalization_mean, self._normalization_std = (
            DMControlReacherNuisance._parse_normalization(settings.normalization)
        )
        self._validate_settings()

        self._environment = None
        self._controller = None
        self._action_minimum = None
        self._action_maximum = None
        self._base_camera_com_position = None
        self._base_light_position = None
        self._base_light_diffuse = None

    def _validate_settings(self) -> None:
        if self.sequence_length < 2:
            raise ValueError("data.settings.sequence_length must be at least 2")
        if self.num_sequences < 1:
            raise ValueError(f"data.settings.num_{self.split}_sequences must be positive")
        if self.burn_in_steps < 0:
            raise ValueError("data.settings.burn_in_steps must be non-negative")
        if self.frame_skip < 1:
            raise ValueError("data.settings.frame_skip must be positive")
        if self.burn_in_steps + self.frame_skip * self.sequence_length > 1000:
            raise ValueError(
                "burn_in_steps + frame_skip * sequence_length must not exceed "
                "the 1000-step Hopper episode"
            )
        for name, value in (
            ("exploration_std", self.exploration_std),
            ("dynamics_noise_std", self.dynamics_noise_std),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"data.settings.{name} must be finite and non-negative")
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
        self._ensure_rollout_components()
        assert self._environment is not None
        assert self._controller is not None
        assert self._action_minimum is not None
        assert self._action_maximum is not None

        anchor_index = idx // self.continuations_per_initial_state
        anchor_rng = self._rng(anchor_index, 0xA11C)
        rollout_rng = self._rng(idx, 0xB12A)
        time_step = self._reset_environment(anchor_index)

        for _ in range(self.burn_in_steps):
            observation = self._flatten_observation(time_step.observation)
            action = np.clip(
                self._controller.action(observation),
                self._action_minimum,
                self._action_maximum,
            )
            time_step = self._environment.step(action)
            if time_step.last():
                raise RuntimeError("Hopper episode ended during controller burn-in")

        nuisance = np.empty(
            (self.sequence_length, self.nuisance_dim), dtype=np.float32
        )
        nuisance[0] = anchor_rng.normal(size=self.nuisance_dim)
        nuisance[1:] = rollout_rng.normal(
            size=(self.sequence_length - 1, self.nuisance_dim)
        )
        background_seed = np.empty(self.sequence_length, dtype=np.int64)
        background_seed[0] = anchor_rng.integers(0, np.iinfo(np.int32).max)
        background_seed[1:] = rollout_rng.integers(
            0,
            np.iinfo(np.int32).max,
            size=self.sequence_length - 1,
            dtype=np.int64,
        )

        frames = []
        controller_state = np.empty(
            (self.sequence_length, self.controller_state_dim), dtype=np.float32
        )
        policy_actions = np.empty(
            (self.sequence_length, self.frame_skip, self.action_dim),
            dtype=np.float32,
        )
        actions = np.empty_like(policy_actions)
        executed_actions = np.empty_like(policy_actions)
        action_noise = np.empty_like(policy_actions)
        rewards = np.empty(self.sequence_length, dtype=np.float32)

        for time in range(self.sequence_length):
            controller_state[time] = self._flatten_observation(
                time_step.observation
            )
            frames.append(
                self._render_frame(
                    nuisance[time], int(background_seed[time])
                )
            )

            exploration_rng = anchor_rng if time == 0 else rollout_rng
            reward = 0.0
            for substep in range(self.frame_skip):
                substep_state = self._flatten_observation(time_step.observation)
                policy_action = np.asarray(
                    self._controller.action(substep_state), dtype=np.float32
                )
                exploration = exploration_rng.normal(
                    scale=self.exploration_std, size=self.action_dim
                ).astype(np.float32)
                command = np.clip(
                    policy_action + exploration,
                    self._action_minimum,
                    self._action_maximum,
                )
                hidden_noise = rollout_rng.normal(
                    scale=self.dynamics_noise_std, size=self.action_dim
                ).astype(np.float32)
                executed = np.clip(
                    command + hidden_noise,
                    self._action_minimum,
                    self._action_maximum,
                )
                policy_actions[time, substep] = policy_action
                actions[time, substep] = command
                executed_actions[time, substep] = executed
                action_noise[time, substep] = executed - command
                next_time_step = self._environment.step(executed)
                reward += float(next_time_step.reward or 0.0)
                if next_time_step.last():
                    if (
                        time < self.sequence_length - 1
                        or substep < self.frame_skip - 1
                    ):
                        raise RuntimeError(
                            "Hopper episode ended inside a saved sequence"
                        )
                    break
            rewards[time] = reward
            time_step = next_time_step

        state = torch.from_numpy(controller_state)
        raw_position = state[:, :6]
        position = self._circular_position(raw_position)
        touch = state[:, 6:8]
        velocity = state[:, 8:]
        labels = {
            "signal": position,
            "position": position,
            "velocity": velocity,
            "touch": touch,
            "physics_state": torch.cat((position, velocity), dim=-1),
            "controller_state": state,
            "joint_angles": position[:, 3:],
            "nuisance": torch.from_numpy(nuisance),
            "background_seed": torch.from_numpy(background_seed),
            "policy_actions": torch.from_numpy(policy_actions),
            "actions": torch.from_numpy(actions),
            "executed_actions": torch.from_numpy(executed_actions),
            "action_noise": torch.from_numpy(action_noise),
            "rewards": torch.from_numpy(rewards),
        }
        return torch.stack(frames), labels

    @staticmethod
    def _circular_position(raw_position: torch.Tensor) -> torch.Tensor:
        """Replace unbounded torso pitch with sine and cosine coordinates."""

        if raw_position.shape[-1] != 6:
            raise ValueError(
                f"raw Hopper position must have width 6, got {raw_position.shape}"
            )
        pitch = raw_position[..., 1]
        return torch.cat(
            (
                raw_position[..., :1],
                torch.sin(pitch).unsqueeze(-1),
                torch.cos(pitch).unsqueeze(-1),
                raw_position[..., 2:],
            ),
            dim=-1,
        )

    def _reset_environment(self, anchor_index: int):
        assert self._environment is not None
        reset_seed = int(
            np.random.SeedSequence(
                [self.seed, self._SPLIT_SEEDS[self.split], anchor_index, 0xE115]
            ).generate_state(1, dtype=np.uint32)[0]
        )
        self._environment.task._random = np.random.RandomState(reset_seed)
        return self._environment.reset()

    def _rng(self, index: int, stream: int) -> np.random.Generator:
        return np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, self._SPLIT_SEEDS[self.split], int(index), stream]
            )
        )

    def _ensure_rollout_components(self) -> None:
        if self._environment is not None:
            return
        try:
            from dm_control import suite
        except ImportError as error:
            raise ImportError(
                "Physical Hopper generation requires dm_control. Install the "
                "sibling checkout non-editably with `uv add ../dm_control`."
            ) from error
        from implicit_nuisance.data.hopper_policy import HopperPPOPolicy

        environment = suite.load(
            domain_name=self._DOMAIN_NAME,
            task_name=self._TASK_NAME,
            task_kwargs={"random": self.mechanism_seed},
        )
        environment.reset()
        action_spec = environment.action_spec()
        if tuple(action_spec.shape) != (self.action_dim,):
            raise ValueError(
                f"Expected Hopper action shape {(self.action_dim,)}, got {action_spec.shape}"
            )

        self._environment = environment
        self._controller = HopperPPOPolicy(
            self.controller_path, device=self.controller_device
        )
        self._action_minimum = np.asarray(action_spec.minimum, dtype=np.float32)
        self._action_maximum = np.asarray(action_spec.maximum, dtype=np.float32)
        physics = environment.physics
        self._base_camera_com_position = np.array(
            physics.named.model.cam_poscom0[self._CAMERA_NAME], copy=True
        )
        self._base_camera_com_position[1] = -1.8
        self._base_light_position = np.array(
            physics.named.model.light_pos[self._LIGHT_NAME], copy=True
        )
        self._base_light_diffuse = np.array(
            physics.named.model.light_diffuse[self._LIGHT_NAME], copy=True
        )

    @classmethod
    def _flatten_observation(cls, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        if tuple(observation) != cls._SOURCE_OBSERVATION_KEYS:
            raise ValueError(
                f"Expected Hopper observations {cls._SOURCE_OBSERVATION_KEYS}, got "
                f"{tuple(observation)}"
            )
        flattened = np.concatenate(
            [
                np.asarray(observation[key], dtype=np.float32).reshape(-1)
                for key in cls._CONTROLLER_OBSERVATION_KEYS
            ]
        )
        if flattened.shape != (cls.controller_state_dim,):
            raise ValueError(
                f"Expected {cls.controller_state_dim} controller values, got "
                f"{flattened.shape}"
            )
        return flattened

    def _render_frame(
        self, nuisance: np.ndarray, background_seed: int
    ) -> torch.Tensor:
        assert self._environment is not None
        physics = self._environment.physics
        self._apply_appearance(nuisance)
        physics.forward()

        height, width = self.image_size
        rgb = physics.render(height=height, width=width, camera_id=self._CAMERA_NAME)
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
        background = DMControlReacherNuisance._procedural_background(
            height, width, background_seed, float(nuisance[11])
        )
        composited = np.where(foreground[..., None], rgb, background)
        tensor = torch.from_numpy(
            composited.astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        return (tensor - self._normalization_mean) / self._normalization_std

    def _apply_appearance(self, nuisance: np.ndarray) -> None:
        assert self._environment is not None
        assert self._base_camera_com_position is not None
        assert self._base_light_position is not None
        assert self._base_light_diffuse is not None
        physics = self._environment.physics

        body_color = DMControlReacherNuisance._sigmoid(nuisance[0:3])
        accent_color = DMControlReacherNuisance._sigmoid(nuisance[3:6])
        for geom_name in ("torso", "pelvis", "thigh", "calf"):
            physics.named.model.geom_rgba[geom_name, :3] = body_color
        for geom_name in ("nose", "foot"):
            physics.named.model.geom_rgba[geom_name, :3] = accent_color

        physics.named.model.cam_poscom0[self._CAMERA_NAME] = (
            self._base_camera_com_position
        )
        physics.named.model.cam_poscom0[self._CAMERA_NAME, ["x", "z"]] += (
            self._CAMERA_POSITION_JITTER_SCALE * np.tanh(nuisance[6:8])
        )
        fovy_midpoint = 0.5 * (
            self._CAMERA_FOVY_MINIMUM + self._CAMERA_FOVY_MAXIMUM
        )
        fovy_half_range = 0.5 * (
            self._CAMERA_FOVY_MAXIMUM - self._CAMERA_FOVY_MINIMUM
        )
        physics.named.model.cam_fovy[self._CAMERA_NAME] = (
            fovy_midpoint + fovy_half_range * np.tanh(nuisance[8])
        )
        physics.named.model.light_pos[self._LIGHT_NAME] = self._base_light_position
        physics.named.model.light_pos[self._LIGHT_NAME, "z"] *= float(
            np.clip(np.exp(0.15 * nuisance[9]), 0.7, 1.4)
        )
        light_scale = float(np.clip(np.exp(0.12 * nuisance[10]), 0.7, 1.4))
        physics.named.model.light_diffuse[self._LIGHT_NAME] = np.clip(
            self._base_light_diffuse * light_scale, 0.0, 1.0
        )

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
    def add_and_assert_specific_cfg(
        cfg: omegaconf.DictConfig,
    ) -> omegaconf.DictConfig:
        settings = cfg.data.settings
        settings.sequence_length = omegaconf_select(
            cfg, "data.settings.sequence_length", 32
        )
        settings.num_train_sequences = omegaconf_select(
            cfg, "data.settings.num_train_sequences", 100000
        )
        settings.num_test_sequences = omegaconf_select(
            cfg, "data.settings.num_test_sequences", 10000
        )
        settings.image_size = omegaconf_select(cfg, "data.settings.image_size", 128)
        settings.normalization = omegaconf_select(
            cfg, "data.settings.normalization", "imagenet"
        )
        settings.seed = omegaconf_select(cfg, "data.settings.seed", 1)
        settings.mechanism_seed = omegaconf_select(
            cfg, "data.settings.mechanism_seed", 1729
        )
        settings.controller_path = omegaconf_select(
            cfg, "data.settings.controller_path", None
        )
        if settings.controller_path is None:
            raise ValueError("data.settings.controller_path is required")
        settings.controller_device = omegaconf_select(
            cfg, "data.settings.controller_device", "cpu"
        )
        settings.burn_in_steps = omegaconf_select(
            cfg, "data.settings.burn_in_steps", 25
        )
        settings.frame_skip = omegaconf_select(
            cfg, "data.settings.frame_skip", 1
        )
        settings.exploration_std = omegaconf_select(
            cfg, "data.settings.exploration_std", 0.10
        )
        settings.dynamics_noise_std = omegaconf_select(
            cfg, "data.settings.dynamics_noise_std", 0.03
        )
        settings.continuations_per_initial_state = omegaconf_select(
            cfg, "data.settings.continuations_per_initial_state", 4
        )
        return cfg


__all__ = ["DMControlHopperNuisance"]
