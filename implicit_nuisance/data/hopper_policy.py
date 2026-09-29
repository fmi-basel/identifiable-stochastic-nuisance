"""Standalone loader for the LAOM proprioceptive Hopper controller.

The checkpoint was produced by LAOM's CleanRL-derived PPO collector.  Its
greedy policy normalizes the flattened DM-Control observation, evaluates a
two-hidden-layer actor, and clips the resulting linear action mean to the
environment's ``[-1, 1]`` action bounds.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn


OBSERVATION_DIM = 15
ACTION_DIM = 4
_HIDDEN_DIM = 512
_OBSERVATION_EPSILON = 1e-8

_REQUIRED_STATE_SHAPES = {
    "obs_rms.mean": (OBSERVATION_DIM,),
    "obs_rms.var": (OBSERVATION_DIM,),
    "obs_rms.count": (),
    "actor_mean.0.weight": (_HIDDEN_DIM, OBSERVATION_DIM),
    "actor_mean.0.bias": (_HIDDEN_DIM,),
    "actor_mean.2.weight": (_HIDDEN_DIM, _HIDDEN_DIM),
    "actor_mean.2.bias": (_HIDDEN_DIM,),
    "actor_mean.4.weight": (ACTION_DIM, _HIDDEN_DIM),
    "actor_mean.4.bias": (ACTION_DIM,),
}


class _ObservationNormalizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(OBSERVATION_DIM))
        self.register_buffer("var", torch.zeros(OBSERVATION_DIM))
        self.register_buffer("count", torch.tensor(0.0))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = (observation - self.mean) / torch.sqrt(
            self.var + _OBSERVATION_EPSILON
        )
        return normalized.clamp(-10.0, 10.0)


class _LAOMHopperActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.obs_rms = _ObservationNormalizer()
        self.actor_mean = nn.Sequential(
            nn.Linear(OBSERVATION_DIM, _HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(_HIDDEN_DIM, _HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(_HIDDEN_DIM, ACTION_DIM),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # The source policy has no final tanh. Its ClipAction environment wrapper
        # clips this linear Gaussian mean after inference.
        return self.actor_mean(self.obs_rms(observation))


class HopperPPOPolicy:
    """Greedy state controller loaded from a LAOM Hopper PPO checkpoint.

    Args:
        checkpoint_path: Either the checkpoint file itself or its directory. A
            directory must contain ``checkpoint.pt``.
        device: Torch device used for inference.

    The controller expects LAOM/Gymnasium's flattened 15-dimensional
    observation order (DM-Control Hopper ``position``, ``touch``, then
    ``velocity``). Calling the controller returns a clipped ``float32`` NumPy
    action with shape ``(4,)``.
    """

    observation_dim = OBSERVATION_DIM
    action_dim = ACTION_DIM

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> None:
        self.checkpoint_path = self._resolve_checkpoint(checkpoint_path)
        self.device = torch.device(device)

        loaded = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(loaded, Mapping):
            raise ValueError("LAOM checkpoint must contain a state-dict mapping")
        self._validate_state_dict(loaded)

        actor_state = {name: loaded[name] for name in _REQUIRED_STATE_SHAPES}
        self._actor = _LAOMHopperActor()
        self._actor.load_state_dict(actor_state, strict=True)
        self._actor.to(self.device).eval()

    @staticmethod
    def _resolve_checkpoint(checkpoint: str | Path) -> Path:
        path = Path(checkpoint).expanduser()
        if path.is_dir():
            path = path / "checkpoint.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing LAOM Hopper checkpoint: {path}")
        return path

    @staticmethod
    def _validate_state_dict(state_dict: Mapping[str, object]) -> None:
        missing = sorted(set(_REQUIRED_STATE_SHAPES) - set(state_dict))
        if missing:
            raise ValueError(f"LAOM checkpoint is missing keys: {missing}")

        for name, expected_shape in _REQUIRED_STATE_SHAPES.items():
            value = state_dict[name]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"LAOM checkpoint value {name!r} must be a tensor")
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"LAOM checkpoint value {name!r} must have shape "
                    f"{expected_shape}, got {tuple(value.shape)}"
                )
            if not torch.is_floating_point(value):
                raise ValueError(f"LAOM checkpoint value {name!r} must be floating point")
            if not torch.isfinite(value).all():
                raise ValueError(f"LAOM checkpoint value {name!r} must be finite")

        if torch.any(state_dict["obs_rms.var"] < 0):
            raise ValueError("LAOM observation variance must be non-negative")

    def action_mean(self, observation: np.ndarray) -> np.ndarray:
        """Return the policy's unclipped linear action mean."""

        observation_array = np.asarray(observation, dtype=np.float32)
        if observation_array.shape != (OBSERVATION_DIM,):
            raise ValueError(
                "LAOM Hopper observation must have shape "
                f"{(OBSERVATION_DIM,)}, got {observation_array.shape}"
            )
        if not np.isfinite(observation_array).all():
            raise ValueError("LAOM Hopper observation must be finite")

        observation_tensor = torch.from_numpy(observation_array).to(self.device)
        with torch.inference_mode():
            action = self._actor(observation_tensor).cpu().numpy()
        if action.shape != (ACTION_DIM,):
            raise RuntimeError(f"LAOM actor returned unexpected shape {action.shape}")
        return action.astype(np.float32, copy=False)

    def action(self, observation: np.ndarray) -> np.ndarray:
        """Return the greedy action after the source environment's clipping."""

        return np.clip(self.action_mean(observation), -1.0, 1.0).astype(
            np.float32, copy=False
        )

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        return self.action(observation)


__all__ = ["ACTION_DIM", "OBSERVATION_DIM", "HopperPPOPolicy"]
