from __future__ import annotations

import numpy as np
import omegaconf
import torch
from torch.utils.data import Dataset

from implicit_nuisance.utils.misc import omegaconf_select


class RandomStepDirichletNuisance(Dataset):
    """Simplex-valued random-step signal observed with fresh nuisance.

    The initial signal is sampled from a uniform Dirichlet distribution. Each
    successor is conditionally Dirichlet distributed with concentration
    ``transition_precision * previous_signal + 1``. At every time step an
    independent Gaussian nuisance vector is concatenated to the signal before
    applying a fixed, injective LeakyReLU MLP. The observation mechanism is
    shared by train and test, while samples use isolated split-specific random
    number generators.
    """

    _VALID_SPLITS = ("train", "test")
    _SPLIT_SEEDS = {"train": 0x54524149, "test": 0x54455354}
    _LEAKY_RELU_SLOPE = 0.2

    def __init__(self, cfg: omegaconf.DictConfig, split: str = "train"):
        super().__init__()
        cfg = self.add_and_assert_specific_cfg(cfg)
        if split not in self._VALID_SPLITS:
            raise ValueError(f"split must be one of {self._VALID_SPLITS}")

        self.split = split
        settings = cfg.data.settings
        self.signal_dim = int(settings.signal_dim)
        self.nuisance_dim = int(settings.nuisance_dim)
        self.latent_dim = self.signal_dim + self.nuisance_dim
        self.image_dim = int(settings.image_dim)
        self.sequence_length = int(settings.sequence_length)
        self.transition_precision = float(settings.transition_precision)
        self.mechanism_seed = int(settings.mechanism_seed)
        self.sample_seed = int(settings.sample_seed)
        self.num_sequences = int(settings[f"num_{split}_sequences"])

        self._validate_settings()
        self._build_mechanism()
        self._generate_samples()

    def _validate_settings(self) -> None:
        if self.signal_dim < 2:
            raise ValueError("data.settings.signal_dim must be at least 2")
        if self.nuisance_dim < 1:
            raise ValueError("data.settings.nuisance_dim must be at least 1")
        if self.image_dim < self.latent_dim:
            raise ValueError(
                "data.settings.image_dim must be at least signal_dim + nuisance_dim "
                "for an injective observation mechanism"
            )
        if self.sequence_length < 2:
            raise ValueError("data.settings.sequence_length must be at least 2")
        if self.num_sequences < 1:
            raise ValueError(
                f"data.settings.num_{self.split}_sequences must be at least 1"
            )
        if not self.transition_precision > 0.0:
            raise ValueError("data.settings.transition_precision must be positive")

    def _build_mechanism(self) -> None:
        rng = np.random.default_rng(self.mechanism_seed)
        hidden_dim_1 = max(self.latent_dim, self.image_dim // 4)
        hidden_dim_2 = max(hidden_dim_1, self.image_dim // 2)
        dimensions = (self.latent_dim, hidden_dim_1, hidden_dim_2, self.image_dim)
        weights = []
        biases = []
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
            raw = rng.normal(size=(output_dim, input_dim))
            weight, _ = np.linalg.qr(raw, mode="reduced")
            weights.append(torch.from_numpy(weight.astype(np.float32)))
            biases.append(
                torch.from_numpy(
                    rng.normal(scale=0.1, size=output_dim).astype(np.float32)
                )
            )
        self.observation_weights = tuple(weights)
        self.observation_biases = tuple(biases)

    def _generate_samples(self) -> None:
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [self.sample_seed, self._SPLIT_SEEDS[self.split]]
            )
        )
        shape = (self.num_sequences, self.sequence_length)
        signal = np.empty((*shape, self.signal_dim), dtype=np.float32)
        transition_concentration = np.zeros_like(signal)
        transition_mean = np.zeros_like(signal)
        transition_mode = np.zeros_like(signal)

        signal[:, 0] = rng.dirichlet(
            np.ones(self.signal_dim), size=self.num_sequences
        ).astype(np.float32)
        for time in range(1, self.sequence_length):
            concentration = (
                self.transition_precision * signal[:, time - 1].astype(np.float64)
                + 1.0
            )
            gamma_samples = rng.gamma(shape=concentration)
            next_signal = gamma_samples / gamma_samples.sum(axis=-1, keepdims=True)

            signal[:, time] = next_signal.astype(np.float32)
            transition_concentration[:, time] = concentration.astype(np.float32)
            transition_mean[:, time] = (
                concentration / concentration.sum(axis=-1, keepdims=True)
            ).astype(np.float32)
            transition_mode[:, time] = signal[:, time - 1]

        nuisance = rng.normal(size=(*shape, self.nuisance_dim)).astype(np.float32)
        latent = np.concatenate((signal, nuisance), axis=-1)

        self.signal = torch.from_numpy(signal)
        self.nuisance = torch.from_numpy(nuisance)
        self.latent = torch.from_numpy(latent)
        self.transition_concentration = torch.from_numpy(transition_concentration)
        self.transition_mean = torch.from_numpy(transition_mean)
        self.transition_mode = torch.from_numpy(transition_mode)
        self.xs = self._mix(self.latent)

    def _mix(self, latent: torch.Tensor) -> torch.Tensor:
        observations = latent
        final_layer = len(self.observation_weights) - 1
        for layer, (weight, bias) in enumerate(
            zip(self.observation_weights, self.observation_biases)
        ):
            observations = observations @ weight.T + bias
            if layer != final_layer:
                observations = torch.nn.functional.leaky_relu(
                    observations, negative_slope=self._LEAKY_RELU_SLOPE
                )
        return observations

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        observations = self.xs[idx]
        signal = self.signal[idx]
        latent = self.latent[idx]
        labels = {
            "signal": signal,
            "nuisance": self.nuisance[idx],
            "latent": latent,
            "factors": latent,
            "pos": signal,
            "transition_concentration": self.transition_concentration[idx],
            "transition_mean": self.transition_mean[idx],
            "transition_mode": self.transition_mode[idx],
            "observations": observations,
        }
        return observations, labels

    @staticmethod
    def add_and_assert_specific_cfg(
        cfg: omegaconf.DictConfig,
    ) -> omegaconf.DictConfig:
        settings = cfg.data.settings
        settings.signal_dim = omegaconf_select(
            cfg, "data.settings.signal_dim", 2
        )
        settings.nuisance_dim = omegaconf_select(
            cfg, "data.settings.nuisance_dim", 8
        )
        settings.image_dim = omegaconf_select(
            cfg, "data.settings.image_dim", 100
        )
        settings.sequence_length = omegaconf_select(
            cfg, "data.settings.sequence_length", 5
        )
        settings.num_train_sequences = omegaconf_select(
            cfg, "data.settings.num_train_sequences", 100000
        )
        settings.num_test_sequences = omegaconf_select(
            cfg, "data.settings.num_test_sequences", 1000
        )
        settings.transition_precision = omegaconf_select(
            cfg, "data.settings.transition_precision", 100.0
        )
        settings.seed = omegaconf_select(cfg, "data.settings.seed", 1)
        settings.sample_seed = omegaconf_select(
            cfg, "data.settings.sample_seed", settings.seed
        )
        settings.mechanism_seed = omegaconf_select(
            cfg, "data.settings.mechanism_seed", 1
        )
        return cfg
