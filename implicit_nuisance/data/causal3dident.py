"""Sequence adapter for the offline Causal3DIdent dataset.

The official archive has the following layout::

    root/
      trainset/
        raw_latents_0.npy
        images_0/00000.png
        ...
      testset/
        raw_latents_0.npy
        images_0/00000.png
        ...

Each item starts from one rendered observation and creates the remaining views by
sampling a Gaussian AR(1) target in normal-score coordinates.  A render
is obtained by looking up one of the ``k`` closest signal values in an independently
sampled object class.  Only the configured signal columns participate in lookup;
all remaining factors and the class are nuisance variables.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import omegaconf
import torch
from PIL import Image
from scipy.spatial import cKDTree
from scipy.special import ndtri
from torch.utils.data import Dataset

from implicit_nuisance.utils.misc import omegaconf_select


class Causal3DIdent(Dataset):
    """Causal3DIdent observations with a probabilistic Gaussian conditional.

    Configuration lives below ``cfg.data.settings``.  The required setting is
    ``root``.  Useful optional settings are ``classes`` (default ``0..6``),
    ``sequence_length`` (2), ``signal_indices`` (``[6, 8, 9]``), ``rho`` (0.9),
    ``transition_std`` (``sqrt(1-rho**2)``), ``k`` (1), ``seed`` (1),
    ``image_size`` (224), and ``normalization`` (``"none"``).  Normalization may
    also be ``"minus_one_one"``, ``"imagenet"``, or a mapping containing
    three-channel ``mean`` and ``std`` values.  Training transitions are fresh
    draws by default; test transitions remain deterministic per index.  Set
    ``stochastic_transitions: false`` to freeze the training pairs as well.

    Returned observations have shape ``[T, C, H, W]``.  Every label is aligned on
    its first (sequence) axis.  ``factors`` contains all ten continuous factors,
    ``class`` contains the object class, ``signal`` contains the configured factor
    subset, and ``nuisance`` concatenates the seven complementary factors with the
    class.  Transition diagnostics are expressed in normal-score space:
    ``signal_normal_score``, ``transition_mean``, ``transition_std``,
    ``transition_target``, and ``transition_noise``.  ``knn_distance``,
    ``neighbor_rank``, and ``source_index`` describe the finite-dataset lookup.
    """

    _VALID_SPLITS = ("train", "test")
    _SPLIT_DIRECTORIES = {"train": "trainset", "test": "testset"}
    _SPLIT_SEEDS = {"train": 0, "test": 1}
    _NUM_FACTORS = 10
    _DEFAULT_SIGNAL_INDICES = (6, 8, 9)
    _NORMAL_SCORE_CLIP = 1e-6
    _CLASS_SAMPLING_MODES = ("uniform", "joint_knn")

    def __init__(self, cfg: omegaconf.DictConfig, split: str = "train") -> None:
        super().__init__()
        if split not in self._VALID_SPLITS:
            raise ValueError(f"split must be one of {self._VALID_SPLITS}, got {split!r}")

        self.split = split
        self.data_cfg = cfg.data

        root = omegaconf_select(cfg, "data.settings.root")
        if root is None:
            raise ValueError("data.settings.root must point to the Causal3DIdent directory")
        self.root = Path(root).expanduser()
        self.split_root = self.root / self._SPLIT_DIRECTORIES[split]
        if not self.split_root.is_dir():
            raise FileNotFoundError(
                f"Causal3DIdent split directory does not exist: {self.split_root}"
            )

        classes = omegaconf_select(cfg, "data.settings.classes", list(range(7)))
        self.classes = tuple(int(class_id) for class_id in classes)
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError("data.settings.classes must contain distinct class ids")

        signal_indices = omegaconf_select(
            cfg,
            "data.settings.signal_indices",
            list(self._DEFAULT_SIGNAL_INDICES),
        )
        self.signal_indices = tuple(int(index) for index in signal_indices)
        if not self.signal_indices or len(set(self.signal_indices)) != len(self.signal_indices):
            raise ValueError("data.settings.signal_indices must contain distinct indices")
        if min(self.signal_indices) < 0 or max(self.signal_indices) >= self._NUM_FACTORS:
            raise ValueError("signal indices must be between 0 and 9")
        signal_index_set = set(self.signal_indices)
        self.nuisance_indices = tuple(
            index for index in range(self._NUM_FACTORS) if index not in signal_index_set
        )

        sequence_length = omegaconf_select(cfg, "data.settings.sequence_length", None)
        if sequence_length is None:
            sequence_length = omegaconf_select(cfg, "data.settings.seq_len", 2)
        self.sequence_length = int(sequence_length)
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")

        rho = omegaconf_select(cfg, "data.settings.rho", None)
        if rho is None:
            rho = omegaconf_select(cfg, "data.settings.transition_rho", 0.9)
        self.rho = float(rho)
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie strictly between -1 and 1")

        transition_std = omegaconf_select(cfg, "data.settings.transition_std", None)
        if transition_std is None:
            transition_std = omegaconf_select(cfg, "data.settings.fixed_std", None)
        if transition_std is None:
            transition_std = np.sqrt(1.0 - self.rho**2)
        self.transition_std = float(transition_std)
        if not np.isfinite(self.transition_std) or self.transition_std <= 0:
            raise ValueError("transition_std must be finite and strictly positive")

        self.k = int(omegaconf_select(cfg, "data.settings.k", 1))
        if self.k < 1:
            raise ValueError("k must be at least 1")
        self.class_sampling = str(
            omegaconf_select(cfg, "data.settings.class_sampling", "uniform")
        )
        if self.class_sampling not in self._CLASS_SAMPLING_MODES:
            raise ValueError(
                "data.settings.class_sampling must be one of "
                f"{self._CLASS_SAMPLING_MODES}"
            )
        self.seed = int(omegaconf_select(cfg, "data.settings.seed", 1))
        self.stochastic_transitions = split == "train" and bool(
            omegaconf_select(
                cfg,
                "data.settings.stochastic_transitions",
                True,
            )
        )
        self._transition_generators: dict[int, np.random.Generator] = {}

        self.image_size = self._parse_image_size(
            omegaconf_select(cfg, "data.settings.image_size", 224)
        )
        self._normalization_mean, self._normalization_std = self._parse_normalization(
            omegaconf_select(cfg, "data.settings.normalization", "none")
        )

        self._latents: dict[int, np.ndarray] = {}
        self._normal_scores: dict[int, np.ndarray] = {}
        self._trees: dict[int, cKDTree] = {}
        self._image_name_width: dict[int, int] = {}
        class_lengths = []
        for class_id in self.classes:
            latent_path = self.split_root / f"raw_latents_{class_id}.npy"
            if not latent_path.is_file():
                raise FileNotFoundError(f"Missing Causal3DIdent latents: {latent_path}")
            latents = np.load(latent_path, mmap_mode="r", allow_pickle=False)
            if latents.ndim != 2 or latents.shape[1] != self._NUM_FACTORS:
                raise ValueError(
                    f"{latent_path} must have shape [N, {self._NUM_FACTORS}], "
                    f"got {latents.shape}"
                )
            if len(latents) == 0:
                raise ValueError(f"{latent_path} must contain at least one sample")

            image_directory = self.split_root / f"images_{class_id}"
            if not image_directory.is_dir():
                raise FileNotFoundError(
                    f"Missing Causal3DIdent image directory: {image_directory}"
                )

            signal = np.asarray(latents[:, self.signal_indices], dtype=np.float64)
            if not np.isfinite(signal).all():
                raise ValueError(f"Signal factors in {latent_path} must be finite")
            normal_scores = self._to_normal_scores(signal)

            self._latents[class_id] = latents
            self._normal_scores[class_id] = normal_scores
            self._trees[class_id] = cKDTree(normal_scores)
            self._image_name_width[class_id] = max(1, len(str(len(latents) - 1)))
            class_lengths.append(len(latents))

        self._class_offsets = np.concatenate(
            ([0], np.cumsum(np.asarray(class_lengths, dtype=np.int64)))
        )
        self._total_observations = int(self._class_offsets[-1])
        if self.class_sampling == "joint_knn":
            self._joint_normal_scores = np.concatenate(
                [self._normal_scores[class_id] for class_id in self.classes], axis=0
            )
            self._joint_tree = cKDTree(self._joint_normal_scores)
            self._joint_class_ids = np.concatenate(
                [
                    np.full(len(self._latents[class_id]), class_id, dtype=np.int64)
                    for class_id in self.classes
                ]
            )
            self._joint_source_indices = np.concatenate(
                [
                    np.arange(len(self._latents[class_id]), dtype=np.int64)
                    for class_id in self.classes
                ]
            )

        count_key = f"data.settings.num_{split}_sequences"
        requested_sequences = omegaconf_select(cfg, count_key, None)
        if requested_sequences is None:
            self._start_indices = None
            self.num_sequences = self._total_observations
        else:
            self.num_sequences = int(requested_sequences)
            if self.num_sequences < 1:
                raise ValueError(f"{count_key} must be at least 1")
            start_rng = np.random.default_rng(
                np.random.SeedSequence(
                    [self.seed, self._SPLIT_SEEDS[self.split], 0xC3D1]
                )
            )
            self._start_indices = start_rng.choice(
                self._total_observations,
                size=self.num_sequences,
                replace=self.num_sequences > self._total_observations,
            )

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not 0 <= idx < len(self):
            raise IndexError(idx)

        global_index = idx if self._start_indices is None else int(self._start_indices[idx])
        class_id, source_index = self._decode_global_index(global_index)
        rng = self._transition_rng(idx)

        class_ids = [class_id]
        source_indices = [source_index]
        normal_signals = [self._normal_scores[class_id][source_index]]
        transition_means = [normal_signals[0].copy()]
        transition_targets = [normal_signals[0].copy()]
        transition_noises = [np.zeros(len(self.signal_indices), dtype=np.float64)]
        knn_distances = [0.0]
        neighbor_ranks = [0]

        for _ in range(1, self.sequence_length):
            mean = self.rho * normal_signals[-1]
            noise = rng.normal(
                loc=0.0,
                scale=self.transition_std,
                size=len(self.signal_indices),
            )
            target = mean + noise

            if self.class_sampling == "uniform":
                next_class = int(rng.choice(self.classes))
                query_k = min(self.k, len(self._latents[next_class]))
                distances, candidates = self._trees[next_class].query(
                    target, k=query_k
                )
            else:
                query_k = min(self.k, self._total_observations)
                distances, candidates = self._joint_tree.query(target, k=query_k)
            distances = np.atleast_1d(distances)
            candidates = np.atleast_1d(candidates)
            neighbor_rank = int(rng.integers(query_k)) if query_k > 1 else 0
            candidate = int(candidates[neighbor_rank])
            if self.class_sampling == "uniform":
                next_index = candidate
            else:
                next_class = int(self._joint_class_ids[candidate])
                next_index = int(self._joint_source_indices[candidate])

            class_ids.append(next_class)
            source_indices.append(next_index)
            normal_signals.append(self._normal_scores[next_class][next_index])
            transition_means.append(mean)
            transition_targets.append(target)
            transition_noises.append(noise)
            knn_distances.append(float(distances[neighbor_rank]))
            neighbor_ranks.append(neighbor_rank)

        images = torch.stack(
            [
                self._load_image(current_class, current_index)
                for current_class, current_index in zip(class_ids, source_indices)
            ],
            dim=0,
        )
        factors = torch.from_numpy(
            np.stack(
                [
                    np.asarray(self._latents[current_class][current_index], dtype=np.float32)
                    for current_class, current_index in zip(class_ids, source_indices)
                ]
            )
        )
        classes = torch.tensor(class_ids, dtype=torch.long)
        signal = factors[:, self.signal_indices]
        nuisance_factors = factors[:, self.nuisance_indices]
        nuisance = torch.cat((nuisance_factors, classes[:, None].to(torch.float32)), dim=1)
        signal_normal_score = torch.from_numpy(
            np.asarray(normal_signals, dtype=np.float32)
        )

        labels = {
            "factors": factors,
            "class": classes,
            "signal": signal,
            "nuisance": nuisance,
            "signal_normal_score": signal_normal_score,
            "transition_mean": torch.from_numpy(
                np.asarray(transition_means, dtype=np.float32)
            ),
            "transition_std": torch.full_like(signal_normal_score, self.transition_std),
            "transition_target": torch.from_numpy(
                np.asarray(transition_targets, dtype=np.float32)
            ),
            "transition_noise": torch.from_numpy(
                np.asarray(transition_noises, dtype=np.float32)
            ),
            "knn_distance": torch.tensor(knn_distances, dtype=torch.float32),
            "neighbor_rank": torch.tensor(neighbor_ranks, dtype=torch.long),
            "source_index": torch.tensor(source_indices, dtype=torch.long),
        }
        return images, labels

    def _decode_global_index(self, global_index: int) -> tuple[int, int]:
        class_position = int(np.searchsorted(self._class_offsets, global_index, side="right") - 1)
        class_id = self.classes[class_position]
        source_index = int(global_index - self._class_offsets[class_position])
        return class_id, source_index

    def _transition_rng(self, idx: int) -> np.random.Generator:
        if not self.stochastic_transitions:
            return np.random.default_rng(
                np.random.SeedSequence(
                    [self.seed, self._SPLIT_SEEDS[self.split], int(idx)]
                )
            )

        worker = torch.utils.data.get_worker_info()
        worker_id = -1 if worker is None else int(worker.id)
        if worker_id not in self._transition_generators:
            worker_seed = self.seed if worker is None else int(worker.seed)
            self._transition_generators[worker_id] = np.random.default_rng(
                np.random.SeedSequence(
                    [worker_seed, self._SPLIT_SEEDS[self.split], 0xC3D2]
                )
            )
        return self._transition_generators[worker_id]

    def _load_image(self, class_id: int, source_index: int) -> torch.Tensor:
        width = self._image_name_width[class_id]
        image_path = (
            self.split_root
            / f"images_{class_id}"
            / f"{source_index:0{width}d}.png"
        )
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing Causal3DIdent image: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            height, width = self.image_size
            if image.size != (width, height):
                resampling = getattr(Image, "Resampling", Image).BILINEAR
                image = image.resize((width, height), resample=resampling)
            image_array = np.array(image, dtype=np.float32, copy=True) / 255.0

        tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        return (tensor - self._normalization_mean) / self._normalization_std

    @staticmethod
    def _to_normal_scores(signal: np.ndarray) -> np.ndarray:
        """Map the dataset's uniform ``[-1, 1]`` factors to standard normals."""

        probabilities = np.clip(
            (signal + 1.0) / 2.0,
            Causal3DIdent._NORMAL_SCORE_CLIP,
            1.0 - Causal3DIdent._NORMAL_SCORE_CLIP,
        )
        return np.ascontiguousarray(ndtri(probabilities), dtype=np.float64)

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
        elif isinstance(value, str) and value.lower() in {
            "minus_one_one",
            "-1_1",
        }:
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        elif isinstance(value, str) and value.lower() == "imagenet":
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        elif isinstance(value, Mapping):
            if "mean" not in value or "std" not in value:
                raise ValueError("custom normalization requires mean and std")
            mean, std = value["mean"], value["std"]
        else:
            raise ValueError(
                "normalization must be none, minus_one_one, imagenet, or a mean/std mapping"
            )

        mean_tensor = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        std_tensor = torch.as_tensor(std, dtype=torch.float32).reshape(-1)
        if mean_tensor.numel() != 3 or std_tensor.numel() != 3:
            raise ValueError("normalization mean and std must each have three values")
        if not torch.isfinite(mean_tensor).all() or not torch.isfinite(std_tensor).all():
            raise ValueError("normalization mean and std must be finite")
        if (std_tensor <= 0).any():
            raise ValueError("normalization std must be strictly positive")
        return mean_tensor[:, None, None], std_tensor[:, None, None]
