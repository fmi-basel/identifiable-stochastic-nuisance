"""Post-hoc representation probes for Causal3DIdent.

All probes are fitted on ``train_*`` and scored on ``test_*``.  In
particular, this module deliberately does not fit a probe on the array on
which its score is reported.  This makes the nuisance leakage scores useful
for distinguishing information in a representation from probe overfitting.

The standard Causal3DIdent factor order is

``(pos_x, pos_y, pos_z, rot_alpha, rot_beta, rot_gamma,
spotlight_position, object_hue, spotlight_hue, background_hue)``.

The temporal experiment in this repository uses spotlight position,
spotlight hue, and background hue (indices ``6, 8, 9``) as its predictive
signal by default.  The remaining continuous factors are treated as
nuisance for the conditional leakage diagnostic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.special import ndtri
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, r2_score
from sklearn.preprocessing import StandardScaler


CAUSAL3DIDENT_FACTOR_NAMES = (
    "object_position_x",
    "object_position_y",
    "object_position_z",
    "object_rotation_alpha",
    "object_rotation_beta",
    "object_rotation_gamma",
    "spotlight_position",
    "object_hue",
    "spotlight_hue",
    "background_hue",
)

CAUSAL3DIDENT_FACTOR_GROUPS = {
    "object_position": (0, 1, 2),
    "object_rotation": (3, 4, 5),
    "spotlight_position": (6,),
    "object_hue": (7,),
    "spotlight_hue": (8,),
    "background_hue": (9,),
}

DEFAULT_SIGNAL_INDICES = (6, 8, 9)


def evaluate_gaussian_conditional(
    samples: Any,
    means: Any,
    covariance: Any,
) -> dict[str, Any]:
    """Measure calibration and likelihood under one fixed Gaussian covariance.

    ``covariance`` may be a scalar variance, a vector containing the diagonal,
    or a full positive-definite matrix.  The returned whitened residual should
    have mean zero and identity covariance when the conditional is calibrated.
    """

    sample_array = _as_float_matrix("samples", samples)
    mean_array = _as_float_matrix("means", means)
    if sample_array.shape != mean_array.shape:
        raise ValueError(
            f"samples and means must have the same shape, got "
            f"{sample_array.shape} and {mean_array.shape}"
        )

    dimension = sample_array.shape[1]
    covariance_array = np.asarray(covariance, dtype=np.float64)
    if covariance_array.ndim == 0:
        covariance_array = np.eye(dimension) * float(covariance_array)
    elif covariance_array.ndim == 1:
        if covariance_array.shape != (dimension,):
            raise ValueError(
                f"diagonal covariance must have shape ({dimension},), "
                f"got {covariance_array.shape}"
            )
        covariance_array = np.diag(covariance_array)
    elif covariance_array.shape != (dimension, dimension):
        raise ValueError(
            f"covariance must have shape ({dimension}, {dimension}), "
            f"got {covariance_array.shape}"
        )
    if not np.isfinite(covariance_array).all():
        raise ValueError("covariance contains a non-finite value")
    if not np.allclose(covariance_array, covariance_array.T):
        raise ValueError("covariance must be symmetric")

    try:
        cholesky = np.linalg.cholesky(covariance_array)
    except np.linalg.LinAlgError as error:
        raise ValueError("covariance must be positive definite") from error

    residual = sample_array - mean_array
    whitened = np.linalg.solve(cholesky, residual.T).T
    squared_mahalanobis = np.sum(whitened**2, axis=1)
    log_determinant = 2.0 * np.log(np.diag(cholesky)).sum()
    nll = 0.5 * (
        dimension * np.log(2.0 * np.pi)
        + log_determinant
        + squared_mahalanobis
    )

    return {
        "n_samples": int(len(sample_array)),
        "dimension": int(dimension),
        "fixed_covariance": covariance_array.tolist(),
        "residual_mean": np.mean(residual, axis=0).tolist(),
        "residual_covariance": np.atleast_2d(
            np.cov(residual, rowvar=False, bias=True)
        ).tolist(),
        "whitened_residual_mean": np.mean(whitened, axis=0).tolist(),
        "whitened_residual_covariance": np.atleast_2d(
            np.cov(whitened, rowvar=False, bias=True)
        ).tolist(),
        "mean_squared_error": float(np.mean(residual**2)),
        "mean_squared_mahalanobis": float(np.mean(squared_mahalanobis)),
        "mean_gaussian_nll": float(np.mean(nll)),
        "mean_gaussian_nll_per_dimension": float(np.mean(nll) / dimension),
        "marginal_coverage_1sigma": float(np.mean(np.abs(whitened) <= 1.0)),
        "marginal_coverage_2sigma": float(np.mean(np.abs(whitened) <= 2.0)),
    }


def _as_float_matrix(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array, got {array.shape}")
    if array.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    return array


def _as_class_vector(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array, got {array.shape}")
    return array


def _validate_rows(reference_name: str, reference: np.ndarray, **arrays: np.ndarray) -> None:
    for name, array in arrays.items():
        if array.shape[0] != reference.shape[0]:
            raise ValueError(
                f"{name} has {array.shape[0]} rows but {reference_name} has "
                f"{reference.shape[0]}"
            )


def _standardize_train_test(
    train_x: np.ndarray, test_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return scaler.fit_transform(train_x), scaler.transform(test_x)


def _per_output_r2(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    values = r2_score(target, prediction, multioutput="raw_values")
    return np.atleast_1d(np.asarray(values, dtype=np.float64))


def _fit_linear_predictions(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    train_x_scaled, test_x_scaled = _standardize_train_test(train_x, test_x)
    return LinearRegression().fit(train_x_scaled, train_y).predict(test_x_scaled)


def _signal_coordinates(
    raw_signal: np.ndarray,
    *,
    transform: str,
    clip: float,
) -> np.ndarray:
    if transform == "identity":
        return raw_signal
    if transform != "normal_score":
        raise ValueError("signal_transform must be 'normal_score' or 'identity'")
    if not 0 < clip < 0.5:
        raise ValueError("signal_clip must lie strictly between 0 and 0.5")
    if np.any(raw_signal < -1.0 - 1e-7) or np.any(raw_signal > 1.0 + 1e-7):
        raise ValueError("normal-score signal factors must lie in [-1, 1]")
    probabilities = np.clip((raw_signal + 1.0) / 2.0, clip, 1.0 - clip)
    return ndtri(probabilities)


def _fit_logistic(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    max_iter: int,
) -> LogisticRegression:
    """Fit an unpenalized probe across supported scikit-learn versions."""

    kwargs = {"solver": "lbfgs", "max_iter": max_iter}
    try:
        return LogisticRegression(penalty=None, **kwargs).fit(train_x, train_y)
    except (TypeError, ValueError) as error:
        # scikit-learn < 1.2 spells an absent penalty as the string ``"none"``.
        try:
            return LogisticRegression(penalty="none", **kwargs).fit(train_x, train_y)
        except (TypeError, ValueError):
            raise error


def _logistic_scores(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    max_iter: int,
) -> tuple[float, float]:
    train_x_scaled, test_x_scaled = _standardize_train_test(train_x, test_x)
    classifier = _fit_logistic(train_x_scaled, train_y, max_iter=max_iter)
    prediction = classifier.predict(test_x_scaled)
    probabilities = classifier.predict_proba(test_x_scaled)
    return (
        float(accuracy_score(test_y, prediction)),
        float(log_loss(test_y, probabilities, labels=classifier.classes_)),
    )


def _named_scores(names: Sequence[str], scores: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, scores, strict=True)}


def _resolve_factor_metadata(
    n_factors: int,
    factor_names: Sequence[str] | None,
    factor_groups: Mapping[str, Sequence[int]] | None,
) -> tuple[tuple[str, ...], dict[str, tuple[int, ...]]]:
    if factor_names is None:
        if n_factors == len(CAUSAL3DIDENT_FACTOR_NAMES):
            names = CAUSAL3DIDENT_FACTOR_NAMES
        else:
            names = tuple(f"factor_{index}" for index in range(n_factors))
    else:
        names = tuple(factor_names)
        if len(names) != n_factors:
            raise ValueError(
                f"factor_names has length {len(names)} but factors have {n_factors} columns"
            )
        if len(set(names)) != len(names):
            raise ValueError("factor_names must be unique")

    if factor_groups is None:
        if n_factors == len(CAUSAL3DIDENT_FACTOR_NAMES):
            groups = dict(CAUSAL3DIDENT_FACTOR_GROUPS)
        else:
            groups = {name: (index,) for index, name in enumerate(names)}
    else:
        groups = {name: tuple(int(index) for index in indices) for name, indices in factor_groups.items()}

    for group_name, indices in groups.items():
        if not indices:
            raise ValueError(f"factor group {group_name!r} is empty")
        if any(index < 0 or index >= n_factors for index in indices):
            raise ValueError(f"factor group {group_name!r} contains an invalid index")
    return names, groups


def evaluate_representations(
    train_z: Any,
    train_factors: Any,
    train_class: Any,
    test_z: Any,
    test_factors: Any,
    test_class: Any,
    *,
    signal_indices: Sequence[int] = DEFAULT_SIGNAL_INDICES,
    factor_names: Sequence[str] | None = None,
    factor_groups: Mapping[str, Sequence[int]] | None = None,
    signal_transform: str = "normal_score",
    signal_clip: float = 1e-6,
    logistic_max_iter: int = 2000,
) -> dict[str, Any]:
    """Evaluate signal recovery and nuisance leakage on a held-out split.

    Args:
        train_z: Training representations with shape ``[n_train, z_dim]``.
        train_factors: Training continuous factors with shape
            ``[n_train, n_factors]``.
        train_class: Training object-class labels with shape ``[n_train]``.
        test_z: Held-out representations with shape ``[n_test, z_dim]``.
        test_factors: Held-out continuous factors.
        test_class: Held-out object-class labels.
        signal_indices: Factor columns defining the predictive signal ``S``.
            All other factor columns define continuous nuisance ``N``.
        factor_names: Optional names for factor columns.
        factor_groups: Optional mapping from reported group names to columns.
        signal_transform: ``"normal_score"`` evaluates affine recovery of
            ``q = Phi^-1((s + 1) / 2)``, as required by the fixed-covariance
            Gaussian experiment.  ``"identity"`` is available for ablations.
        signal_clip: Probability clipping used by the normal-score transform.
        logistic_max_iter: Iteration limit for logistic probes.

    Returns:
        A JSON-serializable nested dictionary.  Every reported score is
        evaluated on ``test_*`` after fitting and standardizing on ``train_*``.

    ``conditional_nuisance_leakage.continuous.incremental_r2`` is
    ``R2(N ~ [q, Z]) - R2(N ~ q)``.  For the class diagnostic, accuracy gain is
    ``accuracy([q, Z]) - accuracy(q)`` and log-loss gain is
    ``log_loss(q) - log_loss([q, Z])``.  Thus positive gains indicate nuisance
    information in ``Z`` beyond what is already predictable from ``q``.  The
    The continuous diagnostic uses held-out linear probes.
    """

    train_z_array = _as_float_matrix("train_z", train_z)
    test_z_array = _as_float_matrix("test_z", test_z)
    train_factor_array = _as_float_matrix("train_factors", train_factors)
    test_factor_array = _as_float_matrix("test_factors", test_factors)
    train_class_array = _as_class_vector("train_class", train_class)
    test_class_array = _as_class_vector("test_class", test_class)

    _validate_rows(
        "train_z",
        train_z_array,
        train_factors=train_factor_array,
        train_class=train_class_array,
    )
    _validate_rows(
        "test_z",
        test_z_array,
        test_factors=test_factor_array,
        test_class=test_class_array,
    )
    if train_z_array.shape[1] != test_z_array.shape[1]:
        raise ValueError("train_z and test_z must have the same number of columns")
    if train_factor_array.shape[1] != test_factor_array.shape[1]:
        raise ValueError("train_factors and test_factors must have the same number of columns")
    train_classes = set(np.unique(train_class_array).tolist())
    test_classes = set(np.unique(test_class_array).tolist())
    if len(train_classes) < 2:
        raise ValueError("train_class must contain at least two classes")
    if not test_classes.issubset(train_classes):
        raise ValueError("test_class contains a class absent from train_class")

    n_factors = train_factor_array.shape[1]
    names, groups = _resolve_factor_metadata(n_factors, factor_names, factor_groups)
    signal_columns = tuple(int(index) for index in signal_indices)
    if not signal_columns:
        raise ValueError("signal_indices must not be empty")
    if len(set(signal_columns)) != len(signal_columns):
        raise ValueError("signal_indices must be unique")
    if any(index < 0 or index >= n_factors for index in signal_columns):
        raise ValueError("signal_indices contains an invalid factor index")
    nuisance_columns = tuple(index for index in range(n_factors) if index not in signal_columns)

    # Global factor probes.  One multi-output fit is both faster and equivalent
    # to fitting one independent regressor per continuous factor here.
    linear_prediction = _fit_linear_predictions(
        train_z_array, train_factor_array, test_z_array
    )
    linear_scores = _per_output_r2(test_factor_array, linear_prediction)

    grouped_scores: dict[str, dict[str, float]] = {}
    for group_name, indices in groups.items():
        grouped_scores[group_name] = {
            "linear_r2": float(np.mean(linear_scores[list(indices)])),
        }

    class_accuracy, _ = _logistic_scores(
        train_z_array,
        train_class_array,
        test_z_array,
        test_class_array,
        max_iter=logistic_max_iter,
    )

    train_signal = _signal_coordinates(
        train_factor_array[:, signal_columns],
        transform=signal_transform,
        clip=signal_clip,
    )
    test_signal = _signal_coordinates(
        test_factor_array[:, signal_columns],
        transform=signal_transform,
        clip=signal_clip,
    )
    signal_prediction = _fit_linear_predictions(train_z_array, train_signal, test_z_array)
    signal_scores = _per_output_r2(test_signal, signal_prediction)

    if nuisance_columns:
        train_nuisance = train_factor_array[:, nuisance_columns]
        test_nuisance = test_factor_array[:, nuisance_columns]
        baseline_nuisance_prediction = _fit_linear_predictions(
            train_signal, train_nuisance, test_signal
        )
        full_nuisance_prediction = _fit_linear_predictions(
            np.concatenate((train_signal, train_z_array), axis=1),
            train_nuisance,
            np.concatenate((test_signal, test_z_array), axis=1),
        )
        baseline_nuisance_scores = _per_output_r2(
            test_nuisance, baseline_nuisance_prediction
        )
        full_nuisance_scores = _per_output_r2(test_nuisance, full_nuisance_prediction)
        incremental_nuisance_scores = full_nuisance_scores - baseline_nuisance_scores

        nuisance_names = tuple(names[index] for index in nuisance_columns)
        continuous_leakage: dict[str, Any] = {
            "baseline_r2": _named_scores(nuisance_names, baseline_nuisance_scores),
            "with_representation_r2": _named_scores(nuisance_names, full_nuisance_scores),
            "incremental_r2": _named_scores(nuisance_names, incremental_nuisance_scores),
            "mean_baseline_r2": float(np.mean(baseline_nuisance_scores)),
            "mean_with_representation_r2": float(np.mean(full_nuisance_scores)),
            "mean_incremental_r2": float(np.mean(incremental_nuisance_scores)),
        }
    else:
        continuous_leakage = {
            "baseline_r2": {},
            "with_representation_r2": {},
            "incremental_r2": {},
            "mean_baseline_r2": None,
            "mean_with_representation_r2": None,
            "mean_incremental_r2": None,
        }

    baseline_class_accuracy, baseline_class_log_loss = _logistic_scores(
        train_signal,
        train_class_array,
        test_signal,
        test_class_array,
        max_iter=logistic_max_iter,
    )
    full_class_accuracy, full_class_log_loss = _logistic_scores(
        np.concatenate((train_signal, train_z_array), axis=1),
        train_class_array,
        np.concatenate((test_signal, test_z_array), axis=1),
        test_class_array,
        max_iter=logistic_max_iter,
    )

    return {
        "metadata": {
            "n_train": int(len(train_z_array)),
            "n_test": int(len(test_z_array)),
            "representation_dim": int(train_z_array.shape[1]),
            "factor_names": list(names),
            "signal_indices": list(signal_columns),
            "signal_transform": signal_transform,
            "signal_clip": float(signal_clip),
            "nuisance_indices": list(nuisance_columns),
        },
        "factors": {
            "linear_r2": _named_scores(names, linear_scores),
            "groups": grouped_scores,
        },
        "class": {"logistic_accuracy": class_accuracy},
        "signal_affine_r2": float(np.mean(signal_scores)),
        "signal": {
            "names": [names[index] for index in signal_columns],
            "affine_r2": float(np.mean(signal_scores)),
            "affine_r2_per_factor": _named_scores(
                tuple(names[index] for index in signal_columns), signal_scores
            ),
        },
        "conditional_nuisance_leakage": {
            "continuous": continuous_leakage,
            "class": {
                "baseline_accuracy": baseline_class_accuracy,
                "with_representation_accuracy": full_class_accuracy,
                "accuracy_gain": full_class_accuracy - baseline_class_accuracy,
                "baseline_log_loss": baseline_class_log_loss,
                "with_representation_log_loss": full_class_log_loss,
                "log_loss_gain": baseline_class_log_loss - full_class_log_loss,
            },
        },
    }


__all__ = [
    "CAUSAL3DIDENT_FACTOR_GROUPS",
    "CAUSAL3DIDENT_FACTOR_NAMES",
    "DEFAULT_SIGNAL_INDICES",
    "evaluate_gaussian_conditional",
    "evaluate_representations",
]
