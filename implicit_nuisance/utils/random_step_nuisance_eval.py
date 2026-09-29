"""Held-out probes for the random-step implicit-nuisance experiment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.stats import kstest
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from implicit_nuisance.utils.causal3dident_eval import (
    evaluate_gaussian_conditional,
)


def _matrix(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got {array.shape}")
    if array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must have at least two rows and one column")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    return array


def _validate_split(prefix: str, **arrays: np.ndarray) -> None:
    row_counts = {name: len(array) for name, array in arrays.items()}
    if len(set(row_counts.values())) != 1:
        raise ValueError(f"{prefix} arrays have inconsistent row counts: {row_counts}")


def _standardize(
    train_x: np.ndarray, test_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return scaler.fit_transform(train_x), scaler.transform(test_x)


def _linear_prediction(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray
) -> np.ndarray:
    scaled_train, scaled_test = _standardize(train_x, test_x)
    return LinearRegression().fit(scaled_train, train_y).predict(scaled_test)


def _rbf_prediction(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float,
    gamma: float | None,
) -> tuple[np.ndarray, float]:
    scaled_train, scaled_test = _standardize(train_x, test_x)
    resolved_gamma = 1.0 / scaled_train.shape[1] if gamma is None else gamma
    prediction = KernelRidge(
        alpha=alpha,
        kernel="rbf",
        gamma=resolved_gamma,
    ).fit(scaled_train, train_y).predict(scaled_test)
    return prediction, float(resolved_gamma)


def _r2(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.atleast_1d(
        np.asarray(r2_score(target, prediction, multioutput="raw_values"))
    )


def _named(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_{index}": float(value)
        for index, value in enumerate(values)
    }


def _scores(prefix: str, values: np.ndarray) -> dict[str, Any]:
    return {
        "per_dimension": _named(prefix, values),
        "mean": float(np.mean(values)),
    }


def _bounded_indices(
    length: int, maximum: int | None, rng: np.random.Generator
) -> np.ndarray:
    if maximum is None or length <= maximum:
        return np.arange(length)
    return np.sort(rng.choice(length, size=maximum, replace=False))


def _transition_metrics(
    name: str, transition: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    if transition is None:
        return None
    missing = {"samples", "means", "covariance"} - set(transition)
    if missing:
        raise ValueError(f"{name} transition is missing {sorted(missing)}")
    return evaluate_gaussian_conditional(
        transition["samples"],
        transition["means"],
        transition["covariance"],
    )


def evaluate_initial_distribution(
    samples: Any,
    distribution: str,
) -> dict[str, Any]:
    """Compare initial signal samples with their configured reference law."""

    sample_array = _matrix("initial signal samples", samples)
    if distribution == "uniform":
        expected_mean = 0.5
        expected_variance = 1.0 / 12.0
        reference = "uniform[0,1]"
        test_name = "uniform"
        test_args = (0.0, 1.0)
    elif distribution == "gaussian":
        expected_mean = 0.0
        expected_variance = 1.0
        reference = "standard_normal"
        test_name = "norm"
        test_args = ()
    else:
        raise ValueError(
            "initial distribution must be 'gaussian' or 'uniform', got "
            f"{distribution!r}"
        )

    tests = [
        kstest(sample_array[:, dimension], test_name, args=test_args)
        for dimension in range(sample_array.shape[1])
    ]
    result: dict[str, Any] = {
        "distribution": distribution,
        "reference": reference,
        "n_samples": int(len(sample_array)),
        "dimension": int(sample_array.shape[1]),
        "expected_mean": expected_mean,
        "expected_variance": expected_variance,
        "empirical_mean": np.mean(sample_array, axis=0).tolist(),
        "empirical_variance": np.var(sample_array, axis=0).tolist(),
        "empirical_min": np.min(sample_array, axis=0).tolist(),
        "empirical_max": np.max(sample_array, axis=0).tolist(),
        "ks_statistic": [float(test.statistic) for test in tests],
        "ks_pvalue": [float(test.pvalue) for test in tests],
    }
    if distribution == "uniform":
        inside = (sample_array >= 0.0) & (sample_array <= 1.0)
        result["fraction_in_unit_cube"] = float(np.mean(np.all(inside, axis=1)))
    return result


def evaluate_random_step_nuisance(
    train_z: Any,
    train_signal: Any,
    train_nuisance: Any,
    train_observations: Any,
    test_z: Any,
    test_signal: Any,
    test_nuisance: Any,
    test_observations: Any,
    *,
    model_transition: Mapping[str, Any] | None = None,
    source_transition: Mapping[str, Any] | None = None,
    kernel_sample_max: int | None = 1024,
    kernel_alpha: float = 1e-3,
    kernel_gamma: float | None = None,
    random_state: int = 0,
) -> dict[str, Any]:
    """Fit probes on training samples and score them on held-out samples.

    Nuisance leakage is measured both directly, ``N ~ Z``, and conditionally
    as the held-out improvement from ``N ~ S`` to ``N ~ [S, Z]``.  Observation
    controls verify that the nuisance is present in ``X`` even when it is absent
    from an equal-dimensional learned representation.
    """

    train_arrays = {
        "z": _matrix("train_z", train_z),
        "signal": _matrix("train_signal", train_signal),
        "nuisance": _matrix("train_nuisance", train_nuisance),
        "observations": _matrix("train_observations", train_observations),
    }
    test_arrays = {
        "z": _matrix("test_z", test_z),
        "signal": _matrix("test_signal", test_signal),
        "nuisance": _matrix("test_nuisance", test_nuisance),
        "observations": _matrix("test_observations", test_observations),
    }
    _validate_split("train", **train_arrays)
    _validate_split("test", **test_arrays)
    for name in train_arrays:
        if train_arrays[name].shape[1] != test_arrays[name].shape[1]:
            raise ValueError(f"train/test {name} dimensions differ")
    if kernel_sample_max is not None and kernel_sample_max < 2:
        raise ValueError("kernel_sample_max must be at least two or None")
    if kernel_alpha < 0:
        raise ValueError("kernel_alpha must be non-negative")
    if kernel_gamma is not None and kernel_gamma <= 0:
        raise ValueError("kernel_gamma must be positive")

    train_z_array = train_arrays["z"]
    train_s = train_arrays["signal"]
    train_n = train_arrays["nuisance"]
    train_x = train_arrays["observations"]
    test_z_array = test_arrays["z"]
    test_s = test_arrays["signal"]
    test_n = test_arrays["nuisance"]
    test_x = test_arrays["observations"]

    signal_linear = _r2(
        test_s, _linear_prediction(train_z_array, train_s, test_z_array)
    )
    nuisance_raw_linear = _r2(
        test_n, _linear_prediction(train_z_array, train_n, test_z_array)
    )
    nuisance_baseline_linear = _r2(
        test_n, _linear_prediction(train_s, train_n, test_s)
    )
    nuisance_full_linear = _r2(
        test_n,
        _linear_prediction(
            np.concatenate((train_s, train_z_array), axis=1),
            train_n,
            np.concatenate((test_s, test_z_array), axis=1),
        ),
    )

    rng = np.random.default_rng(random_state)
    train_indices = _bounded_indices(len(train_z_array), kernel_sample_max, rng)
    test_indices = _bounded_indices(len(test_z_array), kernel_sample_max, rng)
    kernel_train_z = train_z_array[train_indices]
    kernel_test_z = test_z_array[test_indices]
    kernel_train_s = train_s[train_indices]
    kernel_test_s = test_s[test_indices]
    kernel_train_n = train_n[train_indices]
    kernel_test_n = test_n[test_indices]
    kernel_train_x = train_x[train_indices]
    kernel_test_x = test_x[test_indices]

    nuisance_raw_rbf_prediction, raw_gamma = _rbf_prediction(
        kernel_train_z,
        kernel_train_n,
        kernel_test_z,
        alpha=kernel_alpha,
        gamma=kernel_gamma,
    )
    nuisance_baseline_rbf_prediction, baseline_gamma = _rbf_prediction(
        kernel_train_s,
        kernel_train_n,
        kernel_test_s,
        alpha=kernel_alpha,
        gamma=kernel_gamma,
    )
    nuisance_full_rbf_prediction, full_gamma = _rbf_prediction(
        np.concatenate((kernel_train_s, kernel_train_z), axis=1),
        kernel_train_n,
        np.concatenate((kernel_test_s, kernel_test_z), axis=1),
        alpha=kernel_alpha,
        gamma=kernel_gamma,
    )
    nuisance_raw_rbf = _r2(kernel_test_n, nuisance_raw_rbf_prediction)
    nuisance_baseline_rbf = _r2(
        kernel_test_n, nuisance_baseline_rbf_prediction
    )
    nuisance_full_rbf = _r2(kernel_test_n, nuisance_full_rbf_prediction)

    nuisance_from_observation_linear = _r2(
        test_n, _linear_prediction(train_x, train_n, test_x)
    )
    signal_from_observation_linear = _r2(
        test_s, _linear_prediction(train_x, train_s, test_x)
    )
    nuisance_from_observation_rbf_prediction, observation_gamma = _rbf_prediction(
        kernel_train_x,
        kernel_train_n,
        kernel_test_x,
        alpha=kernel_alpha,
        gamma=kernel_gamma,
    )
    signal_from_observation_rbf_prediction, signal_observation_gamma = _rbf_prediction(
        kernel_train_x,
        kernel_train_s,
        kernel_test_x,
        alpha=kernel_alpha,
        gamma=kernel_gamma,
    )
    nuisance_from_observation_rbf = _r2(
        kernel_test_n, nuisance_from_observation_rbf_prediction
    )
    signal_from_observation_rbf = _r2(
        kernel_test_s, signal_from_observation_rbf_prediction
    )

    linear_increment = nuisance_full_linear - nuisance_baseline_linear
    rbf_increment = nuisance_full_rbf - nuisance_baseline_rbf
    observation_linear_gain = (
        nuisance_from_observation_linear - nuisance_baseline_linear
    )
    observation_rbf_gain = (
        nuisance_from_observation_rbf - nuisance_baseline_rbf
    )

    gaussian: dict[str, Any] = {}
    model_metrics = _transition_metrics("model", model_transition)
    source_metrics = _transition_metrics("source", source_transition)
    if model_metrics is not None:
        gaussian["model"] = model_metrics
    if source_metrics is not None:
        gaussian["source"] = source_metrics

    result: dict[str, Any] = {
        "metadata": {
            "n_train": int(len(train_z_array)),
            "n_test": int(len(test_z_array)),
            "representation_dim": int(train_z_array.shape[1]),
            "signal_dim": int(train_s.shape[1]),
            "nuisance_dim": int(train_n.shape[1]),
            "observation_dim": int(train_x.shape[1]),
            "kernel_n_train": int(len(train_indices)),
            "kernel_n_test": int(len(test_indices)),
            "kernel_alpha": float(kernel_alpha),
            "kernel_gamma": None if kernel_gamma is None else float(kernel_gamma),
            "random_state": int(random_state),
        },
        "signal": {"affine_r2": _scores("signal", signal_linear)},
        "nuisance_leakage": {
            "raw": {
                "linear_r2": _scores("nuisance", nuisance_raw_linear),
                "rbf_r2": _scores("nuisance", nuisance_raw_rbf),
                "rbf_gamma": raw_gamma,
            },
            "conditional": {
                "linear_baseline_r2": _scores(
                    "nuisance", nuisance_baseline_linear
                ),
                "linear_with_representation_r2": _scores(
                    "nuisance", nuisance_full_linear
                ),
                "linear_incremental_r2": _scores("nuisance", linear_increment),
                "rbf_baseline_r2": _scores("nuisance", nuisance_baseline_rbf),
                "rbf_with_representation_r2": _scores(
                    "nuisance", nuisance_full_rbf
                ),
                "rbf_incremental_r2": _scores("nuisance", rbf_increment),
                "rbf_baseline_gamma": baseline_gamma,
                "rbf_with_representation_gamma": full_gamma,
            },
        },
        "observation_controls": {
            "nuisance_from_signal_linear_r2": _scores(
                "nuisance", nuisance_baseline_linear
            ),
            "nuisance_from_observation_linear_r2": _scores(
                "nuisance", nuisance_from_observation_linear
            ),
            "nuisance_observation_linear_gain": _scores(
                "nuisance", observation_linear_gain
            ),
            "nuisance_from_signal_rbf_r2": _scores(
                "nuisance", nuisance_baseline_rbf
            ),
            "nuisance_from_observation_rbf_r2": _scores(
                "nuisance", nuisance_from_observation_rbf
            ),
            "nuisance_observation_rbf_gain": _scores(
                "nuisance", observation_rbf_gain
            ),
            "signal_from_observation_linear_r2": _scores(
                "signal", signal_from_observation_linear
            ),
            "signal_from_observation_rbf_r2": _scores(
                "signal", signal_from_observation_rbf
            ),
            "rbf_observation_gamma": observation_gamma,
            "rbf_signal_observation_gamma": signal_observation_gamma,
        },
    }
    if gaussian:
        result["gaussian_conditional"] = gaussian
    return result


__all__ = ["evaluate_initial_distribution", "evaluate_random_step_nuisance"]
