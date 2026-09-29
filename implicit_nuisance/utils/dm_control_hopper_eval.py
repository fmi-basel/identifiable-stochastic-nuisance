"""Held-out linear probes for physical DM-Control Hopper data."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


POSITION_NAMES = (
    "root_height",
    "torso_pitch_sin",
    "torso_pitch_cos",
    "waist",
    "hip",
    "knee",
    "ankle",
)
VELOCITY_NAMES = (
    "root_x_velocity",
    "root_z_velocity",
    "torso_angular_velocity",
    "waist_velocity",
    "hip_velocity",
    "knee_velocity",
    "ankle_velocity",
)
TOUCH_NAMES = ("toe_touch", "heel_touch")
NUISANCE_NAMES = (
    "body_red",
    "body_green",
    "body_blue",
    "accent_red",
    "accent_green",
    "accent_blue",
    "camera_x",
    "camera_z",
    "camera_fovy",
    "light_position",
    "light_diffuse",
    "background",
)


def _matrix(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got {array.shape}")
    if array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least two rows")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    return array


def _linear_prediction(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_x)
    scaled_test = scaler.transform(test_x)
    return LinearRegression().fit(scaled_train, train_y).predict(scaled_test)


def _r2(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.atleast_1d(
        np.asarray(r2_score(target, prediction, multioutput="raw_values"))
    )


def _scores(names: tuple[str, ...], values: np.ndarray) -> dict[str, Any]:
    if len(names) != len(values):
        raise ValueError(f"expected {len(names)} scores, got {len(values)}")
    return {
        "per_dimension": {
            name: float(value) for name, value in zip(names, values, strict=True)
        },
        "mean": float(np.mean(values)),
    }


def evaluate_hopper_representations(
    train_z: Any,
    train_position: Any,
    train_velocity: Any,
    train_touch: Any,
    train_nuisance: Any,
    test_z: Any,
    test_position: Any,
    test_velocity: Any,
    test_touch: Any,
    test_nuisance: Any,
) -> dict[str, Any]:
    """Fit all probes on train frames and score them on held-out test frames."""

    train = {
        "z": _matrix("train_z", train_z),
        "position": _matrix("train_position", train_position),
        "velocity": _matrix("train_velocity", train_velocity),
        "touch": _matrix("train_touch", train_touch),
        "nuisance": _matrix("train_nuisance", train_nuisance),
    }
    test = {
        "z": _matrix("test_z", test_z),
        "position": _matrix("test_position", test_position),
        "velocity": _matrix("test_velocity", test_velocity),
        "touch": _matrix("test_touch", test_touch),
        "nuisance": _matrix("test_nuisance", test_nuisance),
    }
    for split_name, arrays in (("train", train), ("test", test)):
        row_counts = {name: len(array) for name, array in arrays.items()}
        if len(set(row_counts.values())) != 1:
            raise ValueError(
                f"{split_name} arrays have inconsistent row counts: {row_counts}"
            )
    for name in train:
        if train[name].shape[1] != test[name].shape[1]:
            raise ValueError(f"train/test {name} dimensions differ")
    expected_dimensions = {
        "position": len(POSITION_NAMES),
        "velocity": len(VELOCITY_NAMES),
        "touch": len(TOUCH_NAMES),
        "nuisance": len(NUISANCE_NAMES),
    }
    for name, expected in expected_dimensions.items():
        if train[name].shape[1] != expected:
            raise ValueError(
                f"{name} must have {expected} columns, got {train[name].shape[1]}"
            )

    position_r2 = _r2(
        test["position"],
        _linear_prediction(train["z"], train["position"], test["z"]),
    )
    velocity_r2 = _r2(
        test["velocity"],
        _linear_prediction(train["z"], train["velocity"], test["z"]),
    )
    touch_r2 = _r2(
        test["touch"],
        _linear_prediction(train["z"], train["touch"], test["z"]),
    )
    nuisance_r2 = _r2(
        test["nuisance"],
        _linear_prediction(train["z"], train["nuisance"], test["z"]),
    )
    nuisance_baseline_r2 = _r2(
        test["nuisance"],
        _linear_prediction(
            train["position"], train["nuisance"], test["position"]
        ),
    )
    nuisance_with_z_r2 = _r2(
        test["nuisance"],
        _linear_prediction(
            np.concatenate((train["position"], train["z"]), axis=1),
            train["nuisance"],
            np.concatenate((test["position"], test["z"]), axis=1),
        ),
    )
    nuisance_incremental_r2 = nuisance_with_z_r2 - nuisance_baseline_r2

    return {
        "metadata": {
            "n_train": int(len(train["z"])),
            "n_test": int(len(test["z"])),
            "representation_dim": int(train["z"].shape[1]),
            "probe": "held_out_standardized_linear_regression",
        },
        "linear_decoding": {
            "position": _scores(POSITION_NAMES, position_r2),
            "velocity": _scores(VELOCITY_NAMES, velocity_r2),
            "touch": _scores(TOUCH_NAMES, touch_r2),
            "nuisance": _scores(NUISANCE_NAMES, nuisance_r2),
        },
        "conditional_nuisance_leakage": {
            "baseline_from_position": _scores(
                NUISANCE_NAMES, nuisance_baseline_r2
            ),
            "with_representation": _scores(NUISANCE_NAMES, nuisance_with_z_r2),
            "incremental_r2": _scores(
                NUISANCE_NAMES, nuisance_incremental_r2
            ),
        },
    }


def evaluate_hopper_learned_decoders(
    test_position: Any,
    predicted_position: Any,
    test_velocity: Any,
    predicted_velocity: Any,
    test_nuisance: Any,
    predicted_nuisance: Any,
) -> dict[str, Any]:
    """Score checkpoint-learned nonlinear decoders on held-out test frames."""

    pairs = {
        "position": (
            POSITION_NAMES,
            _matrix("test_position", test_position),
            _matrix("predicted_position", predicted_position),
        ),
        "velocity": (
            VELOCITY_NAMES,
            _matrix("test_velocity", test_velocity),
            _matrix("predicted_velocity", predicted_velocity),
        ),
        "nuisance": (
            NUISANCE_NAMES,
            _matrix("test_nuisance", test_nuisance),
            _matrix("predicted_nuisance", predicted_nuisance),
        ),
    }
    scores = {}
    for key, (names, target, prediction) in pairs.items():
        if target.shape != prediction.shape:
            raise ValueError(
                f"{key} target/prediction shapes differ: "
                f"{target.shape} != {prediction.shape}"
            )
        if target.shape[1] != len(names):
            raise ValueError(
                f"{key} must have {len(names)} columns, got {target.shape[1]}"
            )
        scores[key] = _scores(names, _r2(target, prediction))
    return scores


def _plot_decoding_r2(
    metrics: dict[str, Any],
    output: Any,
    *,
    decoding_key: str,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    panels = (
        ("position", POSITION_NAMES, "Rendered position"),
        ("velocity", VELOCITY_NAMES, "Velocity"),
        ("nuisance", NUISANCE_NAMES, "View-private nuisance"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 3.5), sharey=True)
    for axis, (key, names, title) in zip(axes, panels, strict=True):
        score_map = metrics[decoding_key][key]["per_dimension"]
        values = [score_map[name] for name in names]
        positions = np.arange(len(names))
        axis.bar(positions, values)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(
            positions,
            [name.replace("_", "\n") for name in names],
            rotation=45 if len(names) > 7 else 0,
            ha="right" if len(names) > 7 else "center",
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel(ylabel)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_decoding_r2(metrics: dict[str, Any], output: Any) -> None:
    """Save position, velocity, and nuisance linear-decoding bars."""

    _plot_decoding_r2(
        metrics,
        output,
        decoding_key="linear_decoding",
        ylabel=r"Held-out linear probe $R^2$",
    )


def plot_nonlinear_decoding_r2(metrics: dict[str, Any], output: Any) -> None:
    """Save held-out R2 bars for the checkpoint-learned MLP decoders."""

    _plot_decoding_r2(
        metrics,
        output,
        decoding_key="nonlinear_decoding",
        ylabel=r"Held-out learned-decoder $R^2$",
    )


__all__ = [
    "NUISANCE_NAMES",
    "POSITION_NAMES",
    "TOUCH_NAMES",
    "VELOCITY_NAMES",
    "evaluate_hopper_learned_decoders",
    "evaluate_hopper_representations",
    "plot_decoding_r2",
    "plot_nonlinear_decoding_r2",
]
