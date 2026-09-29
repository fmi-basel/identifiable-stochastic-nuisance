"""InfoLDM losses for fixed diagonal-Gaussian conditionals."""

import torch

from implicit_nuisance.losses.ssl.entropy import (
    kde_entropy,
    kozachenko_leonenko_single_entropy,
    logdet_entropy,
)
from implicit_nuisance.utils.metrics import gauss_cross_entropy


def infoLDM_loss_func_stopgrad(
    zs_inf: torch.Tensor,
    zs_pred: torch.Tensor,
    zs_pred_variances: torch.Tensor,
) -> torch.Tensor:
    """Gaussian alignment with a detached inferred-state target."""

    _, n_steps, _ = zs_inf.shape
    zs_pred_covariances = torch.diag_embed(zs_pred_variances)

    alignment_sum = 0.0
    for time in range(1, n_steps):
        z_inf = zs_inf[:, time].detach()
        z_pred = zs_pred[:, time - 1]
        covariance = zs_pred_covariances[:, time - 1]
        alignment_sum += -gauss_cross_entropy(z_pred, z_inf, covariance)

    return -alignment_sum.mean()


def infoLDM_loss_func_knn(
    zs_inf: torch.Tensor,
    zs_pred: torch.Tensor,
    zs_pred_variances: torch.Tensor,
) -> torch.Tensor:
    """Gaussian InfoLDM loss with a k-nearest-neighbour entropy estimate."""

    _, n_steps, _ = zs_inf.shape
    zs_pred_covariances = torch.diag_embed(zs_pred_variances)

    alignment_sum = 0.0
    for time in range(1, n_steps):
        z_inf = zs_inf[:, time]
        z_pred = zs_pred[:, time - 1]
        covariance = zs_pred_covariances[:, time - 1]
        alignment_sum += -gauss_cross_entropy(z_pred, z_inf, covariance)

    entropies = [
        kozachenko_leonenko_single_entropy(
            zs_inf[:, time], k=3, p=2, eps=1e-8, last_only=True
        )
        for time in range(n_steps)
    ]
    entropy = torch.stack(entropies).mean()
    return -alignment_sum.mean() - entropy * (n_steps - 1)


def infoLDM_loss_func_logdet(
    zs_inf: torch.Tensor,
    zs_pred: torch.Tensor,
    zs_pred_variances: torch.Tensor,
) -> torch.Tensor:
    """Gaussian InfoLDM loss with a LogDet entropy estimate."""

    _, n_steps, _ = zs_inf.shape
    zs_pred_covariances = torch.diag_embed(zs_pred_variances)

    alignment_sum = 0.0
    for time in range(1, n_steps):
        z_inf = zs_inf[:, time]
        z_pred = zs_pred[:, time - 1]
        covariance = zs_pred_covariances[:, time - 1]
        alignment_sum += -gauss_cross_entropy(z_pred, z_inf, covariance)

    entropies = [
        logdet_entropy(zs_inf[:, time], eps=1e-8)
        for time in range(n_steps)
    ]
    entropy = torch.stack(entropies).mean()
    return -alignment_sum.mean() - entropy * (n_steps - 1)


def infoLDM_loss_func_kde(
    zs_inf: torch.Tensor,
    zs_pred: torch.Tensor,
    zs_pred_variances: torch.Tensor,
    entropy_multiplier: float = 1,
) -> torch.Tensor:
    """Gaussian InfoLDM loss with a Gaussian-KDE entropy estimate."""

    _, n_steps, _ = zs_inf.shape
    bandwidth = torch.mean(torch.sqrt(zs_pred_variances)).item()
    bandwidth /= entropy_multiplier
    zs_pred_covariances = torch.diag_embed(zs_pred_variances)

    alignment_sum = 0.0
    entropies = []
    for time in range(1, n_steps):
        z_inf = zs_inf[:, time]
        z_pred = zs_pred[:, time - 1]
        covariance = zs_pred_covariances[:, time - 1]
        alignment_sum += -gauss_cross_entropy(z_pred, z_inf, covariance)
        entropies.append(
            kde_entropy(
                z_inf,
                bandwidth=bandwidth,
                eps=1e-8,
                positive=z_pred,
            )
        )

    entropy = torch.stack(entropies).mean()
    return -alignment_sum.mean() - entropy * (n_steps - 1)


__all__ = [
    "infoLDM_loss_func_kde",
    "infoLDM_loss_func_knn",
    "infoLDM_loss_func_logdet",
    "infoLDM_loss_func_stopgrad",
]
