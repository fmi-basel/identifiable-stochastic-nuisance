"""InfoLDM losses for Dirichlet conditionals on the simplex."""

import torch

from implicit_nuisance.losses.ssl.entropy import (
    dirichlet_kde_entropy,
    kozachenko_leonenko_single_entropy,
)


def dirichlet_conditional_log_prob(
    target: torch.Tensor,
    mode: torch.Tensor,
    precision: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Unnormalized ``Dirichlet(precision * mode + 1)`` log density.

    This matches ``dirichlet_log_pdf_approx`` from the reference
    implementation, including its fixed logarithm floor.
    """

    if precision <= 0:
        raise ValueError("precision must be strictly positive")
    if eps <= 0:
        raise ValueError("eps must be strictly positive")

    scaled_mode = precision * mode
    log_prob = (scaled_mode * torch.log(target + eps)).sum(dim=-1)
    log_prob -= torch.lgamma(scaled_mode + 1.0).sum(dim=-1)
    return log_prob


def infoLDM_loss_func_stopgrad(
    zs_inf: torch.Tensor,
    zs_pred: torch.Tensor,
    prediction_precision: float,
) -> torch.Tensor:
    """Dirichlet alignment with a detached inferred-state target."""

    _, n_steps, _ = zs_inf.shape
    alignment_sum = 0.0
    for time in range(1, n_steps):
        alignment_sum += dirichlet_conditional_log_prob(
            target=zs_inf[:, time].detach(),
            mode=zs_pred[:, time - 1],
            precision=prediction_precision,
        )
    return -alignment_sum.mean()


def infoLDM_loss_func_kde(
    zs_inf: torch.Tensor,
    zs_pred: torch.Tensor,
    prediction_precision: float,
) -> torch.Tensor:
    """Dirichlet alignment plus the Dirichlet-KDE entropy from Eq. (51)."""

    _, n_steps, _ = zs_inf.shape
    alignment_sum = 0.0
    entropies = []
    for time in range(1, n_steps):
        alignment_sum += dirichlet_conditional_log_prob(
            target=zs_inf[:, time],
            mode=zs_pred[:, time - 1],
            precision=prediction_precision,
        )
        entropies.append(
            dirichlet_kde_entropy(
                zs_inf[:, time], precision=prediction_precision
            )
        )

    entropy = torch.stack(entropies).mean()
    return -alignment_sum.mean() - entropy * (n_steps - 1)


def infoLDM_loss_func_knn(
    zs_inf: torch.Tensor,
    zs_pred: torch.Tensor,
    prediction_precision: float,
) -> torch.Tensor:
    """Dirichlet alignment plus raw-simplex kNN entropy regularization."""

    _, n_steps, _ = zs_inf.shape
    alignment_sum = 0.0
    entropies = []
    for time in range(1, n_steps):
        alignment_sum += dirichlet_conditional_log_prob(
            target=zs_inf[:, time],
            mode=zs_pred[:, time - 1],
            precision=prediction_precision,
        )
        entropies.append(
            kozachenko_leonenko_single_entropy(zs_inf[:, time])
        )

    entropy = torch.stack(entropies).mean()
    return -alignment_sum.mean() - entropy * (n_steps - 1)


__all__ = [
    "dirichlet_conditional_log_prob",
    "infoLDM_loss_func_kde",
    "infoLDM_loss_func_knn",
    "infoLDM_loss_func_stopgrad",
]
