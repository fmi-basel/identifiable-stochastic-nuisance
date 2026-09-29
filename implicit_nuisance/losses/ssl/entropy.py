"""Differentiable minibatch entropy estimators used by InfoLDM losses."""

from __future__ import annotations

import numpy as np
import torch


def knn(
    x: torch.Tensor,
    k: int = 3,
    p: float = 2,
    last_only: bool = False,
) -> torch.Tensor:
    """Return non-self k-nearest-neighbour distances within a minibatch."""

    distances = torch.cdist(x, x, p=p)
    neighbours, _ = torch.topk(distances, k + 1, largest=False)
    neighbours = neighbours[:, 1:]
    if last_only:
        neighbours = neighbours[:, -1:]
    return neighbours


def kozachenko_leonenko_single_entropy(
    x: torch.Tensor | np.ndarray,
    k: int = 3,
    p: float = 2,
    eps: float = 1e-8,
    outlier_threshold: float = 0.9,
    last_only: bool = True,
) -> torch.Tensor:
    """Estimate entropy from within-batch nearest-neighbour distances."""

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.astype(np.float32))

    neighbour_distances = knn(x, k=k, p=p, last_only=last_only)
    neighbour_distances = neighbour_distances.clamp(min=0.0, max=1e6)
    upper_bound = torch.quantile(
        neighbour_distances, outlier_threshold
    ).detach()
    neighbour_distances = neighbour_distances.clamp(max=upper_bound)
    return torch.log(neighbour_distances + eps).mean()


def logdet_entropy(
    x: torch.Tensor | np.ndarray,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Estimate entropy by fitting a full-covariance Gaussian."""

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.astype(np.float32))

    covariance = torch.cov(x.T)
    covariance = covariance + eps * torch.eye(
        covariance.shape[0], device=covariance.device, dtype=covariance.dtype
    )
    _, logdet = torch.slogdet(covariance)
    dimension = x.shape[1]
    return 0.5 * logdet + 0.5 * dimension * (1 + np.log(2 * np.pi))


def kde_entropy(
    x: torch.Tensor | np.ndarray,
    bandwidth: float = 1.0,
    eps: float = 1e-8,
    positive: torch.Tensor | None = None,
) -> torch.Tensor:
    """Estimate entropy with a leave-one-out isotropic Gaussian KDE.

    ``positive`` optionally adds one paired kernel centre per example. This is
    useful for small predictive batches, where it prevents every kernel value
    for an example from becoming negligible.
    """

    del eps  # Kept for backwards-compatible call sites.
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.astype(np.float32))
    if bandwidth <= 0:
        raise ValueError("bandwidth must be strictly positive")

    n_samples = x.shape[0]
    squared_distances = torch.cdist(x, x).square()
    log_kernel = -squared_distances / (2 * bandwidth**2)
    log_kernel = log_kernel.masked_fill(
        torch.eye(n_samples, device=x.device, dtype=torch.bool),
        -torch.inf,
    )
    if positive is not None:
        positive_squared_distances = (x - positive).square().sum(
            dim=1, keepdim=True
        )
        positive_log_kernel = -positive_squared_distances / (2 * bandwidth**2)
        log_kernel = torch.cat((log_kernel, positive_log_kernel), dim=1)

    log_density = torch.logsumexp(log_kernel, dim=1)
    return -log_density.mean()


def dirichlet_kde_entropy(
    x: torch.Tensor,
    precision: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Estimate simplex entropy with unnormalized Dirichlet kernels.

    This matches ``contrastive_single_entropy_term``: each ``x[j]``
    parameterizes a kernel proportional to
    ``Dirichlet(precision * x[j] + 1)``, while constants shared by all
    kernels and the mixture normalization are omitted.
    """

    if x.ndim != 2:
        raise ValueError("x must have shape [batch, simplex_components]")
    if x.shape[0] < 1:
        raise ValueError("x must contain at least one sample")
    if precision <= 0:
        raise ValueError("precision must be strictly positive")
    if eps <= 0:
        raise ValueError("eps must be strictly positive")

    scaled_modes = precision * x
    log_kernel = torch.log(x + eps) @ scaled_modes.T
    log_kernel -= torch.lgamma(scaled_modes + 1.0).sum(dim=1).unsqueeze(0)
    return -torch.logsumexp(log_kernel, dim=1).mean()


__all__ = [
    "dirichlet_kde_entropy",
    "kde_entropy",
    "knn",
    "kozachenko_leonenko_single_entropy",
    "logdet_entropy",
]
