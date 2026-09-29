

from typing import Dict, List

import torch




def weighted_mean(outputs: List[Dict], key: str, batch_size_key: str) -> float:
    """Computes the mean of the values of a key weighted by the batch size.

    Args:
        outputs (List[Dict]): list of dicts containing the outputs of a validation step.
        key (str): key of the metric of interest.
        batch_size_key (str): key of batch size values.

    Returns:
        float: weighted mean of the values of a key
    """

    value = 0
    n = 0
    for out in outputs:
        value += out[batch_size_key] * out[key]
        n += out[batch_size_key]
    value = value / n
    return value.squeeze(0)


def gauss_cross_entropy(z, mu, Sigma):
        """
        Compute the Gaussian cross-entropy loss for a sample.
        
        Parameters:
            z (torch.Tensor): Sampled state with dim (batch_size, state_dim).
            mu (torch.Tensor): Inferred pseudo observation with dim (batch_size, state_dim).
            Sigma (torch.Tensor): Covariance of the gauss with dim (batch_size, state_dim, state_dim).
        
        Returns:
            torch.Tensor: Computed cross-entropy loss.
        """
        Sigma_inverse = torch.inverse(Sigma)
        log_det_Sigma = torch.logdet(Sigma)

        e = mu - z
        h = 0.5 * log_det_Sigma
        # h += 0.5 * e.T @ Sigma_inverse @ e 
        h += 0.5 * torch.einsum('bj,bj->b', e, torch.einsum('bjk,bk->bj', Sigma_inverse, e))
        return h

def gauss_entropy(Sigma : torch.Tensor) -> torch.Tensor:
    """
    Compute the Gaussian entropy. Ignore the constant term involving the dimension of the Gaussian.
    
    Parameters:
        Sigma (torch.Tensor): Covariance of the Gaussian with dim (batch_size, state_dim, state_dim).
    
    Returns:
        torch.Tensor: Computed entropy.
    """
    Sigma = Sigma + 1e-6 * torch.eye(Sigma.shape[-1], device=Sigma.device)  # Add small value for numerical stability
    log_det_Sigma = torch.logdet(Sigma)
    return 0.5 * log_det_Sigma 
    


def weighted_logsumexp(x : torch.Tensor, weights : torch.Tensor = None, dim: int = -1, keepdim : bool = False) -> torch.Tensor:
    """Computes the weighted log of the sum of exponentials of input elements.

    Args:
        x (torch.Tensor): Input tensor.
        weights (torch.Tensor, optional): Weights for each element in x. Defaults to None.
        dim (int, optional): Dimension along which to compute the logsumexp. Defaults to -1.

    Returns:
        torch.Tensor: Weighted logarithm of the sum of exponentials.
    """
    if weights is None:
        return torch.logsumexp(x, dim=dim, keepdim=keepdim)
    
    c = x.max(dim=dim, keepdim=True).values
    weighted_exp = torch.exp(x - c) * weights
    return c + torch.log(torch.sum(weighted_exp, dim=dim, keepdim=keepdim))

def log_gaussian_mixture_pdf(x, weights, means, covs, min_std=1e-6):
    """
    Compute log probability of a Gaussian Mixture Model.
    
    Args:
        x: torch.Tensor, shape (N, D) - input data points
        weights: torch.Tensor, shape (K,) - mixture weights
        means: torch.Tensor, shape (K, D) - component means
        covs: torch.Tensor, shape (K, D, D) - component covariance matrices
        min_std: float - minimum standard deviation for numerical stability
    
    Returns:
        log_prob: torch.Tensor, shape (N,) - log probability for each data point
    """
    N, D = x.shape
    K = weights.shape[0]
    
    # Ensure weights sum to 1
    weights = torch.softmax(weights, dim=0)
    
    # Initialize log probabilities
    log_probs = torch.zeros(N, K, device=x.device)
    
    for k in range(K):
        # Compute log of Gaussian density for component k
        diff = x - means[k]  # (N, D)
        
        # Cholesky decomposition for numerical stability
        cov = covs[k] + min_std * torch.eye(D, device=x.device)  # Add small diagonal for stability
        chol = torch.linalg.cholesky(cov)
        
        # Log determinant
        log_det = 2 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)), dim=-1)
        
        # Mahalanobis distance term
        solve = torch.linalg.solve_triangular(chol, diff.unsqueeze(-1), upper=False)
        mahalanobis = torch.sum(solve ** 2, dim=(1, 2))
        
        # Log of Gaussian density
        log_gaussian = -0.5 * (D * torch.log(torch.tensor(2 * torch.pi)) + log_det + mahalanobis)
        
        # Add log of mixture weight
        log_probs[:, k] = log_gaussian + torch.log(weights[k])
    
    # Log-sum-exp for numerical stability
    max_log = torch.max(log_probs, dim=1)[0]
    log_prob = max_log + torch.log(torch.sum(torch.exp(log_probs - max_log.unsqueeze(1)), dim=1))
    
    return log_prob


def categorial_kl_divergence(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the KL divergence between two categorical distributions.
    
    Args:
        input (torch.Tensor): Predicted categorical distribution (batch_size, n_categories).
        target (torch.Tensor): Target categorical distribution (batch_size, n_categories).
        
    Returns:
        torch.Tensor: KL divergence for each sample in the batch.
    """
    
    kl_div = torch.sum(target * (torch.log(target + 1e-10) - torch.log(input + 1e-10)), dim=-1)
    return kl_div

