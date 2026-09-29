import omegaconf
import torch
import torch.nn as nn

from torch.nn import Linear


class LinearDecoder(nn.Module):
    """Linear decoder class.

    Args:
        in_dim (int): Dimension of the input.
        out_dim (int): Dimension of the output.
        **kwargs: Additional arguments for the Linear layer.
    """

    def __init__(self, in_dim: int, out_dim: int, **kwargs) -> None:
        super().__init__()
        self.linear = Linear(in_dim, out_dim, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)