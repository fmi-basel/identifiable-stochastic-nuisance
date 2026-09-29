import omegaconf
import torch
import torch.nn as nn

from implicit_nuisance.backbones.mlp import MLP

class MLPDecoder(nn.Module):
    """MLP decoder class.

    Args:
        in_dim (int): Dimension of the input.
        out_dim (int): Dimension of the output.
        **kwargs: Additional arguments for the MLP layer.
        
    """

    def __init__(self, in_dim: int, out_dim: int, **kwargs) -> None:
        super().__init__()

        hidden_dim, new_kwargs = self.add_and_assert_specific_cfg(**kwargs)

        self.mlp = MLP(in_dim, hidden_dim, out_dim, **new_kwargs)

    def add_and_assert_specific_cfg(self, **kwargs):
        """Adds specific default values/checks for the decoder config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """
        new_kwargs = {}
        activation = kwargs.get("activation", "ReLU")
        new_kwargs["activation"] = nn.__dict__[activation]
        out_activation = kwargs.get("output_activation", "Identity")
        new_kwargs["output_activation"] = nn.__dict__[out_activation]
        new_kwargs["n_hidden_layers"] = int(kwargs.get("n_hidden_layers", 2))
        hidden_dim = int(kwargs.get("hidden_dim", 512))
        return hidden_dim, new_kwargs
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)