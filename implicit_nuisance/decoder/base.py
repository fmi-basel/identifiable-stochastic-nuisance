import omegaconf
import torch
import torch.nn as nn

from implicit_nuisance.losses.decoding import DECODER_LOSS_FUNCTIONS
from implicit_nuisance.utils.misc import omegaconf_select
    
class BaseDecoder(nn.Module):
    """Decoder class for the decoder module in SSL methods.
    Forward pass takes a dictionary of tensors with the inputs for each sub-decoder
    and returns a dictionary of tensors with their outputs. The names of the decoders
    are expected to be the names of the target variables.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.
    """

    # these have to be specified for each decoder
    _REQUIRED_DECODER_SETTINGS = ["input_variable", "target_variable","output_dim", "input_dim", "backbone", "lr", "loss_func"]

    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__()
        cfg = self.add_and_assert_specific_cfg(cfg)
        self.cfg: omegaconf.DictConfig = cfg 

        self.input_variables: list = [kwargs.input_variable for kwargs in cfg.decoder_kwargs.values()]
        self.loss_funcs: list = [kwargs.loss_func for kwargs in cfg.decoder_kwargs.values()]    
        self.targets: list = [kwargs.target_variable for kwargs in cfg.decoder_kwargs.values()]
        self.decoder_settings = cfg.decoder_kwargs

        self.decoders = nn.ModuleDict()
        for decoder_name, _ in cfg.decoder_kwargs.items():
            self.decoders[decoder_name] = self._create_decoder(decoder_name)

    def _create_decoder(self, decoder_name: str) -> nn.Module:
        """Creates a decoder module based on the specified settings.

        Args:
            decoder_name (str): Name of the decoder.

        Returns:
            nn.Module: The created decoder module.
        """
        from implicit_nuisance.decoder import DECODERS

        decoder_conf = self.decoder_settings[decoder_name]
        backbone = DECODERS[decoder_conf["backbone"]]
        in_dim = decoder_conf["input_dim"]
        out_dim = decoder_conf["output_dim"]
        kwargs = decoder_conf.backbone_kwargs

        return backbone(in_dim=in_dim, out_dim=out_dim, **kwargs)

    def add_and_assert_specific_cfg(self, cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds specific default values/checks for the decoder config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """
        from implicit_nuisance.decoder import DECODERS

        cfg.decoder_kwargs = omegaconf_select(
            cfg,
            "decoder_kwargs",
            default={},
        )

        for decoder_name, decoder_conf in cfg.decoder_kwargs.items():

            # as shorthand, if target_variable is empty it is set to the name of the decoder
            decoder_conf.target_variable = omegaconf_select(
                decoder_conf,
                "target_variable",
                default=decoder_name,
            )

            decoder_conf.backbone_kwargs = omegaconf_select(
                decoder_conf,
                "backbone_kwargs",
                default={},
            )
            
            decoder_conf.detach_gradient = omegaconf_select(
                decoder_conf,
                "detach_gradient",
                default=True,
            )
            
            decoder_conf.loss_multiplier = omegaconf_select(
                decoder_conf,
                "loss_multiplier",
                default=1.0,
            )
            
            for key in self._REQUIRED_DECODER_SETTINGS:
                assert key in decoder_conf, (
                    f"Decoder settings for {decoder_name} should contain '{key}'."
                )

            # Check if the backbone is valid
            backbone = decoder_conf["backbone"]
            assert backbone in DECODERS, (
                f"Decoder backbone '{backbone}' is not supported. "
                f"Supported backbones are: {list(DECODERS.keys())}."
            )

            # Check if the loss function is valid
            loss_func = decoder_conf["loss_func"]
            assert loss_func in DECODER_LOSS_FUNCTIONS, (
                f"Decoder loss function '{loss_func}' is not supported. "
                f"Supported loss functions are: {list(DECODER_LOSS_FUNCTIONS.keys())}."
            )

        return cfg

    
    @property
    def learnable_params(self) -> list:
        """Returns the learnable parameters of the decoder.

        Returns:
            list: List of dictionaries containing the name and parameters of each decoder.
        """
        return [{
                 "name": name, 
                 "params": decoder.parameters(), 
                 "lr": self.decoder_settings[name]["lr"]
                } 
                for name, decoder in self.decoders.items()]

    
    def forward(self, X: dict) -> dict:
        """Performs forward pass of the decoder. Every decoder estimates the variable <target_variable>.

        Args:
            X (dict): Input data.

        Returns:
            dict: Output of the decoder.
        """
        
        out = {}
        for decoder_name, decoder in self.decoders.items():
            # Feed in the variable for the current decoder
            decoder_conf = self.decoder_settings[decoder_name]
            input_name = decoder_conf["input_variable"]
            input = X[input_name]
            if decoder_conf["detach_gradient"]:
                input = input.detach()
            batch_size, seq_len = input.shape[0], input.shape[1]
            input = input.view(batch_size, seq_len, -1)
            out[decoder_name] = decoder(input)
        
        return out
    
    def loss(self, Y_est: dict, Y: dict) -> dict:
        """Computes the loss for the decoder. Every decoder estimates the variable <target_variable>.

        Args:
            Y_est (dict): Input data.
            Y (dict): Target data.

        Returns:
            dict(torch.Tensor): Computed losses.
        """
        
        loss = {}
        for decoder_name, _ in self.decoders.items():
            # Feed in the variable for the current decoder
            decoder_conf = self.decoder_settings[decoder_name]
            target_name = decoder_conf["target_variable"]
            est = Y_est[decoder_name]
            
            loss_func = DECODER_LOSS_FUNCTIONS[decoder_conf["loss_func"]]
            loss[decoder_name] = decoder_conf["loss_multiplier"] * \
                loss_func(est, Y[target_name])
        
        return loss