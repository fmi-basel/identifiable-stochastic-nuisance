
from typing import Any, List, Sequence

import omegaconf
import torch
import torch.nn as nn
from implicit_nuisance.losses.ssl.infoLDM_dirichlet import (
    infoLDM_loss_func_kde as infoLDM_dirichlet_loss_func_kde,
    infoLDM_loss_func_knn as infoLDM_dirichlet_loss_func_knn,
    infoLDM_loss_func_stopgrad as infoLDM_dirichlet_loss_func_stopgrad,
)
from implicit_nuisance.losses.ssl.infoLDM_gauss import (
    infoLDM_loss_func_kde,
    infoLDM_loss_func_knn,
    infoLDM_loss_func_logdet,
    infoLDM_loss_func_stopgrad,
)
from implicit_nuisance.methods.base import BaseMethod

from implicit_nuisance.utils.misc import omegaconf_select
from implicit_nuisance.backbones.mlp import MLP

_RNN_BACKBONES = {
    "rnn": nn.RNN,
    "lstm": nn.LSTM,
    "gru": nn.GRU,
}

_CONDITIONAL_DISTRIBUTIONS = ("gaussian", "dirichlet")
_LOSS_TYPES = {
    "gaussian": ("stopgrad", "knn", "logdet", "kde"),
    "dirichlet": ("stopgrad", "knn", "kde"),
}
_ENCODER_BACKBONES = ["mlp", "resnet18"]

class InfoLDMEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        state_dim,
        encoder_hidden_dim=100,
        encoder_hidden_layers=1,
        rnn_hidden_dim=10,
        rnn_num_layers=1,
        rnn_backbone=nn.LSTM,
        encoder_backbone="mlp",
        conditional_distribution="gaussian",
        prediction_variance=1.0,
        prediction_precision=1.0,
    ):
        super(InfoLDMEncoder, self).__init__()

        self.input_dim = input_dim
        self.state_dim = state_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.conditional_distribution = conditional_distribution

        if conditional_distribution not in _CONDITIONAL_DISTRIBUTIONS:
            raise ValueError(
                "conditional_distribution must be one of "
                f"{_CONDITIONAL_DISTRIBUTIONS}"
            )

        self.encoder_backbone = encoder_backbone

        if encoder_backbone == "mlp":
            self.encoder = MLP(
                input_dim=input_dim,
                hidden_dim=encoder_hidden_dim,
                output_dim=state_dim,
                n_hidden_layers=encoder_hidden_layers,
            )
        elif encoder_backbone == "resnet18":
            from torchvision.models import resnet18

            self.encoder = resnet18(weights=None)
            self.encoder.fc = nn.Linear(self.encoder.fc.in_features, state_dim)
        else:
            raise ValueError(
                f"encoder_backbone must be one of {_ENCODER_BACKBONES}, got {encoder_backbone!r}"
            )

        self.rnn = rnn_backbone(
            input_size=state_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )

        self.predictor = nn.Linear(
            rnn_hidden_dim,
            state_dim
        )

        # self.predictor = MLP(
        #     input_dim=rnn_hidden_dim,
        #     hidden_dim=20,
        #     output_dim=state_dim,
        #     n_hidden_layers=2,
        #     # activation=nn.GELU
        # )

        if conditional_distribution == "gaussian":
            if prediction_variance <= 0:
                raise ValueError("prediction_variance must be strictly positive")
            self.register_buffer(
                "prediction_variance",
                torch.tensor(float(prediction_variance)),
            )
        else:
            if prediction_precision <= 0:
                raise ValueError("prediction_precision must be strictly positive")
            self.register_buffer(
                "prediction_precision",
                torch.tensor(float(prediction_precision)),
            )

    def forward(self, observations):
        if self.encoder_backbone == "resnet18":
            if observations.ndim != 5:
                raise ValueError(
                    "resnet18 observations must have shape [batch, sequence, channels, height, width]"
                )
            batch_size, sequence_length = observations.shape[:2]
            flat_observations = observations.reshape(
                batch_size * sequence_length, *observations.shape[2:]
            )
            inferences = self.encoder(flat_observations).reshape(
                batch_size, sequence_length, self.state_dim
            )
        else:
            inferences = self.encoder(observations)
        if self.conditional_distribution == "dirichlet":
            inferences = torch.softmax(inferences, dim=-1)

        estimates, hidden_states = self.rnn(inferences)
        predictions = self.predictor(estimates)
        if self.conditional_distribution == "dirichlet":
            predictions = torch.softmax(predictions, dim=-1)

        if isinstance(hidden_states, tuple):  # LSTM
            hidden_states = hidden_states[0]  # take only the hidden state, ignore cell

        states = {
            "inferences": inferences,
            "estimates": estimates,
            "predictions": predictions,
            "hidden_states": hidden_states,
        }
        if self.conditional_distribution == "gaussian":
            states["predictions_variances"] = (
                torch.ones_like(predictions) * self.prediction_variance
            )
        else:
            states["predictions_concentrations"] = (
                self.prediction_precision * predictions + 1.0
            )

        return states

class InfoLDM(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Predictive InfoLDM with a fixed Gaussian or Dirichlet conditional."""

        super().__init__(cfg)
        # This runs
        # cfg = self.add_and_assert_specific_cfg(cfg)
        # self.cfg = cfg

        self.input_dim = cfg.method_kwargs.input_dim
        self.state_dim = cfg.method_kwargs.state_dim
        self.rnn_hidden_dim = cfg.method_kwargs.rnn_hidden_dim
        self.encoder_hidden_dim = cfg.method_kwargs.encoder_hidden_dim
        self.encoder_hidden_layers = cfg.method_kwargs.encoder_hidden_layers
        self.encoder_backbone = cfg.method_kwargs.encoder_backbone
        self.rnn_num_layers = cfg.method_kwargs.rnn_num_layers
        self.rnn_backbone = _RNN_BACKBONES[cfg.method_kwargs.rnn_backbone]
        self.conditional_distribution = cfg.method_kwargs.conditional_distribution
        self.loss_type = cfg.method_kwargs.loss_type
        self.prediction_variance = cfg.method_kwargs.prediction_variance
        self.prediction_precision = cfg.method_kwargs.prediction_precision

        self.encoder = InfoLDMEncoder(
            input_dim=self.input_dim,
            state_dim=self.state_dim,
            encoder_hidden_dim=self.encoder_hidden_dim,
            encoder_hidden_layers=self.encoder_hidden_layers,
            rnn_hidden_dim=self.rnn_hidden_dim,
            rnn_num_layers=self.rnn_num_layers,
            rnn_backbone=self.rnn_backbone,
            encoder_backbone=self.encoder_backbone,
            conditional_distribution=self.conditional_distribution,
            prediction_variance=self.prediction_variance,
            prediction_precision=self.prediction_precision,
        )

        # learning rate multipliers
        self.encoder_lr_multiplier = cfg.method_kwargs.encoder_lr_multiplier
        self.rnn_lr_multiplier = cfg.method_kwargs.rnn_lr_multiplier
        self.predictor_lr_multiplier = cfg.method_kwargs.predictor_lr_multiplier

        self.kde_entropy_multiplier = cfg.method_kwargs.kde_entropy_multiplier


    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        # Make sure to call the parent method first
        cfg = BaseMethod.add_and_assert_specific_cfg(cfg)

        cfg.method_kwargs.encoder_backbone = omegaconf_select(
            cfg, "method_kwargs.encoder_backbone", "mlp"
        )
        assert cfg.method_kwargs.encoder_backbone in _ENCODER_BACKBONES, (
            f"Choose encoder_backbone from {_ENCODER_BACKBONES}"
        )
        if cfg.method_kwargs.encoder_backbone == "mlp":
            assert omegaconf.OmegaConf.select(cfg, "method_kwargs.input_dim") is not None, (
                "method_kwargs.input_dim is required for the MLP encoder"
            )
        else:
            cfg.method_kwargs.input_dim = omegaconf_select(
                cfg, "method_kwargs.input_dim", None
            )
        assert omegaconf.OmegaConf.select(cfg, "method_kwargs.state_dim") is not None, (
            "method_kwargs.state_dim is required"
        )
        assert omegaconf.OmegaConf.select(cfg, "method_kwargs.rnn_hidden_dim") is not None, (
            "method_kwargs.rnn_hidden_dim is required"
        )
        cfg.method_kwargs.encoder_hidden_dim = omegaconf_select(cfg, "method_kwargs.encoder_hidden_dim", 100)
        cfg.method_kwargs.encoder_hidden_layers = omegaconf_select(cfg, "method_kwargs.encoder_hidden_layers", 1)
        cfg.method_kwargs.rnn_num_layers = omegaconf_select(cfg, "method_kwargs.rnn_num_layers", 1)

        cfg.method_kwargs.encoder_lr_multiplier = omegaconf_select(cfg, "method_kwargs.encoder_lr_multiplier", 2.0)
        cfg.method_kwargs.rnn_lr_multiplier = omegaconf_select(cfg, "method_kwargs.rnn_lr_multiplier", 20.0)
        cfg.method_kwargs.predictor_lr_multiplier = omegaconf_select(cfg, "method_kwargs.predictor_lr_multiplier", 30.0)
        cfg.method_kwargs.prediction_variance = float(
            omegaconf_select(cfg, "method_kwargs.prediction_variance", 1.0)
        )
        assert cfg.method_kwargs.prediction_variance > 0, (
            "method_kwargs.prediction_variance must be strictly positive"
        )
        cfg.method_kwargs.prediction_precision = float(
            omegaconf_select(cfg, "method_kwargs.prediction_precision", 1.0)
        )
        assert cfg.method_kwargs.prediction_precision > 0, (
            "method_kwargs.prediction_precision must be strictly positive"
        )

        cfg.method_kwargs.conditional_distribution = omegaconf_select(
            cfg, "method_kwargs.conditional_distribution", "gaussian"
        )
        assert (
            cfg.method_kwargs.conditional_distribution
            in _CONDITIONAL_DISTRIBUTIONS
        ), f"Choose conditional_distribution from {_CONDITIONAL_DISTRIBUTIONS}"

        cfg.method_kwargs.kde_entropy_multiplier = omegaconf_select(cfg, "method_kwargs.kde_entropy_multiplier", 1.0)

        cfg.method_kwargs.loss_type = omegaconf_select(cfg, "method_kwargs.loss_type", "stopgrad")
        valid_loss_types = _LOSS_TYPES[cfg.method_kwargs.conditional_distribution]
        assert cfg.method_kwargs.loss_type in valid_loss_types, (
            f"Choose loss_type from {valid_loss_types} for "
            f"conditional_distribution={cfg.method_kwargs.conditional_distribution!r}"
        )

        cfg.method_kwargs.rnn_backbone = omegaconf_select(cfg, "method_kwargs.rnn_backbone", "lstm")
        assert cfg.method_kwargs.rnn_backbone in _RNN_BACKBONES, f"Choose from {_RNN_BACKBONES.keys()}"

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """
        enc = self.encoder
        extra_learnable_params = [
            {'params': enc.encoder.parameters(), 'lr': self.encoder_lr_multiplier * self.lr},
            {'params': enc.rnn.parameters(), 'lr': self.rnn_lr_multiplier * self.lr},
            {'params': enc.predictor.parameters(), 'lr': self.predictor_lr_multiplier * self.lr},
        ]
        return super().learnable_params + extra_learnable_params

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, X, Y], where
                img_indexes (torch.Tensor): indexes of the images in the batch.
                X (torch.Tensor): input data.
                Y (torch.Tensor): labels of the input data.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss 
        """
        out = super().training_step(batch, batch_idx)

        decoder_loss = out["decoder_loss"]
        latents = out["latents"]
        zs_inf = latents["inferences"]
        zs_pred = latents["predictions"]

        if self.conditional_distribution == "gaussian":
            zs_pred_variances = latents["predictions_variances"]
            if self.loss_type == "stopgrad":
                ssl_loss = infoLDM_loss_func_stopgrad(
                    zs_inf,
                    zs_pred,
                    zs_pred_variances,
                )
            elif self.loss_type == "knn":
                ssl_loss = infoLDM_loss_func_knn(
                    zs_inf,
                    zs_pred,
                    zs_pred_variances,
                )
            elif self.loss_type == "logdet":
                ssl_loss = infoLDM_loss_func_logdet(
                    zs_inf,
                    zs_pred,
                    zs_pred_variances,
                )
            elif self.loss_type == "kde":
                ssl_loss = infoLDM_loss_func_kde(
                    zs_inf,
                    zs_pred,
                    zs_pred_variances,
                    entropy_multiplier=self.kde_entropy_multiplier,
                )
        elif self.loss_type == "stopgrad":
            ssl_loss = infoLDM_dirichlet_loss_func_stopgrad(
                zs_inf,
                zs_pred,
                self.prediction_precision,
            )
        elif self.loss_type == "kde":
            ssl_loss = infoLDM_dirichlet_loss_func_kde(
                zs_inf,
                zs_pred,
                self.prediction_precision,
            )
        elif self.loss_type == "knn":
            ssl_loss = infoLDM_dirichlet_loss_func_knn(
                zs_inf,
                zs_pred,
                self.prediction_precision,
            )

        loss = ssl_loss + decoder_loss

        metrics = {
            "ssl_loss": ssl_loss,
            "loss": loss,
        }
        if self.conditional_distribution == "gaussian":
            metrics["predictions_variances"] = zs_pred_variances.mean()
        else:
            metrics["prediction_precision"] = self.encoder.prediction_precision
        self.log_dict(metrics, on_epoch=None)

        return loss

    
    
