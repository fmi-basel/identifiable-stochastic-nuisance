"""Action-conditioned InfoLDM with a fixed conditional covariance."""

from typing import Any

import omegaconf
import torch

from implicit_nuisance.backbones.mlp import MLP
from implicit_nuisance.methods.infoLDM import InfoLDM, InfoLDMEncoder
from implicit_nuisance.utils.misc import omegaconf_select


class InfoLDMActionEncoder(InfoLDMEncoder):
    """InfoLDM encoder predicting each successor from history and action."""

    def __init__(
        self,
        *args: Any,
        action_dim: int,
        action_encoder_hidden_dim: int | None = None,
        predictor_hidden_dim: int | None = None,
        encode_action_sequences: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if predictor_hidden_dim is None:
            predictor_hidden_dim = self.rnn_hidden_dim
        if action_encoder_hidden_dim is None:
            action_encoder_hidden_dim = action_dim
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        if action_encoder_hidden_dim < 1:
            raise ValueError("action_encoder_hidden_dim must be positive")
        if predictor_hidden_dim < 1:
            raise ValueError("predictor_hidden_dim must be positive")

        self.action_dim = int(action_dim)
        self.action_encoder_hidden_dim = int(action_encoder_hidden_dim)
        self.predictor_hidden_dim = int(predictor_hidden_dim)
        self.encode_action_sequences = bool(encode_action_sequences)
        self.encoded_action_dim = (
            self.action_encoder_hidden_dim
            if self.encode_action_sequences
            else self.action_dim
        )
        if self.encode_action_sequences:
            self.action_encoder = torch.nn.GRU(
                input_size=self.action_dim,
                hidden_size=self.action_encoder_hidden_dim,
                batch_first=True,
            )
        self.predictor = MLP(
            input_dim=self.rnn_hidden_dim + self.encoded_action_dim,
            hidden_dim=self.predictor_hidden_dim,
            output_dim=self.state_dim,
            n_hidden_layers=1,
        )

    def encode_actions(
        self,
        actions: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Encode one or more ordered controls for each rendered transition."""

        expected_prefix = tuple(reference.shape[:2])
        if actions.ndim == 3:
            if tuple(actions.shape) != (*expected_prefix, self.action_dim):
                raise ValueError(
                    "actions must have shape [batch, sequence, action_dim] or "
                    "[batch, sequence, frame_skip, action_dim], got "
                    f"{tuple(actions.shape)}"
                )
            if not self.encode_action_sequences:
                return actions.to(device=reference.device, dtype=reference.dtype)
            actions = actions.unsqueeze(-2)
        elif actions.ndim == 4:
            if (
                tuple(actions.shape[:2]) != expected_prefix
                or actions.shape[2] < 1
                or actions.shape[3] != self.action_dim
            ):
                raise ValueError(
                    "actions must have shape [batch, sequence, action_dim] or "
                    "[batch, sequence, frame_skip, action_dim], got "
                    f"{tuple(actions.shape)}"
                )
            if not self.encode_action_sequences:
                raise ValueError(
                    "four-dimensional action sequences require "
                    "method_kwargs.encode_action_sequences=true"
                )
        else:
            raise ValueError(
                "actions must have shape [batch, sequence, action_dim] or "
                "[batch, sequence, frame_skip, action_dim], got "
                f"{tuple(actions.shape)}"
            )

        actions = actions.to(device=reference.device, dtype=reference.dtype)
        flat_actions = actions.reshape(-1, actions.shape[-2], self.action_dim)
        _, hidden = self.action_encoder(flat_actions)
        return hidden[-1].reshape(*expected_prefix, self.encoded_action_dim)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.encoder_backbone == "resnet18":
            if observations.ndim != 5:
                raise ValueError(
                    "resnet18 observations must have shape "
                    "[batch, sequence, channels, height, width]"
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
        encoded_actions = self.encode_actions(actions, estimates)
        predictions = self.predictor(
            torch.cat((estimates, encoded_actions), dim=-1)
        )
        if self.conditional_distribution == "dirichlet":
            predictions = torch.softmax(predictions, dim=-1)

        if isinstance(hidden_states, tuple):  # LSTM
            hidden_states = hidden_states[0]

        states = {
            "inferences": inferences,
            "estimates": estimates,
            "predictions": predictions,
            "encoded_actions": encoded_actions,
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


class InfoLDMAction(InfoLDM):
    """InfoLDM whose predictor conditions on the action at each time step."""

    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__(cfg)
        self.action_dim = int(cfg.method_kwargs.action_dim)
        self.action_encoder_hidden_dim = int(
            cfg.method_kwargs.action_encoder_hidden_dim
        )
        self.predictor_hidden_dim = int(cfg.method_kwargs.predictor_hidden_dim)
        self.encode_action_sequences = bool(
            cfg.method_kwargs.encode_action_sequences
        )
        self.encoder = InfoLDMActionEncoder(
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
            action_dim=self.action_dim,
            action_encoder_hidden_dim=self.action_encoder_hidden_dim,
            predictor_hidden_dim=self.predictor_hidden_dim,
            encode_action_sequences=self.encode_action_sequences,
        )

    @staticmethod
    def add_and_assert_specific_cfg(
        cfg: omegaconf.DictConfig,
    ) -> omegaconf.DictConfig:
        cfg = InfoLDM.add_and_assert_specific_cfg(cfg)

        action_dim = omegaconf.OmegaConf.select(cfg, "method_kwargs.action_dim")
        assert action_dim is not None, "method_kwargs.action_dim is required"
        cfg.method_kwargs.action_dim = int(action_dim)
        assert cfg.method_kwargs.action_dim > 0, (
            "method_kwargs.action_dim must be positive"
        )
        cfg.method_kwargs.action_encoder_hidden_dim = int(
            omegaconf_select(
                cfg,
                "method_kwargs.action_encoder_hidden_dim",
                cfg.method_kwargs.action_dim,
            )
        )
        assert cfg.method_kwargs.action_encoder_hidden_dim > 0, (
            "method_kwargs.action_encoder_hidden_dim must be positive"
        )

        cfg.method_kwargs.predictor_hidden_dim = int(
            omegaconf_select(
                cfg,
                "method_kwargs.predictor_hidden_dim",
                cfg.method_kwargs.rnn_hidden_dim,
            )
        )
        assert cfg.method_kwargs.predictor_hidden_dim > 0, (
            "method_kwargs.predictor_hidden_dim must be positive"
        )
        cfg.method_kwargs.encode_action_sequences = bool(
            omegaconf_select(
                cfg,
                "method_kwargs.encode_action_sequences",
                False,
            )
        )
        return cfg

    def forward(
        self,
        X: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, Any]:
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        latents = self.encoder(X, actions)
        target_estimates = self.decoder(
            {name: latent for name, latent in latents.items() if latent is not None}
        )
        return {"latents": latents, "target_estimates": target_estimates}

    @property
    def learnable_params(self) -> list[dict[str, Any]]:
        parameters = super().learnable_params
        if self.encode_action_sequences:
            parameters.append(
                {
                    "params": self.encoder.action_encoder.parameters(),
                    "lr": self.predictor_lr_multiplier * self.lr,
                }
            )
        return parameters

    def _apply_model(
        self,
        X: torch.Tensor,
        targets: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        if "actions" not in targets:
            raise KeyError("action-conditioned InfoLDM requires targets['actions']")
        out = self(X, targets["actions"])
        targets["observations"] = X
        out["decoder_losses"] = self.decoder.loss(
            out["target_estimates"], targets
        )
        return out


__all__ = ["InfoLDMAction", "InfoLDMActionEncoder"]
