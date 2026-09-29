"""Action-conditioned recurrent next-observation prediction."""

from typing import Any, Sequence

import omegaconf
import torch
import torch.nn.functional as F

from implicit_nuisance.decoder.image import ImageDecoder
from implicit_nuisance.methods.base import BaseMethod
from implicit_nuisance.methods.infoLDM_action import InfoLDMAction


class NextObservation(InfoLDMAction):
    """Train an InfoLDM-style recurrent predictor in next-image space."""

    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__(cfg)
        if "image" not in cfg.decoder_kwargs:
            raise ValueError(
                "next-observation prediction requires decoder_kwargs.image "
                "for the current-frame reconstruction evaluation"
            )
        image_cfg = cfg.decoder_kwargs.image
        if image_cfg.backbone != "image":
            raise ValueError("decoder_kwargs.image.backbone must be 'image'")

        self.prediction_decoder = ImageDecoder(
            in_dim=self.state_dim,
            out_dim=int(image_cfg.output_dim),
            **image_cfg.backbone_kwargs,
        )

    @property
    def learnable_params(self) -> list[dict[str, Any]]:
        return super().learnable_params + [
            {
                "name": "prediction_decoder",
                "params": self.prediction_decoder.parameters(),
                "lr": self.encoder_lr_multiplier * self.lr,
            }
        ]

    def forward(
        self,
        X: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, Any]:
        if X.ndim < 2 or X.shape[1] < 2:
            raise ValueError(
                "next-observation prediction requires at least two time steps"
            )
        out = super().forward(X, actions)
        out["target_estimates"]["next_image"] = self.prediction_decoder(
            out["latents"]["predictions"][:, :-1]
        )
        return out

    def _apply_model(self, X, targets):
        out = super()._apply_model(X, targets)
        out["decoder_losses"]["next_image"] = F.mse_loss(
            out["target_estimates"]["next_image"], X[:, 1:]
        )
        return out

    def training_step(
        self, batch: Sequence[Any], batch_idx: int
    ) -> torch.Tensor:
        # The predictor is trained in image space, without InfoLDM's latent loss.
        out = BaseMethod.training_step(self, batch, batch_idx)
        loss = out["decoder_loss"]
        self.log("loss", loss, on_epoch=True, sync_dist=True)
        return loss


__all__ = ["NextObservation"]
