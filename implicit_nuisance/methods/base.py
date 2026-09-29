

import logging
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Union

import lightning.pytorch as pl
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR

from implicit_nuisance.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from implicit_nuisance.utils.metrics import weighted_mean
from implicit_nuisance.utils.misc import omegaconf_select

from implicit_nuisance.decoder import BaseDecoder


class BaseMethod(pl.LightningModule):
    _OPTIMIZERS = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    _SCHEDULERS = [
        "reduce",
        "warmup_cosine",
        "step",
        "exponential",
        "none",
    ]

    def __init__(self, cfg: omegaconf.DictConfig):
        """Base model that implements all basic operations for all self-supervised methods.
        It adds shared arguments, extract basic learnable parameters, creates optimizers
        and schedulers, implements basic training_step, trains the online decoders and implements
        validation_step.

        .. note:: This class is not meant to be used directly, but as a base class for
            all self-supervised methods. It implements the basic operations that are
            common to all self-supervised methods. It is not meant to be used directly.
        
        .. note:: IMPORTANT!: Every child class has to implement the encoder. The encoder
            is not implemented in this class.
        
        .. note:: IMPORTANT!: The child class has to call this constructor in its own
            constructor.
        
        .. note:: IMPORTANT!: The child class has to call the `add_and_assert_specific_cfg`
            method in its own `add_and_assert_specific_cfg` method. 


        Cfg basic structure:
            data:
                dataset (str): name of the dataset.
                num_workers (int): number of workers for the dataloader.
                setting (cfg): settings for the particular dataset.
            max_epochs (int): number of training epochs.

            decoder_kwargs (cfg): names and settings for the decoders. the names
                are the names of the target variables. The settings are:
                <decoder_name>:
                    input_variable (str): name of the input variable.
                    target_variable (str): name of the target variable. (optional)
                        If not specified, <decoder_name> is used as the target variable.
                    input_dim (int): dimension of the input variable.
                    output_dim (int): dimension of the output variable.
                    backbone (str): name of the backbone.
                    backbone_kwargs (cfg): settings for the backbone (optional).
                    lr (float): learning rate.

            optimizer:
                name (str): name of the optimizer.
                batch_size (int): number of samples in the batch.
                lr (float): learning rate.
                weight_decay (float): weight decay for optimizer.
                kwargs (Dict): extra named arguments for the optimizer.
            scheduler:
                name (str): name of the scheduler.
                min_lr (float): minimum learning rate for warmup scheduler. Defaults to 0.0.
                warmup_start_lr (float): initial learning rate for warmup scheduler.
                    Defaults to 0.00003.
                warmup_epochs (float): number of warmup epochs. Defaults to 10.
                lr_decay_steps (Sequence, optional): if scheduler is 'step', the learning rate will be decreased  
                    at these timesteps (given in fraction of train time). Defaults to [0.25,0.5,0.75].
                interval (str): interval to update the lr scheduler. Defaults to 'step'.
            performance:
                disable_channel_last (bool). Disables channel last conversion operation which
                speeds up training considerably. Defaults to False.
                https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models

        """

        super().__init__()

        # add default values and assert that config has the basic needed settings
        # this can be overwritten in the child class, but the child class function
        # then has to call this method first from the base class
        cfg = self.add_and_assert_specific_cfg(cfg)

        self.cfg: omegaconf.DictConfig = cfg

        # encoder has to be defined in the child class
        # we need it here because we need to pass it to the decoder
        self.encoder: nn.Module = None

        # online evaluation
        self.decoder: nn.Module = BaseDecoder(cfg)
        self.validation_step_outputs: List[Dict[str, Any]] = []

        # training related
        self.max_epochs: int = cfg.max_epochs

        # optimizer related
        self.optimizer: str = cfg.optimizer.name
        self.batch_size: int = cfg.optimizer.batch_size
        self.lr: float = cfg.optimizer.lr
        self.weight_decay: float = cfg.optimizer.weight_decay
        self.extra_optimizer_args: Dict[str, Any] = cfg.optimizer.kwargs

        # scheduler related
        self.scheduler: str = cfg.scheduler.name
        self.lr_decay_steps: Union[List[int], None] = cfg.scheduler.lr_decay_steps
        self.min_lr: float = cfg.scheduler.min_lr
        self.warmup_start_lr: float = cfg.scheduler.warmup_start_lr
        self.warmup_epochs: int = cfg.scheduler.warmup_epochs
        self.scheduler_interval: str = cfg.scheduler.interval
        assert self.scheduler_interval in ["step", "epoch"]
        if self.scheduler_interval == "step":
            logging.warn(
                f"Using scheduler_interval={self.scheduler_interval} might generate "
                "issues when resuming a checkpoint."
            )
            print("WARNING: Using scheduler_interval=step leads to issues in some schedules.")

        # for performance
        self.no_channel_last = cfg.performance.disable_channel_last


    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.

        .. note:: IMPORTANT!: If the child class overwrites this method, it has to
            call the parent method first.
        """

        cfg.num_nodes = omegaconf_select(cfg, "num_nodes", 1)

        # extra optimizer kwargs
        cfg.optimizer.kwargs = omegaconf_select(cfg, "optimizer.kwargs", {})
        if cfg.optimizer.name == "sgd":
            cfg.optimizer.kwargs.momentum = omegaconf_select(cfg, "optimizer.kwargs.momentum", 0.9)
        elif cfg.optimizer.name == "lars":
            cfg.optimizer.kwargs.momentum = omegaconf_select(cfg, "optimizer.kwargs.momentum", 0.9)
            cfg.optimizer.kwargs.eta = omegaconf_select(cfg, "optimizer.kwargs.eta", 1e-3)
            cfg.optimizer.kwargs.clip_lr = omegaconf_select(cfg, "optimizer.kwargs.clip_lr", False)
            cfg.optimizer.kwargs.exclude_bias_n_norm = omegaconf_select(
                cfg,
                "optimizer.kwargs.exclude_bias_n_norm",
                False,
            )
        elif cfg.optimizer.name == "adamw":
            cfg.optimizer.kwargs.betas = omegaconf_select(cfg, "optimizer.kwargs.betas", [0.9, 0.999])

        # scheduler
        cfg.scheduler.lr_decay_steps = omegaconf_select(
            cfg, "scheduler.lr_decay_steps", [0.25,0.5,0.75]
        )
        cfg.scheduler.min_lr = omegaconf_select(cfg, "scheduler.min_lr", 0.0)
        cfg.scheduler.warmup_start_lr = omegaconf_select(
            cfg, "scheduler.warmup_start_lr", 0.0
        )
        cfg.scheduler.warmup_epochs = omegaconf_select(
            cfg, "scheduler.warmup_epochs", 10
        )
        cfg.scheduler.interval = omegaconf_select(
            cfg, "scheduler.interval", "epoch"
        )

        # performance related
        cfg.performance = omegaconf_select(cfg, "performance", {})
        cfg.performance.disable_channel_last = omegaconf_select(
            cfg, "performance.disable_channel_last", True
        )

        return cfg

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        return self.decoder.learnable_params

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        learnable_params = self.learnable_params

        assert self.optimizer in self._OPTIMIZERS
        optimizer = self._OPTIMIZERS[self.optimizer]

        # create optimizer
        optimizer = optimizer(
            learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        # create scheduler
        scheduler = self._get_scheduler(optimizer)

        return [optimizer], [scheduler]

    def _get_scheduler(self, optimizer) -> Callable:
        if self.scheduler.lower() == "none":
            return optimizer

        if self.scheduler == "warmup_cosine":
            max_warmup_steps = (
                self.warmup_epochs * (self.trainer.estimated_stepping_batches / self.max_epochs)
                if self.scheduler_interval == "step"
                else self.warmup_epochs
            )
            max_scheduler_steps = (
                self.trainer.estimated_stepping_batches
                if self.scheduler_interval == "step"
                else self.max_epochs
            )
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=max_warmup_steps,
                    max_epochs=max_scheduler_steps,
                    warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr,
                    eta_min=self.min_lr,
                ),
                "interval": self.scheduler_interval,
                "frequency": 1,
            }
        elif self.scheduler == "step":
            max_scheduler_steps = (
                self.trainer.estimated_stepping_batches
                if self.scheduler_interval == "step"
                else self.max_epochs
            )
            s = [int(x * max_scheduler_steps) for x in self.lr_decay_steps]
            scheduler = MultiStepLR(optimizer, s, gamma=0.5)
        elif self.scheduler == "linear":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=self.min_lr / self.lr,
                    total_iters=self.max_epochs,
                ),
                "interval": self.scheduler_interval,
                "frequency": 1,
            }
        elif self.scheduler == "exponential":
            gamma = np.pow(self.min_lr / self.lr, 1.0 / self.max_epochs)
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma=gamma
                ),
                "interval": self.scheduler_interval,
                "frequency": 1,
            }
        else:
            raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step, linear, none)")
        
        return scheduler

    def forward(self, X) -> Dict:
        """Basic forward method. Children methods should call this function,
        modify the ouputs (without deleting anything) and return it.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of latents and target estimates.
        """

        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        latents = self.encoder(X)
        target_estimates = self.decoder({name : latent for name, latent in latents.items() if latent is not None})
        
        return {"latents": latents, "target_estimates": target_estimates}

    def _apply_model(self, X: torch.Tensor, targets: dict) -> Dict:
        """Forwards a batch of images X and computes the decoder losses.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (dict): batch of labels for X stored in dict.

        Returns:
            Dict: dict containing 
                - latents: the latents of the encoder
                - target_estimates: the estimates of the decoder
                - decoder_losses: the losses of the decoder
        """

        out = self(X)
        target_estimates = out["target_estimates"]

        # add observations to the targets
        targets["observations"] = X

        losses = self.decoder.loss(target_estimates, targets)
        out.update({"decoder_losses": losses})

        return out

    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for pytorch lightning. It does all the shared operations, such as
        forwarding the images, computing logits and computing statistics.

        Args:
            batch (List[Any]): a batch of data in the format of [img_indexes, X, Y].
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: dict with the classification loss, features and logits.
        """

        X, targets = batch

        outs = self._apply_model(X, targets)

        metrics = {
            "train_decoder_loss_" + target_name: loss for target_name, loss in outs["decoder_losses"].items()
        }

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        outs["decoder_loss"] = sum(outs["decoder_losses"].values()) 

        return outs

    def validation_step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = None,
        update_validation_step_outputs: bool = True,
    ) -> Dict[str, Any]:
        """Validation step for pytorch lightning. It does all the shared operations, such as
        forwarding a batch of images, computing logits and computing metrics.

        Args:
            batch (List[torch.Tensor]):a batch of data in the format of [img_indexes, X, Y].
            batch_idx (int): index of the batch.
            update_validation_step_outputs (bool): whether or not to append the
                metrics to validation_step_outputs

        Returns:
            Dict[str, Any]: dict with the batch_size (used for averaging), the classification loss
                and accuracies.
        """

        X, targets = batch
        batch_size = X.size(0)

        out = self._apply_model(X, targets)

        metrics = {
            "val_decoder_loss_" + target_name: loss for target_name, loss in out["decoder_losses"].items()
        }
        metrics["batch_size"] = batch_size
        if update_validation_step_outputs:
            self.validation_step_outputs.append(metrics)
        return metrics

    def on_validation_epoch_end(self):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.
        """

        log = {}
        loss_keys = [key for key in self.validation_step_outputs[0] if not key == "batch_size"]
        for key in loss_keys:
            log[key] = weighted_mean(self.validation_step_outputs, key, "batch_size")

        self.log_dict(log, sync_dist=True)

        self.validation_step_outputs.clear()
