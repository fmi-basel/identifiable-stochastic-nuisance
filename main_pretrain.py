
import inspect
import os

import hydra
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from implicit_nuisance.utils.misc import omegaconf_select
from implicit_nuisance.args.pretrain import parse_cfg
from implicit_nuisance.data.base import prepare_data
from implicit_nuisance.methods import METHODS
from implicit_nuisance.utils.checkpointer import Checkpointer

@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    # hydra doesn't allow us to add new keys for "safety"
    # set_struct(..., False) disables this behavior and allows us to add more parameters
    # without making the user specify every single thing about the model
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)
    cfg.device = omegaconf_select(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu")

    seed_everything(cfg.seed)

    assert cfg.method in METHODS, f"Choose from {METHODS.keys()}"

    # prepare model
    model = METHODS[cfg.method](cfg)

    ckpt_path, wandb_run_id = None, None
    if cfg.resume_from_checkpoint is not None:
        ckpt_path = cfg.resume_from_checkpoint
        wandb_run_id = cfg.resume_from_checkpoint.split("/")[-2]
        del cfg.resume_from_checkpoint
    else:
        wandb_run_id = Checkpointer.time_string() + "_" + Checkpointer.random_string()

    callbacks = []

    if cfg.checkpoint.enabled:
        ckpt = Checkpointer(
            cfg,
            logdir=os.path.join(cfg.checkpoint.dir, cfg.name),
            frequency=cfg.checkpoint.frequency,
            keep_prev=cfg.checkpoint.keep_prev,
        )
        callbacks.append(ckpt)

    # wandb logging
    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(
            name=cfg.name,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            offline=cfg.wandb.offline,
            resume="allow" if wandb_run_id else None,
            id=wandb_run_id,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    trainer_kwargs = OmegaConf.to_container(cfg)
    # we only want to pass in valid Trainer args, the rest may be user specific
    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {name: trainer_kwargs[name] for name in valid_kwargs if name in trainer_kwargs}
    trainer_kwargs.update(
        {
            "logger": wandb_logger if cfg.wandb.enabled else None,
            "callbacks": callbacks,
            "enable_checkpointing": False,
            "strategy": DDPStrategy(find_unused_parameters=False)
            if cfg.strategy == "ddp"
            else cfg.strategy,
        }
    )

    # prepare data
    train_loader, val_loader = prepare_data(cfg)

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    # destroy process group if using DDP
    if cfg.strategy == "ddp":
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
