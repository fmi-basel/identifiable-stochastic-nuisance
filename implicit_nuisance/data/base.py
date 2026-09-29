

import omegaconf

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from implicit_nuisance.data import DATASETS
    
def prepare_datasets(cfg: omegaconf.DictConfig) -> Dataset:
    """Prepares the desired dataset.

    Args:
        cfg (DictConfig): the configuration file.
    Returns:
        Dataset: the desired dataset with transformations.
    """

    dataset = cfg.data.dataset

    train_dataset = DATASETS[dataset](cfg, split="train")
    val_dataset = DATASETS[dataset](cfg, split="test")

    return train_dataset, val_dataset

def prepare_dataloaders(
    train_dataset: Dataset, test_dataset: Dataset, batch_size: int = 64, num_workers: int = 4
) -> DataLoader:
    """Prepares the training dataloader for pretraining and the validation dataloader.
    Args:
        train_dataset (Dataset): the name of the dataset.
        test_dataset (Dataset): the name of the dataset.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of workers. Defaults to 4.
    Returns:
        DataLoader: the training dataloader with the desired dataset.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader

def prepare_data(
    cfg: omegaconf.DictConfig,
) -> tuple[DataLoader, DataLoader]:
    """Prepares the training and validation dataloaders.

    Args:
        cfg (DictConfig): the configuration file.

    Returns:
        tuple[DataLoader, DataLoader]: the training and validation dataloaders.
    """
    train_dataset, val_dataset = prepare_datasets(cfg)
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=cfg.optimizer.batch_size,
        num_workers=cfg.data.num_workers,
    )
    return train_loader, val_loader
