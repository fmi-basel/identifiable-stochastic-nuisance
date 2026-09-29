from torch.utils.data import Dataset, DataLoader
from implicit_nuisance.utils.misc import omegaconf_select
import omegaconf
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import tqdm

class RandomStep(Dataset):

    _VALID_SPLITS = ["train", "test"]

    def __init__(self, cfg, split="train"):
        """
        Initialize the RandomStep dataset.
        Args:
            cfg (omegaconf.DictConfig): Configuration object.
            split (str): Split type, either "train" or "test".

        Cfg.data.settings contains:
            - latent_dim (int): dimension of the latent space.
            - image_dim (int): dimension of the images.
            - num_train_sequences (int): number of training sequences.
            - num_test_sequences (int): number of test sequences.
            - sequence_length (int): number of steps per sequence.
            - prediction_noise (float): noise level for predictions.
        """

        super().__init__()
        cfg = self.add_and_assert_specific_cfg(cfg)
        self.data_cfg = cfg.data
        assert split in self._VALID_SPLITS, f"split must be one of {self._VALID_SPLITS}"
        self.split = split 

        c = self.data_cfg.settings
        
        self.latent_dim = c.latent_dim
        self.image_dim = c.image_dim
        self.sequence_length = c.sequence_length
        assert self.sequence_length >= 2, "sequence_length must be at least 2"
        assert self.latent_dim * 4 < self.image_dim , "latent_dim must be less than image_dim/4 for the generation function"
        if self.split == "train":
            self.num_sequences = c.num_train_sequences
        else:
            self.num_sequences = c.num_test_sequences
        self.indices = np.arange(self.num_sequences)
        self.device = cfg.device
        
        # noise options
        self.prediction_noise = c.prediction_noise
        self.prediction_noise_beta = c.prediction_noise_beta

        # seed for reproducibility
        np.random.seed(c.seed)
        torch.manual_seed(c.seed)

        # create random rotation matrix for prediction
        R = np.linalg.qr(np.random.randn(self.latent_dim, self.latent_dim))
        self.prediction_matrix = torch.from_numpy(R[0].astype(np.float32))

        # create random invertible network for generation
        self.generation_function = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.image_dim // 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.image_dim // 4, self.image_dim // 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.image_dim // 2, self.image_dim)
        )
        # ensure generation function is invertible by checking ranks of matrices
        with torch.no_grad():
            for layer in self.generation_function:
                if isinstance(layer, torch.nn.Linear):
                    while torch.linalg.matrix_rank(layer.weight) < min(layer.weight.shape):
                        layer.weight.copy_(torch.randn_like(layer.weight))

        for param in self.generation_function.parameters():
            param.requires_grad_(False)

        
        # precompute all samples to avoid having nn.Module in DataLoader workers
        z_steps = [torch.randn(self.num_sequences, self.latent_dim)]
        for _ in range(1, self.sequence_length):
            noises = self.prediction_noise * self._gen_normal(
                (self.num_sequences, self.latent_dim),
                beta=self.prediction_noise_beta,
            )
            z_next = (self.prediction_matrix @ z_steps[-1].T).T + noises
            # # if first dim negative, flip sign of second dim to ensure non-trivial prediction task
            # if self.latent_dim > 1:
            #     sign_flips = z_next[:, 0] < 0
            #     z_next[:, 1] *= 1 - 2 * sign_flips
            # gentle nonlinear function
            z_next = z_next + 0.1 * torch.tanh(z_next)
            z_steps.append(z_next)

        self.zs = torch.stack(z_steps, dim=1)  # (N, sequence_length, latent_dim)
        flat_zs = self.zs.reshape(self.num_sequences * self.sequence_length, self.latent_dim)
        self.xs = self.generation_function(flat_zs).reshape(
            self.num_sequences, self.sequence_length, self.image_dim
        )  # (N, sequence_length, image_dim)

    def add_and_assert_specific_cfg(self, cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds specific default values/checks for the dataset config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """
        
        cfg.data.settings.image_dim = omegaconf_select(cfg, "data.settings.image_dim")
        cfg.data.settings.latent_dim = omegaconf_select(cfg, "data.settings.latent_dim")
        cfg.data.settings.sequence_length = omegaconf_select(cfg, "data.settings.sequence_length", default=2)
        cfg.data.settings.num_train_sequences = omegaconf_select(cfg, "data.settings.num_train_sequences")
        cfg.data.settings.num_test_sequences = omegaconf_select(cfg, "data.settings.num_test_sequences")
        cfg.data.settings.prediction_noise = omegaconf_select(cfg, "data.settings.prediction_noise")
        cfg.data.settings.prediction_noise_beta = omegaconf_select(cfg, "data.settings.prediction_noise_beta", default=2.0) # generalized gaussian noise beta parameter
        cfg.data.settings.seed = omegaconf_select(cfg, "data.settings.seed", default=1)

        return cfg

    def __getitem__(self, idx):
        """
        Get one generated sequence from the dataset.
        """
        return self.xs[idx], {"pos": self.zs[idx]}

    def __len__(self):
        """Return the number of sequences in the split"""
        return len(self.indices)

    
    def _gen_normal(self, size, mu=0.0, alpha=1.0, beta=2.0):
        if isinstance(size, int):
            size = (size,)
        g = torch.distributions.Gamma(1/beta, 1).sample(size)
        return mu + alpha * g**(1/beta) * torch.sign(torch.rand(size) - 0.5)
