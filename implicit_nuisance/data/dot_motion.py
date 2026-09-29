from torch.utils.data import Dataset, DataLoader
from implicit_nuisance.utils.misc import omegaconf_select
import omegaconf
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import tqdm

class DotMotion(Dataset):

    _VALID_MOTION_TYPES = ["circular", "square", "dual_circular", "random_square", "reflection", "figure8", "wobbly", "dual_wobbly"]
    _VALID_NOISE_TYPES = ["gaussian", "random_gaussian", "state_dependent_gaussian"]
    _VALID_POST_PROCESSING = [None, "half_flip", "distort", "random"]
    _VALID_SPLITS = ["train", "test"]

    def __init__(self, cfg, split="train"):
        """
        Initialize the DotMotion dataset.
        Args:
            cfg (omegaconf.DictConfig): Configuration object.
            split (str): Split type, either "train" or "test".

        Cfg.data.settings contains:
            - motion_type (str): type of motion.
            - noise_type (str): type of noise to apply.
            - post_processing (str): post-processing type.
            - image_dim (int): dimension of the images.
            - seq_len (int): length of the sequences.
            - num_train_sequences (int): number of training sequences.
            - num_test_sequences (int): number of test sequences.
            - observation_noise (float): noise level for observations.
            - prediction_noise (float): noise level for predictions.
            - period_duration (float): duration of the period.
            - jump_rate (float): rate of jumps in the trajectory.
            - dt (float): time step for the simulation.
        """

        super().__init__()
        cfg = self.add_and_assert_specific_cfg(cfg)
        self.data_cfg = cfg.data
        assert split in self._VALID_SPLITS, f"split must be one of {self._VALID_SPLITS}"
        self.split = split 

        c = self.data_cfg.settings
        
        self.image_dim = (c.image_dim, c.image_dim)
        self.seq_len = c.seq_len
        if self.split == "train":
            self.num_sequences = c.num_train_sequences
        else:
            self.num_sequences = c.num_test_sequences
        self.indices = np.arange(self.num_sequences)
        self.device = cfg.device
        
        # noise options
        self.observation_noise = c.observation_noise
        self.prediction_noise = c.prediction_noise
        self.prediction_noise_beta = c.prediction_noise_beta
        self.noise_type = c.noise_type
        self.random_noise_level = self.noise_type == "random_gaussian"
        self.state_dependent_noise_level = self.noise_type == "state_dependent_gaussian"
        self.jump_rate = c.jump_rate

        # dynamics options
        self.motion_type = c.motion_type
        self.period_duration = c.period_duration
        self.dt = c.dt

        # seed for reproducibility
        np.random.seed(c.seed)
        torch.manual_seed(c.seed)

        # random network for post-processing
        if c.post_processing == "random":
            self.random_network = torch.nn.Sequential(
                torch.nn.Linear(2, 5),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(5, 5),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(5, 5),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(5, 5),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(5, self.image_dim[0] * self.image_dim[1], bias=False),
            ).to(self.device)
            # ensure invertibility 
            for i in [0,2,4,6]:
                layer = self.random_network[i]
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.orthogonal_(layer.weight)
            for param in self.random_network.parameters():
                param.requires_grad = False  # freeze the random network

        # Cache the meshgrid for image generation
        img_dim_x, img_dim_y = self.image_dim
        grid_x = torch.arange(img_dim_x)
        grid_y = torch.arange(img_dim_y)
        self._grid_x, self._grid_y = torch.meshgrid(grid_x, grid_y, indexing='ij')    
        
        # Generate trajectories
        self.store_videos = self.split == "test" or self.motion_type in {"wobbly", "dual_wobbly"}
        self.trajectories, self.noise_levels, self.labels = self._generate_trajectories()
        if self.store_videos:
            print("Pre-generating videos...")
            self.videos = [self._create_video(traj, noise) for traj, noise in tqdm.tqdm(zip(self.trajectories, self.noise_levels))]

    def add_and_assert_specific_cfg(self, cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds specific default values/checks for the dataset config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """
        
        cfg.data.settings.motion_type = omegaconf_select(cfg, "data.settings.motion_type")
        assert cfg.data.settings.motion_type in self._VALID_MOTION_TYPES, f"motion_type must be one of {self._VALID_MOTION_TYPES}"
        cfg.data.settings.post_processing = omegaconf_select(cfg, "data.settings.post_processing", default=None)
        assert cfg.data.settings.post_processing in self._VALID_POST_PROCESSING, f"post_processing must be one of {self._VALID_POST_PROCESSING}"
        cfg.data.settings.noise_type = omegaconf_select(cfg, "data.settings.noise_type")
        assert cfg.data.settings.noise_type in self._VALID_NOISE_TYPES, f"noise_type must be one of {self._VALID_NOISE_TYPES}"
        cfg.data.settings.period_duration = omegaconf_select(cfg, "data.settings.period_duration")
        cfg.data.settings.dt = omegaconf_select(cfg, "data.settings.dt")
        cfg.data.settings.image_dim = omegaconf_select(cfg, "data.settings.image_dim")
        cfg.data.settings.seq_len = omegaconf_select(cfg, "data.settings.seq_len")
        cfg.data.settings.num_train_sequences = omegaconf_select(cfg, "data.settings.num_train_sequences")
        cfg.data.settings.num_test_sequences = omegaconf_select(cfg, "data.settings.num_test_sequences")
        cfg.data.settings.observation_noise = omegaconf_select(cfg, "data.settings.observation_noise")
        cfg.data.settings.prediction_noise = omegaconf_select(cfg, "data.settings.prediction_noise")
        cfg.data.settings.prediction_noise_beta = omegaconf_select(cfg, "data.settings.prediction_noise_beta", default=2.0) # generalized gaussian noise beta parameter
        cfg.data.settings.jump_rate = omegaconf_select(cfg, "data.settings.jump_rate", default=0.0)
        cfg.data.settings.seed = omegaconf_select(cfg, "data.settings.seed", default=1)

        return cfg

    def __getitem__(self, idx):
        """
        Get a synthetic video sample with associated labels
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            Tuple containing:
                - video (torch.Tensor): Video tensor with shape [T, H, W]
                - labels (tuple): Tuple containing various label tensors
        """
        
        # Map idx to the actual index in trajectories
        traj_idx = self.indices[idx]
        
        if not self.store_videos:
            trajectory = self.trajectories[traj_idx]
            noise_level = self.noise_levels[traj_idx]
            video = self._create_video(trajectory, noise_level)
        else:
            video = self.videos[traj_idx]
        x = video.reshape(self.seq_len, -1)

        label = self.labels[traj_idx]
        
        return x, label

    def __len__(self):
        """Return the number of sequences in the split"""
        return len(self.indices)

    def _generate_trajectories(self):
        """
        Generate synthetic trajectories based on the specified motion type
        
        Returns:
            trajectories (list): List of positions
            noise_levels (list): List of noise levels
            labels (dict): Dictionary of labels for each trajectory
        """
        
        if self.motion_type == "circular":
            return self._generate_circular_trajectories()
        elif self.motion_type == "dual_circular":
            return self._generate_dual_circular_trajectories()
        elif self.motion_type == "square":
            return self._generate_square_trajectories()
        elif self.motion_type == "random_square":
            return self._generate_random_square_trajectories()
        elif self.motion_type == "reflection":
            return self._generate_reflection_trajectories()
        elif self.motion_type == "figure8":
            return self._generate_figure8_trajectories()
        elif self.motion_type == "wobbly":
            return self._generate_wobbly_trajectories()
        elif self.motion_type == "dual_wobbly":
            return self._generate_dual_wobbly_trajectories()
        else:
            raise ValueError(f"Invalid motion type: {self.motion_type}. Must be one of {self._VALID_MOTION_TYPES}.")
        
    def _generate_figure8_trajectories(self):
        """
        Generate trajectories of figure-8 motion with noise
        
        Returns:
            trajectories (list): List of positions
            labels (dict): Dictionary of labels for each trajectory
        """
        
        if self.jump_rate > 0:
            print("Warning: jump_rate is not supported for figure-8 trajectories. Setting it to 0.")
            self.jump_rate = 0
        
        T = self.period_duration
        dt = self.dt
        n_steps = self.seq_len
        pred_noise = self.prediction_noise
        max_obs_noise = self.observation_noise
        
        labels = []

        trajectories = []
        noise_levels = []
        for _ in range(self.num_sequences):
            trajectory = torch.zeros(n_steps, 2)
            noise_level = torch.zeros(n_steps, 1)

            for t in range(0, n_steps):
                theta = t * dt * np.pi / T
                x = np.cos(theta)
                y = np.sin(theta) * np.cos(theta)
                trajectory[t] = torch.tensor([x, y])

                # observation noise
                obs_noise = self.observation_noise
                if self.noise_type == "gaussian":
                    obs_noise = max_obs_noise
                elif self.noise_type == "random_gaussian":
                    mean_noise = max_obs_noise
                    obs_noise = np.random.exponential(mean_noise)
                    obs_noise = max(0, obs_noise - 0.3 * mean_noise)
                elif self.noise_type == "state_dependent_gaussian":
                    obs_noise = max_obs_noise * torch.abs((trajectory[t,0] + 1) / 2)
                noise_level[t] = obs_noise

            label = {
                'pos': trajectory,
                'noise_level': noise_level,
            }

            trajectories.append(trajectory)
            noise_levels.append(noise_level)
            labels.append(label)
            
        return trajectories, noise_levels, labels

    def _generate_reflection_trajectories(self):
        """
        Generate trajectories of reflection motion with noise.
        pred_noise is used as the probability of reversal at reflection points.
        
        Returns:
            trajectories (list): List of positions
            labels (dict): Dictionary of labels for each trajectory
        """

        if self.jump_rate > 0:
            print("Warning: jump_rate is not supported for reflection trajectories. Setting it to 0.")
            self.jump_rate = 0
        
        dt = self.dt
        n_steps = self.seq_len
        pred_noise = self.prediction_noise
        max_obs_noise = self.observation_noise
        
        labels = []

        trajectories = []
        noise_levels = []
        for _ in range(self.num_sequences):
            trajectory = torch.zeros(n_steps, 2)
            trajectory[0] = torch.rand(2) * 2 - 1  # Random start position in [-1, 1]
            velocity = torch.zeros(n_steps, 2)
            velocity[0] = torch.rand(2) * 2 - 1  # Random initial velocity
            noise_level = torch.zeros(n_steps, 1)
            noise_level[0] = max_obs_noise

            for t in range(1, n_steps):
                projection = trajectory[t-1] + velocity[t-1] * dt
                # trajectory[t] = trajectory[t] + torch.randn(2) * np.sqrt(pred_noise)
                # Reflect trajectory at boundaries
                if projection[0] < -1 or projection[0] > 1:
                    reversal = np.random.rand() < pred_noise
                    r = -1 if reversal else 1
                    velocity[t] = torch.tensor([-velocity[t-1, 0], r * velocity[t-1, 1]])
                    trajectory[t] = trajectory[t-1] + velocity[t] * dt
                elif projection[1] < -1 or projection[1] > 1:
                    reversal = np.random.rand() < pred_noise
                    r = -1 if reversal else 1
                    velocity[t] = torch.tensor([r * velocity[t-1, 0], -velocity[t-1, 1]])
                    trajectory[t] = trajectory[t-1] + velocity[t] * dt
                else:
                    velocity[t] = velocity[t-1]
                    trajectory[t] = projection
                trajectory[t] = torch.clamp(trajectory[t], -1, 1)

                # observation noise
                obs_noise = self.observation_noise
                if self.noise_type == "gaussian":
                    obs_noise = max_obs_noise
                elif self.noise_type == "random_gaussian":
                    mean_noise = max_obs_noise
                    obs_noise = np.random.exponential(mean_noise)
                    obs_noise = max(0, obs_noise - 0.3 * mean_noise)
                elif self.noise_type == "state_dependent_gaussian":
                    obs_noise = max_obs_noise * torch.abs((trajectory[t,0] + 1) / 2)
                noise_level[t] = obs_noise

            label = {
                'pos': trajectory,
                'noise_level': noise_level,
                'velocity': velocity,
            }

            trajectories.append(trajectory)
            noise_levels.append(noise_level)
            labels.append(label)
            
        return trajectories, noise_levels, labels


    def _generate_square_trajectories(self):
        """
        Generate trajectories of square motion with noise
        
        Returns:
            trajectories (list): List of positions
            labels (dict): Dictionary of labels for each trajectory
        """

        if self.jump_rate > 0:
            print("Warning: jump_rate is not supported for square trajectories. Setting it to 0.")
            self.jump_rate = 0
        
        T = self.period_duration
        waiting_time = T / 4
        dt = self.dt
        n_steps = self.seq_len
        pred_noise = self.prediction_noise
        max_obs_noise = self.observation_noise
        
        labels = []

        trajectories = []
        noise_levels = []
        for _ in range(self.num_sequences):
            trajectory = torch.zeros(n_steps, 2)
            trajectory[0] = torch.tensor([1.0, 0.0])
            noise_level = torch.zeros(n_steps, 1)
            noise_level[0] = max_obs_noise
            sin_trajectory = torch.zeros(n_steps, 2)
            sin_trajectory[0] = torch.tensor([1.0, 0.0]) # pseudo position for label

            for t in range(1, n_steps):
                pos = (t * dt / waiting_time % 1) * 2 - 1
                if int(t * dt / waiting_time % 4) < 1:
                    trajectory[t] = torch.tensor([1.0, pos]) 
                elif int(t * dt / waiting_time % 4) < 2:
                    trajectory[t] = torch.tensor([-pos, 1.0])
                elif int(t * dt / waiting_time % 4) < 3:
                    trajectory[t] = torch.tensor([-1.0, -pos])
                elif int(t * dt / waiting_time % 4) < 4:
                    trajectory[t] = torch.tensor([pos, -1.0])

                sin_pos = torch.tensor([np.cos(np.pi * 2 * (t * dt / T)),
                                        np.sin(np.pi * 2 * (t * dt / T))])
                sin_trajectory[t] = sin_pos

                # observation noise
                obs_noise = self.observation_noise
                if self.noise_type == "gaussian":
                    obs_noise = max_obs_noise
                elif self.noise_type == "random_gaussian":
                    mean_noise = max_obs_noise
                    obs_noise = np.random.exponential(mean_noise)
                    obs_noise = max(0, obs_noise - 0.3 * mean_noise)
                elif self.noise_type == "state_dependent_gaussian":
                    obs_noise = max_obs_noise * torch.abs((trajectory[t,0] + 1) / 2)
                noise_level[t] = obs_noise

            label = {
                'pos': trajectory,
                'sin_pos': sin_trajectory,
                'noise_level': noise_level,
            }

            trajectories.append(trajectory)
            noise_levels.append(noise_level)
            labels.append(label)
            
        return trajectories, noise_levels, labels

    def _generate_random_square_trajectories(self):
        """
        Generate trajectories of square motion with noise
        
        Returns:
            trajectories (list): List of positions
            labels (dict): Dictionary of labels for each trajectory
        """

        if self.jump_rate > 0:
            print("Warning: jump_rate is not supported for square trajectories. Setting it to 0.")
            self.jump_rate = 0
        
        T = self.period_duration
        waiting_time = T 
        dt = self.dt
        n_steps = self.seq_len
        pred_noise = self.prediction_noise
        max_obs_noise = self.observation_noise
        
        labels = []

        trajectories = []
        noise_levels = []
        for _ in range(self.num_sequences):
            trajectory = torch.zeros(n_steps, 2)
            trajectory[0] = torch.tensor([1.0, 0.0])
            noise_level = torch.zeros(n_steps, 1)
            noise_level[0] = max_obs_noise

            for t in range(1, n_steps):
                trajectory[t] = trajectory[t-1]
                if int(t * dt % waiting_time) == 0:
                    trajectory[t] = trajectory[t, [1,0]] * np.random.choice([-1, 1])
                

                # observation noise
                obs_noise = self.observation_noise
                if self.noise_type == "gaussian":
                    obs_noise = max_obs_noise
                elif self.noise_type == "random_gaussian":
                    mean_noise = max_obs_noise
                    obs_noise = np.random.exponential(mean_noise)
                    obs_noise = max(0, obs_noise - 0.3 * mean_noise)
                elif self.noise_type == "state_dependent_gaussian":
                    obs_noise = max_obs_noise * torch.abs((trajectory[t,0] + 1) / 2)
                noise_level[t] = obs_noise

            label = {
                'pos': trajectory,
                'noise_level': noise_level,
            }

            trajectories.append(trajectory)
            noise_levels.append(noise_level)
            labels.append(label)
            
        return trajectories, noise_levels, labels

    def _generate_circular_trajectories(self):
        """
        Generate trajectories of circular motion with noise
        
        Returns:
            trajectories (list): List of positions
            labels (dict): Dictionary of labels for each trajectory
        """
        T = self.period_duration
        dt = self.dt
        n_steps = self.seq_len
        pred_noise = self.prediction_noise
        max_obs_noise = self.observation_noise

        w = 2 * np.pi / T
        A = torch.tensor([[np.cos(w * dt), -np.sin(w * dt)],
                        [np.sin(w * dt), np.cos(w * dt)]],
                        dtype=torch.float32)

        # Rotation matrix for jumps 
        w2 = np.pi * 135 / 180
        R = torch.tensor([[np.cos(w2), -np.sin(w2)],
                        [np.sin(w2), np.cos(w2)]],
                        dtype=torch.float32)
        
        labels = []

        trajectories = []
        noise_levels = []
        for _ in range(self.num_sequences):
            trajectory = torch.zeros(n_steps, 2)
            trajectory[0] = torch.tensor([1.0, 0.0])
            noise_level = torch.zeros(n_steps, 1)
            noise_level[0] = max_obs_noise

            jumps = np.random.rand(n_steps) < self.jump_rate
            for t in range(1, n_steps):
                trajectory[t] = A @ trajectory[t - 1] + torch.randn(2) * np.sqrt(pred_noise)
                # renormalize
                trajectory[t] = trajectory[t] / torch.norm(trajectory[t])

                # Randomly jump to a new position
                if jumps[t]:
                    trajectory[t] = R @ trajectory[t]

                # observation noise
                obs_noise = self.observation_noise
                if self.noise_type == "gaussian":
                    obs_noise = max_obs_noise
                elif self.noise_type == "random_gaussian":
                    mean_noise = max_obs_noise
                    obs_noise = np.random.exponential(mean_noise)
                    obs_noise = max(0, obs_noise - 0.3 * mean_noise)
                elif self.noise_type == "state_dependent_gaussian":
                    obs_noise = max_obs_noise * torch.abs((trajectory[t,0] + 1) / 2)
                noise_level[t] = obs_noise

            jump_classes = np.vstack([jumps, 1 - jumps]).T
            label = {
                'pos': trajectory,
                'noise_level': noise_level,
                "jumps": torch.tensor(jump_classes, dtype=torch.float32),
            }

            trajectories.append(trajectory)
            noise_levels.append(noise_level)
            labels.append(label)
            
        return trajectories, noise_levels, labels


    # def _generate_wobbly_trajectories(self):
    #     """
    #     Generate trajectories of wobbly circular motion with noise
        
    #     Returns:
    #         trajectories (list): List of positions
    #         labels (dict): Dictionary of labels for each trajectory
    #     """
    #     T = self.period_duration
    #     dt = self.dt
    #     n_steps = self.seq_len
    #     pred_noise = self.prediction_noise
    #     max_obs_noise = self.observation_noise
    #     k = 4  # frequency of wobble
    #     a = 0.5  # amplitude of wobble

    #     def rk4_step(func, state, dt):
    #         """
    #         func: a function that takes (state) and returns the derivative (d_state/dt)
    #         state: current [x, y] (numpy array)
    #         dt: time step
    #         """
    #         k1 = func(state)
    #         k2 = func(state + dt/2 * k1)
    #         k3 = func(state + dt/2 * k2)
    #         k4 = func(state + dt * k3)
            
    #         return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    #     def dz_func(z, theta0=-np.pi*(3/8)):
    #         r = torch.sqrt(z[0]**2 + z[1]**2)
    #         theta = torch.atan2(z[1], z[0])
    #         dz0 = a * k * torch.cos(theta) * torch.cos(k * (theta - theta0)) - r * torch.sin(theta)
    #         dz1 = a * k * torch.sin(theta) * torch.cos(k * (theta - theta0)) + r * torch.cos(theta)
    #         return torch.tensor([dz0, dz1])
    #     labels = []

    #     trajectories = []
    #     noise_levels = []
    #     print("Generating wobbly trajectories...")
    #     for _ in tqdm.tqdm(range(self.num_sequences)):
    #         trajectory = torch.zeros(n_steps, 2)
    #         trajectory[0] = torch.randn(2)
    #         # normalize initial position
    #         trajectory[0] = trajectory[0] / torch.norm(trajectory[0])

    #         noise_level = torch.zeros(n_steps, 1)
    #         noise_level[0] = max_obs_noise

    #         for t in range(1, n_steps):
    #             trajectory[t] = rk4_step(dz_func, trajectory[t - 1], dt) + torch.randn(2) * np.sqrt(pred_noise)

    #             # observation noise
    #             obs_noise = self.observation_noise
    #             if self.noise_type == "gaussian":
    #                 obs_noise = max_obs_noise
    #             elif self.noise_type == "random_gaussian":
    #                 mean_noise = max_obs_noise
    #                 obs_noise = np.random.exponential(mean_noise)
    #                 obs_noise = max(0, obs_noise - 0.3 * mean_noise)
    #             elif self.noise_type == "state_dependent_gaussian":
    #                 obs_noise = max_obs_noise * torch.abs((trajectory[t,0] + 1) / 2)
    #             noise_level[t] = obs_noise

    #         label = {
    #             'pos': trajectory,
    #             'noise_level': noise_level,
    #         }

    #         trajectories.append(trajectory)
    #         noise_levels.append(noise_level)
    #         labels.append(label)
            
    #     return trajectories, noise_levels, labels

    def _generate_wobbly_trajectories(self):
        """
        Generate trajectories of wobbly circular motion with noise
        Vectorized implementation for speed.
        
        Returns:
            trajectories (list): List of positions
            labels (dict): Dictionary of labels for each trajectory
        """
        T = self.period_duration
        dt = self.dt
        n_steps = self.seq_len
        pred_noise = self.prediction_noise
        pred_noise_beta = self.prediction_noise_beta
        if pred_noise_beta != 2.0:
            if pred_noise_beta > 0.0:
                def gen_normal(dim, mu=0.0, alpha=1.0, beta=2.0):
                    d = dim if isinstance(dim, int) else len(dim)
                    direction = torch.randn(dim)
                    direction = direction / direction.norm(2)
                    g = torch.distributions.Gamma(d / beta, 1.0).sample()
                    r = alpha * g**(1/beta)
                    return mu + r * direction
            else:
                def gen_normal(dim, mu=0.0, alpha=1.0, beta=2.0):
                    g = torch.distributions.Gamma(1/beta, 1).sample(dim)
                    return mu + alpha * g**(1/beta) * torch.sign(torch.rand(dim) - 0.5)

        max_obs_noise = self.observation_noise
        k = 4  # frequency of wobble
        a = 0.5  # amplitude of wobble
        num_sequences = self.num_sequences

        # Vectorized RK4 step
        def rk4_step_batch(func, state, dt):
            k1 = func(state)
            k2 = func(state + dt/2 * k1)
            k3 = func(state + dt/2 * k2)
            k4 = func(state + dt * k3)
            return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        # Vectorized dynamics function
        def dz_func_batch(z, theta0=-np.pi*(3/8)):
            # z shape: [Batch, 2]
            x, y = z[:, 0], z[:, 1]
            r = torch.sqrt(x**2 + y**2)
            theta = torch.atan2(y, x)
            
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            cos_k_diff = torch.cos(k * (theta - theta0))
            
            dz0 = a * k * cos_theta * cos_k_diff - r * sin_theta
            dz1 = a * k * sin_theta * cos_k_diff + r * cos_theta
            return torch.stack([dz0, dz1], dim=1)
        
        # Initialize tensors for all sequences at once: [N, T, 2]
        trajectories = torch.zeros(num_sequences, n_steps, 2)
        noise_levels = torch.zeros(num_sequences, n_steps, 1)
        
        # Initial positions
        init_pos = torch.randn(num_sequences, 2)
        trajectories[:, 0] = init_pos / torch.norm(init_pos, dim=1, keepdim=True)
        noise_levels[:, 0] = max_obs_noise

        for t in tqdm.tqdm(range(1, n_steps)):
            prev_state = trajectories[:, t-1]
            
            # Update state
            step_delta = rk4_step_batch(dz_func_batch, prev_state, dt) - prev_state
            # Reconstruct next state with prediction noise
            if pred_noise_beta == 2.0:
                noise = torch.randn(num_sequences, 2) * np.sqrt(pred_noise)
            else:
                noise = gen_normal((num_sequences, 2), alpha=np.sqrt(pred_noise), beta=pred_noise_beta)
            next_state = prev_state + step_delta + noise
            
            trajectories[:, t] = next_state

            # Calculate observation noise for the whole batch
            if self.noise_type == "gaussian":
                obs_noise = torch.full((num_sequences, 1), max_obs_noise)
            elif self.noise_type == "random_gaussian":
                mean_noise = max_obs_noise
                obs_noise = torch.zeros(num_sequences, 1).exponential_(mean_noise)
                obs_noise = torch.clamp(obs_noise - 0.3 * mean_noise, min=0)
            elif self.noise_type == "state_dependent_gaussian":
                obs_noise = max_obs_noise * torch.abs((next_state[:, 0:1] + 1) / 2)
            else:
                 obs_noise = torch.full((num_sequences, 1), max_obs_noise)
                 
            noise_levels[:, t] = obs_noise

        # Convert batched tensors back to lists to match the original API expected by __getitem__
        trajectories_list = [t for t in trajectories]
        noise_levels_list = [n for n in noise_levels]
        labels = []
        for i in range(num_sequences):
            labels.append({
                'pos': trajectories_list[i],
                'noise_level': noise_levels_list[i],
            })

        return trajectories_list, noise_levels_list, labels
            


    def _generate_dual_circular_trajectories(self):
        """
        Generate trajectories of circular motion with noise and two possible trajectories
        
        Returns:
            trajectories (list): List of positions
            labels (dict): Dictionary of labels for each trajectory
        """
        T = self.period_duration
        dt = self.dt
        n_steps = self.seq_len
        pred_noise = self.prediction_noise
        max_obs_noise = self.observation_noise

        w = 2 * np.pi / T
        A = torch.tensor([[np.cos(w * dt), -np.sin(w * dt)],
                        [np.sin(w * dt), np.cos(w * dt)]],
                        dtype=torch.float32)
        A_inv = torch.inverse(A)

        # Rotation matrix for jumps 
        w2 = np.pi * 135 / 180
        R = torch.tensor([[np.cos(w2), -np.sin(w2)],
                        [np.sin(w2), np.cos(w2)]],
                        dtype=torch.float32)
        
        labels = []

        trajectories = []
        noise_levels = []
        for _ in range(self.num_sequences):
            trajectory = torch.zeros(n_steps, 2)
            trajectory[0] = torch.tensor([1.0, 0.0])
            noise_level = torch.zeros(n_steps, 1)
            noise_level[0] = max_obs_noise

            jumps = np.random.rand(n_steps) < self.jump_rate
            current_trajectory = np.random.choice([0, 1])  # Randomly choose trajectory 0 or 1
            changed_flag = False
            for t in range(1, n_steps):
                M = A if current_trajectory == 0 else A_inv   
                trajectory[t] = M @ trajectory[t - 1] + torch.randn(2) * np.sqrt(pred_noise)
                # renormalize
                trajectory[t] = trajectory[t] / torch.norm(trajectory[t])

                # Randomly jump to a new position
                if jumps[t]:
                    trajectory[t] = R @ trajectory[t]

                # observation noise
                obs_noise = self.observation_noise
                if self.noise_type == "gaussian":
                    obs_noise = max_obs_noise
                elif self.noise_type == "random_gaussian":
                    mean_noise = max_obs_noise
                    obs_noise = np.random.exponential(mean_noise)
                    obs_noise = max(0, obs_noise - 0.3 * mean_noise)
                elif self.noise_type == "state_dependent_gaussian":
                    obs_noise = max_obs_noise * torch.abs((trajectory[t,0] + 1) / 2)
                noise_level[t] = obs_noise

                # Randomly switch trajectory when crossing the x-axis
                # current_trajectory = np.random.choice([0, 1])  
                if not changed_flag:
                    if trajectory[t, 1] < 0 and trajectory[t - 1, 1] >= 0 \
                    or trajectory[t, 1] >= 0 and trajectory[t - 1, 1] < 0:
                        current_trajectory = np.random.choice([0, 1])  
                        changed_flag = True
                else:
                    changed_flag = False

            jump_classes = np.vstack([jumps, 1 - jumps]).T
            label = {
                'pos': trajectory,
                'noise_level': noise_level,
                "jumps": torch.tensor(jump_classes, dtype=torch.float32),
            }

            trajectories.append(trajectory)
            noise_levels.append(noise_level)
            labels.append(label)
            
        return trajectories, noise_levels, labels
    
    
    
    def _generate_dual_wobbly_trajectories(self):
        """
        Generate trajectories of wobbly circular motion with noise and two possible trajectories
        
        Returns:
            trajectories (list): List of positions
            labels (dict): Dictionary of labels for each trajectory
        """
        
        T = self.period_duration
        dt = self.dt
        n_steps = self.seq_len
        pred_noise = self.prediction_noise
        max_obs_noise = self.observation_noise
        
        k1 = 4  # frequency of wobble
        k2 = 7
        a = 0.5  # amplitude of wobble
        p_switch = self.jump_rate  # probability of switching between the two trajectories
        num_sequences = self.num_sequences

        # Vectorized RK4 step
        def rk4_step_batch(func, state, dt):
            k1 = func(state)
            k2 = func(state + dt/2 * k1)
            k3 = func(state + dt/2 * k2)
            k4 = func(state + dt * k3)
            return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        def dz_func_batch(z, k, theta0=-np.pi*(3/8), r0=1.0, lam=1.0, a=3.0): # lam 0.2->0.5
            x, y = z[:, 0], z[:, 1]
            r = torch.sqrt(x**2 + y**2)
            theta = torch.atan2(y, x)
            
            # Radial: attract to r0 and petal curve
            dr = -lam * (r - r0) + a * torch.cos(k * (theta - theta0))
            # Angular: constant rotation
            dtheta = 1.0
            
            # Convert to Cartesian
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            dz0 = dr * cos_theta - r * dtheta * sin_theta
            dz1 = dr * sin_theta + r * dtheta * cos_theta
            return torch.stack([dz0, dz1], dim=1)

        # Initialize tensors for all sequences at once: [N, T, 2]
        positions = torch.zeros(2, num_sequences, 2)
        current_trajectories = torch.zeros(num_sequences, dtype=torch.int)
        trajectories = torch.zeros(num_sequences, n_steps, 2)
        noise_levels = torch.zeros(num_sequences, n_steps, 1)
        
        # Initial positions
        init_pos = torch.randn(num_sequences, 2)
        positions[0] = init_pos / torch.norm(init_pos, dim=1, keepdim=True)
        positions[1] = init_pos / torch.norm(init_pos, dim=1, keepdim=True) # start in opposite sides for the two trajectories
        trajectories[:, 0] = init_pos / torch.norm(init_pos, dim=1, keepdim=True)
        noise_levels[:, 0] = max_obs_noise

        for t in tqdm.tqdm(range(1, n_steps)):

            # update positions for both trajectories
            positions[0] = rk4_step_batch(lambda z: dz_func_batch(z, k1), positions[0], dt) #* (1 - current_trajectories).unsqueeze(1)
            positions[1] = rk4_step_batch(lambda z: dz_func_batch(z, k2), positions[1], dt) #* (current_trajectories).unsqueeze(1)
            
            positions[0] = positions[0] + torch.randn(num_sequences, 2) * np.sqrt(pred_noise) * torch.tensor([2.0,1.0]) 
            positions[1] = positions[1] + torch.randn(num_sequences, 2) * np.sqrt(pred_noise) * torch.tensor([2.0,1.0])

            # Select position for each sequence based on current_trajectories
            trajectories[:, t] = positions[0, torch.arange(num_sequences), :]

            # Trajectory 1 is left side, Trajectory 2 is right side
            trajectories[:, t, 0] = torch.where(current_trajectories == 0, trajectories[:, t, 0] / 1.5 + 1.5, trajectories[:, t, 0] / 1.5 - 1.5)

            # randomly switch trajectory with rate
            switch_mask = torch.rand(num_sequences) < p_switch
            current_trajectories[switch_mask] = 1 - current_trajectories[switch_mask]

            # Calculate observation noise for the whole batch
            if self.noise_type == "gaussian":
                obs_noise = torch.full((num_sequences, 1), max_obs_noise)
            elif self.noise_type == "random_gaussian":
                mean_noise = max_obs_noise
                obs_noise = torch.zeros(num_sequences, 1).exponential_(mean_noise)
                obs_noise = torch.clamp(obs_noise - 0.3 * mean_noise, min=0)
            elif self.noise_type == "state_dependent_gaussian":
                obs_noise = max_obs_noise * torch.abs((trajectories[:, t, 0:1] + 1) / 2)
            else:
                 obs_noise = torch.full((num_sequences, 1), max_obs_noise)
                 
            noise_levels[:, t] = obs_noise

        # Convert batched tensors back to lists to match the original API expected by __getitem__
        trajectories_list = [t for t in trajectories]
        noise_levels_list = [n for n in noise_levels]
        labels = []
        for i in range(num_sequences):
            labels.append({
                'pos': trajectories_list[i],
                'noise_level': noise_levels_list[i],
            })

        return trajectories_list, noise_levels_list, labels

    
    
    def _create_img_from_pos(self, pos, noise, img_dim=(16, 16), sigma=1.0):

        from skimage import io, transform
        img_dim_x, img_dim_y = img_dim
        x, y = pos

        # If half_flip post-processing is enabled, flip the position
        # if x > 0
        if self.data_cfg.settings.post_processing == "half_flip" and x > 0:
            y = -y

        if self.data_cfg.settings.post_processing == "random":
            pos_tensor = torch.tensor([x, y], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                img = self.random_network(pos_tensor).cpu()
            img = img.reshape(img_dim)
            return img + torch.randn(img.shape) * np.sqrt(noise)
            
        # Scale position to image coordinates
        x_img = ((x + 3) * img_dim_x / 6).clamp(0, img_dim_x-1)
        y_img = ((y + 3) * img_dim_y / 6).clamp(0, img_dim_y-1)

        # Create meshgrid for calculating Gaussian
        grid_x = self._grid_x
        grid_y = self._grid_y
        
        # Calculate Gaussian kernel
        gaussian = torch.exp(-((grid_x - x_img)**2 + (grid_y - y_img)**2) / (2 * sigma**2))
        
        if self.data_cfg.settings.post_processing == "distort":
            img = gaussian.numpy().astype(np.float32)
            img = transform.swirl(img, rotation=0, strength=10, radius=200)
            gaussian = torch.from_numpy(img)

        # # Normalize the Gaussian (optional)
        # gaussian = gaussian / gaussian.max()

        return gaussian + torch.randn(*img_dim) * np.sqrt(noise)

    def _create_video(self, trajectory, noise_level):
        """
        Create a video from the trajectory and noise level

        Args:
            trajectory (torch.Tensor): Trajectory of shape [T, 2]
            noise_level (torch.Tensor): Noise level of shape [T, 1]      
                
        Returns:
            torch.Tensor: Video as a tensor with shape [T, H, W]
        """
        
        # Fast path for "random" post-processing: batch all positions
        if self.data_cfg.settings.post_processing == "random":
            pos_tensor = trajectory.float().to(self.device)  # [T, 2]
            with torch.no_grad():
                imgs = self.random_network(pos_tensor).cpu()  # [T, H*W]
            imgs = imgs.reshape(self.seq_len, 1, *self.image_dim)  # [T, 1, H, W]
            # Add noise
            noise = noise_level.view(self.seq_len, 1, 1, 1).sqrt() * torch.randn_like(imgs)
            return imgs + noise

        observations = torch.zeros(self.seq_len, 1, *self.image_dim)
        for t in range(self.seq_len):
            obs_noise = noise_level[t]
        
            # Create image from position
            observations[t] = self._create_img_from_pos(trajectory[t], obs_noise, img_dim=self.image_dim)
        
        return observations

        
def get_video_frames(video, positions, pointer=None, colormap=None):
    from PIL import Image
    import matplotlib.pyplot as plt

    # Get the number of frames, channels, height, width
    num_frames, height_width = video.shape
    img_dim = int(np.sqrt(height_width))
    height = img_dim
    width = img_dim
    # Reshape video to [num_frames, C, H, W]
    video = video.reshape(num_frames, 1, height, width)  # [T, C, H, W]
    
    # Create lists to store the processed frames
    frames = []

    # Normalize the video frames to [0, 1] range
    video = (video - video.min()) / (video.max() - video.min())
    
    # Process each frame
    for t in range(num_frames):
        # Get the frame
        frame = video[t]  # [C, H, W]
        
        # Convert to 8-bit for display
        frame = (frame * 255).astype(np.uint8)
        
        # Reshape grayscale frame 
        frame = frame.reshape(height, width)  # [H, W]
        
        # Apply colormap if specified
        if colormap is not None:
            # Normalize to 0-1 for colormap
            frame_norm = frame.astype(float) / 255.0
            # Apply colormap (using updated API)
            colored = plt.colormaps.get_cmap(colormap)(frame_norm)
            # Convert to 0-255 uint8
            frame = (colored[:, :, :3] * 255).astype(np.uint8)
        
        # Create a figure and axis for plotting the trajectory with higher resolution
        # Increase resolution by using a higher DPI
        upscale_factor = 10  # Upscale by 10x 
        fig = plt.figure(figsize=(width/100, height/100), dpi=100 * upscale_factor)
        ax = fig.add_axes([0, 0, 1, 1])
        
        # Show the frame without interpolation, but allow it to be upscaled
        ax.imshow(frame, interpolation='none')
        
        # Add green pointer and path if pointer is enabled
        if pointer:
            # Draw path up to current frame with the specified color
            if t > 0:
                y_positions = [pos[0] for pos in positions[:t+1]]
                x_positions = [pos[1] for pos in positions[:t+1]]
                # convert to image coordinates
                x_positions = (np.array(x_positions) + 3) * width / 6
                y_positions = (np.array(y_positions) + 3) * height / 6
                # Use a smoother path representation with antialiasing and higher resolution
                ax.plot(x_positions, y_positions, color='#EE6677', linewidth=0.2, alpha=0.8,
                        solid_capstyle='round', solid_joinstyle='round', antialiased=True)
            
            # Draw current position pointer
            ypos, xpos = positions[t]
            # Convert to image coordinates
            xpos = (xpos + 3) * width / 6
            ypos = (ypos + 3) * height / 6
            pointer_radius = 0.1
            
            # # Draw white outline - scaled for higher resolution
            # circle_outline = plt.Circle((xpos, ypos), (pointer_radius + 1) /2, 
            #                             color='white', fill=False, linewidth=0.5/2)
            # ax.add_patch(circle_outline)
            
            # Draw colored circle - scaled for higher resolution
            circle = plt.Circle((xpos, ypos), pointer_radius/2, color='#EE6677')
            ax.add_patch(circle)
        
        # Remove axis and set limits
        ax.axis('off')
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Flipped to match image coordinates
        
        # Convert the figure to an image
        fig.canvas.draw()
        frame_with_plot = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        
        # Convert to PIL image
        pil_img = Image.fromarray(frame_with_plot)
        
        # Append to frames list for GIF
        frames.append(pil_img)

    return frames

def _save_as_pdf(pdf_frames, pdf_path, num_frames=8):
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    height, width = pdf_frames[0].size

    # Save as PDF with only the first num_frames frames horizontally with black borders
    with PdfPages(pdf_path) as pdf:
        # Determine how many frames to include (minimum of num_frames or actual frame count)
        frames_to_include = min(num_frames, len(pdf_frames))
        
        # Create a figure that will contain all frames in a row
        # Set exact width to 3 inches as requested
        total_width = 3  # inches
        
        # Width of each frame (accounting for spacing)
        frame_width = total_width / frames_to_include
        frame_height = frame_width * (height / width)  # maintain aspect ratio
        
        # Create figure for the frames with fixed width
        fig = plt.figure(figsize=(total_width, frame_height))
        
        # Create a grid layout with just one row and multiple columns
        # Remove spacing between frames
        gs = GridSpec(1, frames_to_include, figure=fig, wspace=0)
        
        # Add each frame to the grid with black borders
        for idx in range(frames_to_include):
            ax = fig.add_subplot(gs[0, idx])
            ax.imshow(np.array(pdf_frames[idx]))
            
            # # Add thinner black border around each frame
            # for spine in ax.spines.values():
            #     spine.set_edgecolor('black')
            #     spine.set_linewidth(0.1)  # Reduced from 1 to 0.1
            #     spine.set_visible(True)
            
            # Remove ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Adjust layout and save
        pdf.savefig(fig, bbox_inches='tight', dpi=300, transparent=True)
        plt.close(fig)

def save_videos_as_gifs_and_pdfs(videos, labels, output_dir="video_gifs", fps=10, max_videos=8, colormap=None, pointer=True):
    """
    Save a batch of videos as both GIF animations and PDF files.
    Images are denormalized, inverted, and a green pointer and path trace are added based on object position from labels.
    The PDF saves only the first 8 frames horizontally with black borders around each frame, 
    and velocity plots are saved as separate files.
    
    Args:
        videos (torch.Tensor): Batch of videos with shape [B, T, C, H, W] 
                              where B=batch, T=time, C=channels, H=height, W=width
        labels (tuple): Dict of label tensors from the dataset
        output_dir (str): Directory to save the GIFs/PDFs (will be created if it doesn't exist)
        fps (int): Frames per second for the GIF animation
        max_videos (int): Maximum number of videos to save
        colormap (str, optional): Matplotlib colormap name to apply
        pointer (bool): Whether to add position pointer and path trace
    
    Returns:
        list: Paths to the saved GIF and PDF files
    """
    import imageio
    import os
    import torch

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure videos is on CPU and convert to numpy
    if isinstance(videos, torch.Tensor):
        videos = videos.detach().cpu().numpy()
    
    # Limit the number of videos to save
    num_videos = min(videos.shape[0], max_videos)
    saved_paths = []
    positions = labels["pos"]
    # Process each video
    for i in range(num_videos):
        # Get the video frames
        video = videos[i]  # [T, C, H, W]
        pos = positions[i]  # [T, C, H, W]

        frames = get_video_frames(video, pos, pointer=pointer, colormap=colormap)
        pdf_frames = [frame.copy() for frame in frames]  # Copy frames for PDF

        # Create base filename
        base_filename = f"video_{i}"
        
        # Save as GIF
        gif_path = os.path.join(output_dir, f"{base_filename}.gif")
        imageio.mimsave(gif_path, frames, fps=fps)
        saved_paths.append(gif_path)
        
        # Save as PDF
        pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
        _save_as_pdf(pdf_frames, pdf_path, num_frames=8)
        
        saved_paths.append(pdf_path)
        
        print(f"Saved video {i} as GIF: {gif_path}")
        print(f"Saved video {i} as PDF: {pdf_path}")
    
    return saved_paths


def test_random_square():
# Set device
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Create the dataset
    data_dir = "datasets"

    cfg = {
        "data": {
            "data_dir": data_dir,
            "settings": {
                "motion_type": "random_square",  # or "square"
                "post_processing": None, #"half_flip",  # or None
                "image_dim": 10,
                "num_train_sequences": 1024,
                "num_test_sequences": 8,
                "seq_len": 100,
                "observation_noise": 0.01,
                "prediction_noise": 0.0,
                "noise_type": "gaussian",
                "period_duration": 0.1,
                "dt": 0.1
            }
        },
        "device": device
    }
  
    cfg = omegaconf.OmegaConf.create(cfg)

    # Create the dataset with fixed noise and grid patterns
    dataset = DotMotion(cfg)
    
    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        pin_memory=False,
    )
    
    # Get the first batch
    videos, labels = next(iter(dataloader))
    
    # Save videos as GIFs
    save_videos_as_gifs_and_pdfs(
        videos=videos,
        labels=labels,
        output_dir="random_square_motion_videos",
        fps=10,
        max_videos=3,
        colormap='gray',  # Try different colormaps like 'viridis', 'plasma', 'gray'
        pointer=True,  # Disable pointer and path trace
    )

    save_videos_as_gifs_and_pdfs(
        videos=videos,
        labels=labels,
        output_dir="random_square_motion_videos_no_pointer",
        fps=10,
        max_videos=3,
        colormap='gray',  # Try different colormaps like 'viridis', 'plasma', 'gray'
        pointer=False,  # Disable pointer and path trace
    )

def test_dual_circular_motion():
    mp.set_start_method('spawn', force=True)
    
    # Set device
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Create the dataset
    data_dir = "datasets"

    cfg = {
        "data": {
            "data_dir": data_dir,
            "settings": {
                "motion_type": "dual_circular",  # or "square"
                "post_processing": None, #"half_flip",  # or None
                "image_dim": 10,
                "num_train_sequences": 1024,
                "num_test_sequences": 8,
                "seq_len": 100,
                "observation_noise": 0.01,
                "prediction_noise": 0.0,
                "noise_type": "gaussian",
                "period_duration": 1.0,
                "jump_rate": 0.00,
                "dt": 0.1
            }
        },
        "device": device
    }
  
    cfg = omegaconf.OmegaConf.create(cfg)

    # Create the dataset with fixed noise and grid patterns
    dataset = DotMotion(cfg)
    
    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        pin_memory=False,
    )
    
    # Get the first batch
    videos, labels = next(iter(dataloader))
    
    # Save videos as GIFs
    save_videos_as_gifs_and_pdfs(
        videos=videos,
        labels=labels,
        output_dir="dual_circular_motion_videos",
        fps=10,
        max_videos=8,
        colormap='gray',  # Try different colormaps like 'viridis', 'plasma', 'gray'
        pointer=True,  # Disable pointer and path trace
    )

    save_videos_as_gifs_and_pdfs(
        videos=videos,
        labels=labels,
        output_dir="dual_circular_motion_videos_no_pointer",
        fps=10,
        max_videos=8,
        colormap='gray',  # Try different colormaps like 'viridis', 'plasma', 'gray'
        pointer=False,  # Disable pointer and path trace
    )

def test_square_motion():
    mp.set_start_method('spawn', force=True)
    
    # Set device
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Create the dataset
    data_dir = "datasets"

    cfg = {
        "data": {
            "data_dir": data_dir,
            "settings": {
                "motion_type": "square",
                "post_processing": None,
                "image_dim": 10,
                "num_train_sequences": 1024,
                "num_test_sequences": 8,
                "seq_len": 100,
                "observation_noise": 0.13,
                "prediction_noise": 0.0,
                "noise_type": "gaussian",
                "period_duration": 4.0,
                "jump_rate": 0.00,
                "dt": 0.2
            }
        },
        "device": device
    }
  
    cfg = omegaconf.OmegaConf.create(cfg)

    # Create the dataset with fixed noise and grid patterns
    dataset = DotMotion(cfg)
    
    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        pin_memory=False,
    )
    
    # Get the first batch
    videos, labels = next(iter(dataloader))
    
    # Save videos as GIFs
    save_videos_as_gifs_and_pdfs(
        videos=videos,
        labels=labels,
        output_dir="square_motion_videos",
        fps=10,
        max_videos=8,
        colormap='gray',  # Try different colormaps like 'viridis', 'plasma', 'gray'
        pointer=True,  # Disable pointer and path trace
    )

    save_videos_as_gifs_and_pdfs(
        videos=videos,
        labels=labels,
        output_dir="square_motion_videos_no_pointer",
        fps=10,
        max_videos=8,
        colormap='gray',  # Try different colormaps like 'viridis', 'plasma', 'gray'
        pointer=False,  # Disable pointer and path trace
    )


def test_circular_motion():
    mp.set_start_method('spawn', force=True)
    
    # Set device
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Create the dataset
    data_dir = "datasets"

    cfg = {
        "data": {
            "data_dir": data_dir,
            "settings": {
                "motion_type": "circular",  # or "square"
                "post_processing": "half_flip",  # or None
                "image_dim": 10,
                "num_train_sequences": 1024,
                "num_test_sequences": 8,
                "seq_len": 100,
                "observation_noise": 0.13,
                "prediction_noise": 0.0,
                "noise_type": "gaussian",
                "period_duration": 1.0,
                "jump_rate": 0.00,
                "dt": 0.1
            }
        },
        "device": device
    }
  
    cfg = omegaconf.OmegaConf.create(cfg)

    # Create the dataset with fixed noise and grid patterns
    dataset = DotMotion(cfg)
    
    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        pin_memory=False,
    )
    
    # Get the first batch
    videos, labels = next(iter(dataloader))
    
    # Save videos as GIFs
    save_videos_as_gifs_and_pdfs(
        videos=videos,
        labels=labels,
        output_dir="circular_motion_videos",
        fps=10,
        max_videos=8,
        colormap='gray',  # Try different colormaps like 'viridis', 'plasma', 'gray'
        pointer=True,  # Disable pointer and path trace
    )

    save_videos_as_gifs_and_pdfs(
        videos=videos,
        labels=labels,
        output_dir="circular_motion_videos_no_pointer",
        fps=10,
        max_videos=8,
        colormap='gray',  # Try different colormaps like 'viridis', 'plasma', 'gray'
        pointer=False,  # Disable pointer and path trace
    )

def test_figure8_motion():
    mp.set_start_method('spawn', force=True)
    
    # Set device
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Create the dataset
    data_dir = "datasets"

    cfg = {
        "data": {
            "data_dir": data_dir,
            "settings": {
                "motion_type": "figure8",  # or "square"
                "post_processing": None, #"half_flip",  # or None
                "image_dim": 10,
                "num_train_sequences": 1024,
                "num_test_sequences": 8,
                "seq_len": 100,
                "observation_noise": 0.01,
                "prediction_noise": 0.0,
                "noise_type": "gaussian",
                "period_duration": 1.0,
                "jump_rate": 0.00,
                "dt": 0.1
            }
        },
        "device": device
    }
  
    cfg = omegaconf.OmegaConf.create(cfg)

    # Create the dataset with fixed noise and grid patterns
    dataset = DotMotion(cfg)
    
    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        pin_memory=False,
    )
    
    # Get the first batch
    videos, labels = next(iter(dataloader))
    
    # Save videos as GIFs
    save_videos_as_gifs_and_pdfs(
        videos=videos,
        labels=labels,
        output_dir="figure8_motion_videos",
        fps=10,
        max_videos=8,
        colormap='gray',  # Try different colormaps like 'viridis', 'plasma', 'gray'
        pointer=True,  # Disable pointer and path trace
    )

    save_videos_as_gifs_and_pdfs(
        videos=videos,
        labels=labels,
        output_dir="figure8_motion_videos_no_pointer",
        fps=10,
        max_videos=8,
        colormap='gray',  # Try different colormaps like 'viridis', 'plasma', 'gray'
        pointer=False,  # Disable pointer and path trace
    )

def test_wobbly_motion():
    mp.set_start_method('spawn', force=True)
    
    # Set device
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Create the dataset
    data_dir = "datasets"

    cfg = {
        "data": {
            "data_dir": data_dir,
            "settings": {
                "motion_type": "wobbly",
                "post_processing": "distort",
                "image_dim": 10,
                "num_train_sequences": 20,
                "num_test_sequences": 8,
                "seq_len": 100,
                "observation_noise": 0.0,
                "prediction_noise": 0.1,
                "prediction_noise_beta": 4.0,
                "noise_type": "gaussian",
                "period_duration": 1.0,
                "jump_rate": 0.00,
                "dt": 0.3
            }
        },
        "device": device
    }
  
    cfg = omegaconf.OmegaConf.create(cfg)

    # Create the dataset with fixed noise and grid patterns
    dataset = DotMotion(cfg)
    
    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        pin_memory=False,
    )
    
    # Get the first batch
    videos, labels = next(iter(dataloader))
    
    # Save videos as GIFs
    save_videos_as_gifs_and_pdfs(
        videos=videos,
        labels=labels,
        output_dir="wobbly_motion_videos",
        fps=10,
        max_videos=8,
        colormap='gray',  # Try different colormaps like 'viridis', 'plasma', 'gray'
        pointer=True,  # Disable pointer and path trace
    )

    save_videos_as_gifs_and_pdfs(
        videos=videos,
        labels=labels,
        output_dir="wobbly_motion_videos_no_pointer",
        fps=10,
        max_videos=8,
        colormap='gray',  # Try different colormaps like 'viridis', 'plasma', 'gray'
        pointer=False,  # Disable pointer and path trace
    )

def test_reflection_motion():
    mp.set_start_method('spawn', force=True)
    
    # Set device
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Create the dataset
    data_dir = "datasets"

    cfg = {
        "data": {
            "data_dir": data_dir,
            "settings": {
                "motion_type": "reflection", #dual_circular",  # or "square"
                "post_processing": None, #"half_flip",  # or None
                "image_dim": 10,
                "num_train_sequences": 1024,
                "num_test_sequences": 8,
                "seq_len": 100,
                "observation_noise": 0.01,
                "prediction_noise": 0.5,
                "noise_type": "gaussian",
                "period_duration": 1.0,
                "jump_rate": 0.00,
                "dt": 0.4
            }
        },
        "device": device
    }
  
    cfg = omegaconf.OmegaConf.create(cfg)

    # Create the dataset with fixed noise and grid patterns
    dataset = DotMotion(cfg)
    
    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        pin_memory=False,
    )
    
    # Get the first batch
    videos, labels = next(iter(dataloader))
    
    # Save videos as GIFs
    save_videos_as_gifs_and_pdfs(
        videos=videos,
        labels=labels,
        output_dir="reflection_motion_videos",
        fps=10,
        max_videos=8,
        colormap='gray',  # Try different colormaps like 'viridis', 'plasma', 'gray'
        pointer=True,  # Disable pointer and path trace
    )

    save_videos_as_gifs_and_pdfs(
        videos=videos,
        labels=labels,
        output_dir="reflection_motion_videos_no_pointer",
        fps=10,
        max_videos=8,
        colormap='gray',  # Try different colormaps like 'viridis', 'plasma', 'gray'
        pointer=False,  # Disable pointer and path trace
    )

# Example usage
if __name__ == "__main__":
    # test_random_square()
    # test_circular_motion()
    # test_square_motion()
    # test_dual_circular_motion()
    # test_reflection_motion()
    # test_figure8_motion()
    test_wobbly_motion()
