import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .common import *

from stable_baselines3.common.running_mean_std import RunningMeanStd


class RunningMeanStdWithMomentum:
    """Running mean and std with momentum for exponential moving average."""
    def __init__(self, epsilon=1e-4, momentum=0.99):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon
        self.momentum = momentum
    
    def update(self, x):
        """Update running statistics with momentum."""
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x) if isinstance(x, np.ndarray) else 1
        
        if self.count == 1e-4:  # First update
            self.mean = batch_mean
            self.var = batch_var
            self.count = batch_count
        else:
            # Exponential moving average
            self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
            self.var = self.momentum * self.var + (1 - self.momentum) * batch_var
            self.count += batch_count


class NGU(nn.Module):
    """
    Never Give Up (NGU) for exploration.
    
    NGU combines episodic and lifelong (RND-based) intrinsic rewards:
    - Episodic: Uses k-NN in embedding space to measure novelty within episode
    - Lifelong: Uses RND to measure novelty across all experience
    
    Reference: Badia et al. "Never Give Up: Learning Directed Exploration Strategies" (2020)
    """
    def __init__(self, obs_size, act_size, state_size, hidden_size, 
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        use_bn=False,
                        use_ln=False,
                        ngu_knn_k=10,
                        ngu_dst_momentum=0.997,
                        ngu_use_rnd=True,
                        rnd_err_norm=False,
                        rnd_err_momentum=-1):
        super(NGU, self).__init__()

        self.use_bn = use_bn
        self.use_ln = use_ln

        self.obs_size = obs_size
        self.act_size = act_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        
        # NGU-specific parameters
        self.ngu_knn_k = ngu_knn_k
        self.ngu_use_rnd = ngu_use_rnd
        self.ngu_dst_momentum = ngu_dst_momentum
        self.ngu_moving_avg_dists = RunningMeanStdWithMomentum(momentum=self.ngu_dst_momentum)
        self.rnd_err_norm = rnd_err_norm
        self.rnd_err_momentum = rnd_err_momentum
        if rnd_err_momentum > 0:
            self.rnd_err_running_stats = RunningMeanStdWithMomentum(momentum=self.rnd_err_momentum)
        else:
            self.rnd_err_running_stats = RunningMeanStd()
        
        print("Initializing NGU (Never Give Up) module")
        print(f"  K-NN k={self.ngu_knn_k}, use_rnd={self.ngu_use_rnd}")
        
        # Determine if we're dealing with visual observations or vector observations
        if len(obs_size) == 1:
            # Vector observations
            obs_size_flat = obs_size[0]
            
            # Embedding network (for episodic memory)
            if use_bn:
                self.embedding_network = nn.Sequential(
                    nn.Linear(obs_size_flat, hidden_size, bias=False),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size, bias=False),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, state_size, bias=False),
                    nn.BatchNorm1d(state_size)
                )
            else:
                self.embedding_network = nn.Sequential(
                    nn.Linear(obs_size_flat, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, state_size)
                )
            
            if use_ln:
                self.embedding_network = nn.Sequential(
                    self.embedding_network,
                    nn.LayerNorm(state_size)
                )
            
            # Inverse dynamics model (predicts action from state embeddings)
            if use_bn:
                self.inverse_model = nn.Sequential(
                    nn.Linear(state_size * 2, hidden_size, bias=False),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size, bias=False),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, act_size, bias=False)
                )
            else:
                self.inverse_model = nn.Sequential(
                    nn.Linear(state_size * 2, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, act_size)
                )
                
        elif len(obs_size) == 3:
            # Visual observations (e.g., Atari)
            
            # Embedding network (CNN for episodic memory)
            feature_output = 7 * 7 * 64
            if use_bn:
                self.embedding_network = nn.Sequential(
                    nn.Conv2d(
                        in_channels=4,
                        out_channels=32,
                        kernel_size=8,
                        stride=4,
                        bias=False),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(),
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=4,
                        stride=2,
                        bias=False),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        bias=False),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(),
                    nn.Flatten(),
                    nn.Linear(feature_output, hidden_size, bias=False),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, state_size, bias=False),
                    nn.BatchNorm1d(state_size)
                )
            else:
                self.embedding_network = nn.Sequential(
                    nn.Conv2d(
                        in_channels=4,
                        out_channels=32,
                        kernel_size=8,
                        stride=4),
                    nn.LeakyReLU(),
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=4,
                        stride=2),
                    nn.LeakyReLU(),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=1),
                    nn.LeakyReLU(),
                    nn.Flatten(),
                    nn.Linear(feature_output, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, state_size)
                )
            
            if use_ln:
                self.embedding_network = nn.Sequential(
                    self.embedding_network,
                    nn.LayerNorm(state_size)
                )
            
            # Inverse dynamics model
            if use_bn:
                self.inverse_model = nn.Sequential(
                    nn.Linear(state_size * 2, hidden_size, bias=False),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size, bias=False),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, act_size, bias=False)
                )
            else:
                self.inverse_model = nn.Sequential(
                    nn.Linear(state_size * 2, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, act_size)
                )
        
        # RND for lifelong novelty (optional)
        if self.ngu_use_rnd:
            # Target network (fixed random network)
            if len(obs_size) == 1:
                obs_size_flat = obs_size[0]
                self.rnd_target = get_random_mlp(obs_size_flat, state_size, use_bn=use_bn, use_ln=use_ln)
            else:
                self.rnd_target = get_random_cnn(state_size, use_bn=use_bn, use_ln=use_ln)
            
            # Predictor network (trainable)
            if len(obs_size) == 1:
                if use_bn:
                    self.rnd_predictor = nn.Sequential(
                        nn.Linear(obs_size_flat, hidden_size, bias=False),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size, bias=False),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, state_size, bias=False),
                        nn.BatchNorm1d(state_size)
                    )
                else:
                    self.rnd_predictor = nn.Sequential(
                        nn.Linear(obs_size_flat, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, state_size)
                    )
                
                if use_ln:
                    self.rnd_predictor = nn.Sequential(
                        self.rnd_predictor,
                        nn.LayerNorm(state_size)
                    )
            else:
                feature_output = 7 * 7 * 64
                if use_bn:
                    self.rnd_predictor = nn.Sequential(
                        nn.Conv2d(
                            in_channels=4,
                            out_channels=32,
                            kernel_size=8,
                            stride=4,
                            bias=False),
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(),
                        nn.Conv2d(
                            in_channels=32,
                            out_channels=64,
                            kernel_size=4,
                            stride=2,
                            bias=False),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(),
                        nn.Conv2d(
                            in_channels=64,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            bias=False),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(),
                        nn.Flatten(),
                        nn.Linear(feature_output, hidden_size, bias=False),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, state_size, bias=False),
                        nn.BatchNorm1d(state_size)
                    )
                else:
                    self.rnd_predictor = nn.Sequential(
                        nn.Conv2d(
                            in_channels=4,
                            out_channels=32,
                            kernel_size=8,
                            stride=4),
                        nn.LeakyReLU(),
                        nn.Conv2d(
                            in_channels=32,
                            out_channels=64,
                            kernel_size=4,
                            stride=2),
                        nn.LeakyReLU(),
                        nn.Conv2d(
                            in_channels=64,
                            out_channels=64,
                            kernel_size=3,
                            stride=1),
                        nn.LeakyReLU(),
                        nn.Flatten(),
                        nn.Linear(feature_output, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, state_size)
                    )
                
                if use_ln:
                    self.rnd_predictor = nn.Sequential(
                        self.rnd_predictor,
                        nn.LayerNorm(state_size)
                    )
            
            # RND feature normalization
            self.rnd_feat_mean = 0.
            self.rnd_feat_std = 1.
            
            if not self.use_bn and not self.use_ln:
                print("  RND: Will collect features mean and std at the beginning ...")
        
        # Episodic memory: store embeddings for each parallel environment
        self.episodic_memory = None
        self.num_processes = None
        
        self.device = device
        self.to(device)
        
        # Track statistics
        self.last_inv_loss = 0.
        self.last_rnd_loss = 0.
        self.last_episodic_reward = 0.
        self.last_lifelong_reward = 0.
        
    def init_episodic_memory(self, num_processes):
        """Initialize episodic memory for parallel environments."""
        self.num_processes = num_processes
        self.episodic_memory = [None for _ in range(num_processes)]
        print(f"NGU: Initialized episodic memory for {num_processes} processes")
    
    def reset_episodic_memory(self, env_indices):
        """Reset episodic memory for specific environments (called on episode done)."""
        if self.episodic_memory is not None:
            for idx in env_indices:
                if idx < len(self.episodic_memory):
                    self.episodic_memory[idx] = None
    
    def get_feature_moments(self, obs):
        """Compute mean and std of RND target network outputs for normalization."""
        if not self.ngu_use_rnd:
            return
            
        states = []
        if len(obs) > 512:
            for i in range(0, len(obs), 512):
                states.append(self.rnd_target(obs[i:i+512]))
            states = torch.cat(states)
        else:
            states = self.rnd_target(obs)
        
        self.rnd_feat_mean = torch.mean(states, dim=0) 
        self.rnd_feat_std = torch.std(states, dim=0) + 1e-8

    def normalize_rnd_features(self, features):
        """Normalize RND target network outputs."""
        return (features - self.rnd_feat_mean) / self.rnd_feat_std

    def forward(self, obs, act, next_obs):
        """
        Forward pass for NGU.
        
        Args:
            obs: Current observation
            act: Action taken
            next_obs: Next observation
        
        Returns:
            curr_emb: Embedding of current observation
            next_emb: Embedding of next observation
            action_logits: Predicted action logits from inverse model
            rnd_target: RND target features (if using RND)
            rnd_pred: RND predicted features (if using RND)
        """
        # Get embeddings for episodic memory
        curr_emb = self.embedding_network(obs)
        next_emb = self.embedding_network(next_obs)
        
        # Inverse dynamics model
        combined_emb = torch.cat([curr_emb, next_emb], dim=1)
        action_logits = self.inverse_model(combined_emb)
        
        # RND for lifelong novelty
        rnd_target, rnd_pred = None, None
        if self.ngu_use_rnd:
            with torch.no_grad():
                rnd_target = self.rnd_target(next_obs)
                if not self.use_bn and not self.use_ln:
                    rnd_target = self.normalize_rnd_features(rnd_target)
            
            rnd_pred = self.rnd_predictor(next_obs)
        
        return curr_emb, next_emb, action_logits, rnd_target, rnd_pred

    def loss(self, obs, act, next_obs, *args, **kwargs):
        """
        Compute the training loss for NGU networks.
        
        Args:
            obs: Current observation
            act: Action taken (ground truth)
            next_obs: Next observation
        
        Returns:
            loss: Combined loss (inverse model + RND if enabled)
        """
        curr_emb, next_emb, action_logits, rnd_target, rnd_pred = self.forward(obs, act, next_obs)
        
        # Inverse dynamics loss (cross-entropy for discrete actions)
        inv_loss = F.cross_entropy(action_logits, act, reduction='mean')
        
        total_loss = inv_loss
        self.last_inv_loss = inv_loss.detach().cpu().item()
        
        # RND loss
        if self.ngu_use_rnd and rnd_target is not None:
            rnd_loss = F.mse_loss(rnd_pred, rnd_target, reduction='mean')
            total_loss = total_loss + rnd_loss
            self.last_rnd_loss = rnd_loss.detach().cpu().item()
        
        return total_loss

    def calc_euclidean_dists(self, embeddings, query):
        """
        Calculate Euclidean distances between query and all embeddings.
        
        Args:
            embeddings: Tensor of shape [N, D] (historical embeddings) on CPU or GPU
            query: Tensor of shape [D] or [1, D] (current embedding) on CPU or GPU
        
        Returns:
            dists: Tensor of shape [N] (distances) on CPU
        """
        # Move to CPU if needed (for memory efficiency, episodic memory is stored on CPU)
        if embeddings.device.type != 'cpu':
            embeddings = embeddings.cpu()
        if query.device.type != 'cpu':
            query = query.cpu()
            
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        # Compute Euclidean distance: sqrt(sum((x - y)^2))
        dists = torch.sqrt(((embeddings - query) ** 2).sum(dim=1))
        return dists

    def compute_episodic_reward(self, env_id, curr_emb, next_emb):
        """
        Compute episodic intrinsic reward for a single environment.
        
        Implements Algorithm 1 from NGU paper.
        
        Args:
            env_id: Environment index
            curr_emb: Current state embedding (on GPU)
            next_emb: Next state embedding (on GPU)
        
        Returns:
            episodic_reward: Scalar episodic reward
        
        Note:
            Episodic memory is stored on CPU to save GPU memory. With 128 parallel 
            environments and long episodes (3000+ steps), storing embeddings on GPU 
            can consume 1-2 GB of VRAM. Since k-NN computation already happens on CPU, 
            there's minimal performance impact from storing on CPU.
        """
        # Move embeddings to CPU for storage (saves GPU memory)
        curr_emb_cpu = curr_emb.detach().cpu()
        next_emb_cpu = next_emb.detach().cpu()
        
        # Update episodic memory (stored on CPU)
        if self.episodic_memory[env_id] is None:
            self.episodic_memory[env_id] = curr_emb_cpu.unsqueeze(0)
        else:
            self.episodic_memory[env_id] = torch.cat([self.episodic_memory[env_id], curr_emb_cpu.unsqueeze(0)], dim=0)
        
        # Add next_emb to memory
        self.episodic_memory[env_id] = torch.cat([self.episodic_memory[env_id], next_emb_cpu.unsqueeze(0)], dim=0)
        
        # Compute episodic reward based on k-NN
        episodic_reward = 0.0
        memory_size = self.episodic_memory[env_id].shape[0]
        
        if memory_size > 1:
            # Get all embeddings except the last one (which is next_emb)
            historical_embs = self.episodic_memory[env_id][:-1]
            
            # Compute k-nearest neighbors
            dists = self.calc_euclidean_dists(historical_embs, next_emb)
            dists_squared = dists ** 2
            dists_np = dists_squared.detach().cpu().numpy()
            
            # Get k nearest neighbors
            k = min(self.ngu_knn_k, len(dists_np))
            knn_dists = np.sort(dists_np)[:k]
            
            # Update moving average of distances
            self.ngu_moving_avg_dists.update(knn_dists)
            moving_avg_dist = self.ngu_moving_avg_dists.mean
            
            # Normalize distances
            normalized_dists = knn_dists / (moving_avg_dist + 1e-5)
            
            # Apply clustering (set small distances to 0)
            normalized_dists = np.maximum(normalized_dists - 0.008, np.zeros_like(knn_dists))
            
            # Compute kernel values
            kernel_values = 0.0001 / (normalized_dists + 0.0001)
            
            # Compute similarity
            similarity = np.sqrt(kernel_values.sum()) + 0.001
            
            # Compute episodic reward
            if similarity <= 8:
                episodic_reward = 1.0 / similarity
        
        return episodic_reward

    def curiosity(self, obs, act, next_obs, masks=None, *args, **kwargs):
        """
        Compute intrinsic reward (curiosity) based on NGU.
        
        Combines episodic and lifelong rewards:
        - Episodic: k-NN based novelty within episode
        - Lifelong: RND-based novelty across all experience
        
        Args:
            obs: Current observation [batch_size, ...] 
                 batch_size = num_steps × num_processes (flattened rollout)
            act: Action taken [batch_size]
            next_obs: Next observation [batch_size, ...]
            masks: Episode masks [batch_size] (0 if done, 1 otherwise)
        
        Returns:
            curiosity: Combined intrinsic rewards [batch_size]
        
        Note:
            The batch is flattened from [num_steps, num_processes] to [batch_size].
            We need to map each batch index back to its environment ID.
        """
        batch_size = obs.shape[0]
        
        # Initialize episodic memory if needed
        # CRITICAL: We need num_processes, not batch_size!
        # batch_size = num_steps × num_processes, but we only need one memory per process
        if self.episodic_memory is None:
            # Try to infer num_processes from batch_size
            # This is a fallback if not explicitly initialized
            # Typically num_processes is batch_size // num_steps
            # For safety, we'll detect this in first call
            if self.num_processes is None:
                raise ValueError(
                    "NGU episodic memory not initialized! "
                    "Please call model.init_episodic_memory(num_processes) "
                    "before training starts."
                )
        
        with torch.no_grad():
            curr_emb, next_emb, _, rnd_target, rnd_pred = self.forward(obs, act, next_obs)
            
            # Compute lifelong reward (RND)
            lifelong_rewards = None
            if self.ngu_use_rnd and rnd_target is not None:
                rnd_error = F.mse_loss(rnd_pred, rnd_target, reduction='none').mean(dim=1)
                rnd_error_np = rnd_error.cpu().numpy()
                
                if self.rnd_err_norm:
                    # Normalize RND error
                    if isinstance(self.rnd_err_running_stats, RunningMeanStdWithMomentum):
                        self.rnd_err_running_stats.update(rnd_error_np)
                        rnd_error_np = (rnd_error_np - self.rnd_err_running_stats.mean) / (np.sqrt(self.rnd_err_running_stats.var) + 1e-8)
                    else:
                        # Standard RunningMeanStd from stable_baselines3
                        self.rnd_err_running_stats.update(rnd_error_np)
                        rnd_error_np = (rnd_error_np - self.rnd_err_running_stats.mean) / (np.sqrt(self.rnd_err_running_stats.var) + 1e-8)
                
                # Lifelong reward: RND error + 1 (as in NGU paper)
                lifelong_rewards = rnd_error_np + 1.0
                self.last_lifelong_reward = np.mean(lifelong_rewards)
            
            # Compute episodic rewards for each observation in batch
            # Map batch index to environment ID: env_id = batch_idx % num_processes
            episodic_rewards = np.zeros(batch_size, dtype=np.float32)
            for batch_idx in range(batch_size):
                # Map flattened index back to environment ID
                env_id = batch_idx % self.num_processes
                
                episodic_reward = self.compute_episodic_reward(
                    env_id, 
                    curr_emb[batch_idx], 
                    next_emb[batch_idx]
                )
                episodic_rewards[batch_idx] = episodic_reward
            
            self.last_episodic_reward = np.mean(episodic_rewards)
            
            # Combine episodic and lifelong rewards
            if self.ngu_use_rnd and lifelong_rewards is not None:
                # L is the maximum reward scaling (default: 5.0 in paper)
                L = 5.0
                lifelong_rewards_clipped = np.clip(lifelong_rewards, 1.0, L)
                intrinsic_rewards = episodic_rewards * lifelong_rewards_clipped
            else:
                intrinsic_rewards = episodic_rewards
            
            # Reset episodic memory for environments that are done
            if masks is not None:
                done_indices = set()  # Use set to avoid duplicates
                for batch_idx in range(batch_size):
                    if masks[batch_idx].item() < 0.5:  # Episode done
                        # Map batch index to environment ID
                        env_id = batch_idx % self.num_processes
                        done_indices.add(env_id)
                
                if len(done_indices) > 0:
                    self.reset_episodic_memory(list(done_indices))
        
        return torch.from_numpy(intrinsic_rewards).float().to(self.device)

