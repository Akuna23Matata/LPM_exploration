import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .common import *

from stable_baselines3.common.running_mean_std import RunningMeanStd


class RND(nn.Module):
    """
    Random Network Distillation (RND) for exploration.
    
    RND uses the prediction error of a randomly initialized fixed target network
    as an intrinsic reward signal. The predictor network is trained to match
    the output of the fixed target network.
    """
    def __init__(self, obs_size, act_size, state_size, hidden_size, 
                        device='cuda' if torch.cuda.is_available() else 'cpu', 
                        use_bn=False,
                        use_ln=False):
        super(RND, self).__init__()

        self.use_bn = use_bn
        self.use_ln = use_ln

        self.obs_size = obs_size
        self.act_size = act_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        
        print("Initializing RND (Random Network Distillation) module")
        
        # Determine if we're dealing with visual observations or vector observations
        if len(obs_size) == 1:
            # Vector observations (e.g., MountainCar, CartPole)
            obs_size_flat = obs_size[0]
            
            # Target network (fixed random network)
            self.target_network = get_random_mlp(obs_size_flat, state_size, use_bn=use_bn, use_ln=use_ln)
            
            # Predictor network (trainable)
            if use_bn:
                self.predictor_network = nn.Sequential(
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
                self.predictor_network = nn.Sequential(
                    nn.Linear(obs_size_flat, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, state_size)
                )
            
            if use_ln:
                self.predictor_network = nn.Sequential(
                    self.predictor_network,
                    nn.LayerNorm(state_size)
                )
                
        elif len(obs_size) == 3:
            # Visual observations (e.g., Atari)
            
            # Target network (fixed random CNN)
            self.target_network = get_random_cnn(state_size, use_bn=use_bn, use_ln=use_ln)
            
            # Predictor network (trainable CNN)
            feature_output = 7 * 7 * 64
            if use_bn:
                self.predictor_network = nn.Sequential(
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
                self.predictor_network = nn.Sequential(
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
                self.predictor_network = nn.Sequential(
                    self.predictor_network,
                    nn.LayerNorm(state_size)
                )
        
        # Feature normalization (for target network outputs)
        self.feat_mean = 0.
        self.feat_std = 1.
        
        if not self.use_bn and not self.use_ln:
            print("Collecting features mean and std at the beginning ...")
        
        self.device = device
        self.to(device)
        
        # Track statistics
        self.last_pred_error = 0.
        
    def get_feature_moments(self, obs):
        """Compute mean and std of target network outputs for normalization."""
        states = []
        if len(obs) > 512:
            for i in range(0, len(obs), 512):
                states.append(self.target_network(obs[i:i+512]))
            states = torch.cat(states)
        else:
            states = self.target_network(obs)
        
        self.feat_mean = torch.mean(states, dim=0) 
        self.feat_std = torch.std(states, dim=0) + 1e-8  # Add small epsilon to avoid division by zero

    def normalize(self, state):
        """Normalize target network outputs."""
        state = (state - self.feat_mean) / self.feat_std
        return state

    def forward(self, obs, act=None, next_obs=None):
        """
        Forward pass for RND.
        
        Args:
            obs: Current observation (used for next_obs in RND, acts and obs are not used)
            act: Action (not used in RND, kept for API compatibility)
            next_obs: Next observation (this is what RND actually predicts on)
        
        Returns:
            target_features: Output of fixed target network
            predicted_features: Output of trainable predictor network
        """
        # RND only uses next_obs (the resulting state)
        if next_obs is None:
            next_obs = obs
            
        with torch.no_grad():
            target_features = self.target_network(next_obs)
            if not self.use_bn and not self.use_ln:
                target_features = self.normalize(target_features)
        
        predicted_features = self.predictor_network(next_obs)
        
        return target_features, predicted_features

    def loss(self, obs, act, next_obs, *args, **kwargs):
        """
        Compute the prediction loss for training the predictor network.
        
        Args:
            obs: Current observation
            act: Action taken
            next_obs: Next observation (what we actually predict on)
        
        Returns:
            loss: MSE between predicted and target features
        """
        target_features, predicted_features = self.forward(obs, act, next_obs)
        
        # MSE loss between predictor and target
        loss = F.mse_loss(predicted_features, target_features, reduction='none').mean(dim=1)
        
        # Track statistics
        self.last_pred_error = loss.detach().mean().cpu().item()
        
        return loss

    def curiosity(self, obs, act, next_obs, *args, **kwargs):
        """
        Compute intrinsic reward (curiosity) based on prediction error.
        
        Args:
            obs: Current observation
            act: Action taken
            next_obs: Next observation
        
        Returns:
            curiosity: Prediction error as intrinsic reward
        """
        target_features, predicted_features = self.forward(obs, act, next_obs)
        
        # Prediction error as intrinsic reward
        prediction_error = (predicted_features - target_features).pow(2).mean(dim=1)
        
        return prediction_error

