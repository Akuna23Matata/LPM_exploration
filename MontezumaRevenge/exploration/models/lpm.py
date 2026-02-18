import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .rnd import RND
from collections import deque

class LPM(nn.Module):
    """
    Learning Progress Monitor (LPM) for exploration.
    
    LPM builds on RND by adding an error predictor network that learns to predict
    the RND prediction errors. The final curiosity is computed as:
        curiosity = prediction_error - alpha * predicted_error
    
    This represents the learning progress - if we can predict the error well,
    it means we've seen similar states before. The difference tells us about
    genuine novelty and learning progress.
    """
    def __init__(self, obs_size, act_size, state_size, hidden_size, 
                        device='cuda' if torch.cuda.is_available() else 'cpu', 
                        use_bn=False,
                        use_ln=False,
                        buffer_size=10000,
                        lpm_alpha=0.1):
        super(LPM, self).__init__()

        self.use_bn = use_bn
        self.use_ln = use_ln
        self.obs_size = obs_size
        self.act_size = act_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.device = device
        self.lpm_alpha = lpm_alpha
        
        print("Initializing LPM (Learning Progress Monitor) module")
        print(f"LPM alpha (curiosity scaling): {self.lpm_alpha}")
        
        # Create RND module (unchanged)
        self.rnd = RND(obs_size, act_size, state_size, hidden_size, device, use_bn, use_ln)
        
        # Error predictor network - predicts RND prediction error
        # Input: target_network_features + action, Output: scalar prediction error
        # Note: Uses target network features for consistent representation
        if len(obs_size) == 1:
            # Vector observations
            input_size = state_size + act_size  # state_size is the target network output dim
            if use_bn:
                self.error_predictor = nn.Sequential(
                    nn.Linear(input_size, hidden_size, bias=False),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size, bias=False),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)  # Predict scalar error
                )
            else:
                self.error_predictor = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)  # Predict scalar error
                )
            
            if use_ln:
                # Note: LayerNorm on scalar output doesn't make sense, so we apply it before the final layer
                self.error_predictor = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                )
                
        elif len(obs_size) == 3:
            # Visual observations - use target network features + action
            # Error predictor works directly on target features
            input_size = state_size + act_size
            
            if use_bn:
                self.error_predictor = nn.Sequential(
                    nn.Linear(input_size, hidden_size, bias=False),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size, bias=False),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                )
            else:
                self.error_predictor = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                )
            
            if use_ln:
                self.error_predictor = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                )
        
        # Experience buffer for error predictor training
        # Stores (target_features, act, prediction_error) tuples
        self.buffer_size = buffer_size
        self.error_buffer = deque(maxlen=buffer_size)
        
        # Track statistics
        self.last_rnd_error = 0.
        self.last_predicted_error = 0.
        self.last_error_pred_loss = 0.
        self.last_lpm_curiosity = 0.
        
        self.to(device)
        
    def get_feature_moments(self, obs):
        """Compute mean and std of RND target network outputs for normalization."""
        self.rnd.get_feature_moments(obs)
    
    def predict_error(self, target_features, act):
        """
        Predict the expected RND prediction error from target network features.
        
        Args:
            target_features: Output from frozen target network
            act: Action taken
            
        Returns:
            predicted_error: Predicted RND prediction error (scalar per sample)
        """
        if len(act.shape) == 1:
            act = act.unsqueeze(1)
        inp = torch.cat([target_features, act], dim=1)
        predicted_error = self.error_predictor(inp).squeeze(-1)
        
        return predicted_error

    def forward(self, obs, act=None, next_obs=None):
        """
        Forward pass - uses RND forward pass.
        
        Args:
            obs: Current observation
            act: Action (not used in RND forward, kept for API compatibility)
            next_obs: Next observation (this is what RND predicts on)
        
        Returns:
            target_features: Output of RND fixed target network
            predicted_features: Output of RND trainable predictor network
        """
        return self.rnd.forward(obs, act, next_obs)

    def loss(self, obs, act, next_obs, *args, **kwargs):
        """
        Compute the RND prediction loss for training the RND predictor network.
        Also stores experiences in buffer for error predictor training.
        
        Args:
            obs: Current observation
            act: Action taken
            next_obs: Next observation
        
        Returns:
            loss: MSE between RND predicted and target features
        """
        # Get RND loss (this trains the RND predictor)
        rnd_loss = self.rnd.loss(obs, act, next_obs)
        
        # Get target features for storing in buffer
        with torch.no_grad():
            target_features, _ = self.rnd.forward(obs, act, next_obs)
            
            # Store (target_features, act, prediction_error) in buffer
            for i in range(next_obs.shape[0]):
                self.error_buffer.append({
                    'target_features': target_features[i].detach().cpu(),
                    'act': act[i].detach().cpu(),
                    'error': rnd_loss[i].detach().cpu()
                })
        
        # Update statistics
        self.last_rnd_error = rnd_loss.detach().mean().cpu().item()
        
        return rnd_loss
    
    def error_predictor_loss(self, batch_size=32):
        """
        Compute loss for training the error predictor network.
        Samples from the experience buffer and trains to predict RND errors.
        
        Args:
            batch_size: Number of samples to use for training
            
        Returns:
            loss: MSE between predicted and actual RND errors (or None if buffer too small)
        """
        if len(self.error_buffer) < batch_size:
            return None
        
        # Sample from buffer
        indices = np.random.choice(len(self.error_buffer), size=batch_size, replace=False)
        batch = [self.error_buffer[i] for i in indices]
        
        # Prepare batch tensors
        target_features_batch = torch.stack([item['target_features'] for item in batch]).to(self.device)
        act_batch = torch.stack([item['act'] for item in batch]).to(self.device)
        error_batch = torch.stack([item['error'] for item in batch]).to(self.device)
        
        # Predict errors from target features
        predicted_errors = self.predict_error(target_features_batch, act_batch)
        
        # MSE loss
        loss = F.mse_loss(predicted_errors, error_batch)
        
        # Update statistics
        self.last_predicted_error = predicted_errors.detach().mean().cpu().item()
        self.last_error_pred_loss = loss.detach().cpu().item()
        
        return loss

    def curiosity(self, obs, act, next_obs, *args, **kwargs):
        """
        Compute intrinsic reward (curiosity) based on learning progress.
        
        Curiosity = RND_prediction_error - alpha * predicted_error
        
        This represents learning progress: if we can predict the error well,
        it means we've seen similar states. The difference indicates genuine novelty.
        
        Args:
            obs: Current observation
            act: Action taken
            next_obs: Next observation
        
        Returns:
            curiosity: Learning progress as intrinsic reward
        """
        # Get RND prediction error (novelty) and target features
        rnd_curiosity = self.rnd.curiosity(obs, act, next_obs)
        
        # Get target features from next_obs
        with torch.no_grad():
            target_features, _ = self.rnd.forward(obs, act, next_obs)
            
            # Predict expected error from target features
            predicted_error = self.predict_error(target_features, act)
            # Ensure non-negative
            predicted_error = torch.clamp(predicted_error, min=0.0)
        
        # Learning progress = actual error - expected error
        # If we can predict the error well, subtract it (not novel)
        # If we can't predict it, keep it (genuinely novel)
        lpm_curiosity = rnd_curiosity - self.lpm_alpha * predicted_error
        
        # Ensure non-negative curiosity
        # lpm_curiosity = torch.clamp(lpm_curiosity, min=0.0)
        
        # Update statistics
        self.last_lpm_curiosity = lpm_curiosity.detach().mean().cpu().item()
        
        return lpm_curiosity

