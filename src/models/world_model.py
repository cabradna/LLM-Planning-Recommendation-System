"""
World Model for predicting environment dynamics (rewards and next states).

This model is used in the Dyna-Q algorithm's planning phase to simulate
experiences without interacting with the real environment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any

# Import configuration
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config.config import MODEL_CONFIG

class WorldModel(nn.Module):
    """
    Neural network for modeling environment dynamics.
    
    Takes a state-action pair (s, a) and predicts the resulting reward R.
    In the future, this could be extended to also predict the next state s',
    but in the current implementation where states are static, this is omitted.
    """
    
    def __init__(self, input_dim: Optional[int] = None, hidden_dims: Optional[List[int]] = None,
                 dropout_rate: Optional[float] = None, activation: str = "relu"):
        """
        Initialize the World Model.
        
        Args:
            input_dim: Dimension of the input vector (state_dim + action_dim).
                       If None, uses config value.
            hidden_dims: List of hidden layer dimensions. If None, uses config value.
            dropout_rate: Dropout rate for regularization. If None, uses config value.
            activation: Activation function to use ('relu', 'leaky_relu', 'tanh').
        """
        super(WorldModel, self).__init__()
        
        # Set dimensions from config if not specified
        self.input_dim = input_dim or MODEL_CONFIG["world_model"]["input_dim"]
        self.hidden_dims = hidden_dims or MODEL_CONFIG["world_model"]["hidden_dims"]
        self.dropout_rate = dropout_rate or MODEL_CONFIG["world_model"]["dropout_rate"]
        
        # Build reward prediction network
        self.reward_layers = nn.ModuleList()
        
        # Input layer
        self.reward_layers.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(self.hidden_dims) - 1):
            self.reward_layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
        
        # Output layer for reward prediction (single scalar)
        self.reward_output = nn.Linear(self.hidden_dims[-1], 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        
        # Set activation function
        self.activation_name = activation
        if activation == "relu":
            self.activation = F.relu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size, action_dim)
            
        Returns:
            torch.Tensor: Predicted rewards of shape (batch_size, 1)
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        
        # Forward pass through reward prediction network
        for layer in self.reward_layers:
            x = layer(x)
            x = self.activation(x)
            if self.dropout:
                x = self.dropout(x)
        
        # Output layer for reward prediction
        predicted_reward = self.reward_output(x)
        
        return predicted_reward
    
    def predict_reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict the reward for a state-action pair.
        
        This is just an alias for forward() for clarity.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size, action_dim)
            
        Returns:
            torch.Tensor: Predicted rewards of shape (batch_size, 1)
        """
        return self.forward(state, action)
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_name
        }, path)
    
    @classmethod
    def load(cls, path: str, device: torch.device = None) -> 'WorldModel':
        """
        Load a model from disk.
        
        Args:
            path: Path to load the model from.
            device: Device to load the model to.
            
        Returns:
            WorldModel: Loaded model.
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=device if device else torch.device('cpu'))
        
        # Create model with same architecture
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dims=checkpoint['hidden_dims'],
            dropout_rate=checkpoint['dropout_rate'],
            activation=checkpoint['activation']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
        
        # Move to device if specified
        if device:
            model = model.to(device)
            
        return model 