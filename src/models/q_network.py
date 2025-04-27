"""
Q-Network model for value function approximation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Any

# Import configuration
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config.config import MODEL_CONFIG

class QNetwork(nn.Module):
    """
    Neural network for approximating the Q-function (action-value function).
    
    Takes a state-action pair (s, a) as input and outputs a scalar Q-value Q(s, a),
    representing the expected cumulative discounted reward of taking action a in state s.
    """
    
    def __init__(self, state_dim: Optional[int] = None, action_dim: Optional[int] = None, 
                 hidden_dims: Optional[List[int]] = None, dropout_rate: Optional[float] = None,
                 activation: str = "relu"):
        """
        Initialize the Q-Network.
        
        Args:
            state_dim: Dimension of the state vector. If None, uses config value.
            action_dim: Dimension of the action vector. If None, uses config value.
            hidden_dims: List of hidden layer dimensions. If None, uses config value.
            dropout_rate: Dropout rate for regularization. If None, uses config value.
            activation: Activation function to use ('relu', 'leaky_relu', 'tanh').
        """
        super(QNetwork, self).__init__()
        
        # Set dimensions from config if not specified
        self.state_dim = state_dim or MODEL_CONFIG["q_network"]["state_dim"]
        self.action_dim = action_dim or MODEL_CONFIG["q_network"]["action_dim"]
        self.hidden_dims = hidden_dims or MODEL_CONFIG["q_network"]["hidden_dims"]
        self.dropout_rate = dropout_rate or MODEL_CONFIG["q_network"]["dropout_rate"]
        
        # Input dimension is state_dim + action_dim (concatenated vectors)
        input_dim = self.state_dim + self.action_dim
        
        # Build the network
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, self.hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(self.hidden_dims) - 1):
            self.layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
        
        # Output layer (single Q-value)
        self.output_layer = nn.Linear(self.hidden_dims[-1], 1)
        
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
            torch.Tensor: Q-values tensor of shape (batch_size, 1)
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        
        # Forward pass through hidden layers
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            if self.dropout:
                x = self.dropout(x)
        
        # Output layer (no activation - Q-value can be any real number)
        q_value = self.output_layer(x)
        
        return q_value
    
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
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_name
        }, path)
    
    @classmethod
    def load(cls, path: str, device: torch.device = None) -> 'QNetwork':
        """
        Load a model from disk.
        
        Args:
            path: Path to load the model from.
            device: Device to load the model to.
            
        Returns:
            QNetwork: Loaded model.
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=device if device else torch.device('cpu'))
        
        # Create model with same architecture
        model = cls(
            state_dim=checkpoint['state_dim'],
            action_dim=checkpoint['action_dim'],
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