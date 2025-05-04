import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import torch
import numpy as np
from src.models.q_network import QNetwork
from src.config.config import MODEL_CONFIG, TRAINING_CONFIG

def test_q_network_initialization():
    """Test Q-Network initialization with config parameters"""
    model = QNetwork(
        state_dim=MODEL_CONFIG["q_network"]["state_dim"],
        action_dim=MODEL_CONFIG["q_network"]["action_dim"],
        hidden_dims=MODEL_CONFIG["q_network"]["hidden_dims"],
        dropout_rate=MODEL_CONFIG["q_network"]["dropout_rate"],
        activation=MODEL_CONFIG["q_network"]["activation"]
    )
    
    assert model.state_dim == MODEL_CONFIG["q_network"]["state_dim"]
    assert model.action_dim == MODEL_CONFIG["q_network"]["action_dim"]
    assert isinstance(model, torch.nn.Module)

def test_forward_pass():
    """Test forward pass through the network"""
    model = QNetwork(
        state_dim=MODEL_CONFIG["q_network"]["state_dim"],
        action_dim=MODEL_CONFIG["q_network"]["action_dim"],
        hidden_dims=MODEL_CONFIG["q_network"]["hidden_dims"],
        dropout_rate=MODEL_CONFIG["q_network"]["dropout_rate"],
        activation=MODEL_CONFIG["q_network"]["activation"]
    )
    
    # Create test input
    batch_size = 32
    state = torch.randn(batch_size, MODEL_CONFIG["q_network"]["state_dim"])
    action = torch.randn(batch_size, MODEL_CONFIG["q_network"]["action_dim"])
    
    # Test forward pass
    q_values = model(state, action)
    assert q_values.shape == (batch_size, 1)
    assert not torch.isnan(q_values).any()
    assert not torch.isinf(q_values).any()

def test_output_range():
    """Test that Q-values are in a reasonable range"""
    model = QNetwork(
        state_dim=MODEL_CONFIG["q_network"]["state_dim"],
        action_dim=MODEL_CONFIG["q_network"]["action_dim"],
        hidden_dims=MODEL_CONFIG["q_network"]["hidden_dims"],
        dropout_rate=MODEL_CONFIG["q_network"]["dropout_rate"],
        activation=MODEL_CONFIG["q_network"]["activation"]
    )
    
    # Create test input
    batch_size = 32
    state = torch.randn(batch_size, MODEL_CONFIG["q_network"]["state_dim"])
    action = torch.randn(batch_size, MODEL_CONFIG["q_network"]["action_dim"])
    
    # Test forward pass
    q_values = model(state, action)
    
    # Q-values should be finite and in a reasonable range
    assert torch.all(torch.isfinite(q_values))
    assert torch.all(q_values > -1000) and torch.all(q_values < 1000)

def test_update():
    """Test network update functionality"""
    model = QNetwork(
        state_dim=MODEL_CONFIG["q_network"]["state_dim"],
        action_dim=MODEL_CONFIG["q_network"]["action_dim"],
        hidden_dims=MODEL_CONFIG["q_network"]["hidden_dims"],
        dropout_rate=MODEL_CONFIG["q_network"]["dropout_rate"],
        activation=MODEL_CONFIG["q_network"]["activation"]
    )
    
    # Create test batch
    batch_size = 32
    states = torch.randn(batch_size, MODEL_CONFIG["q_network"]["state_dim"])
    actions = torch.randn(batch_size, MODEL_CONFIG["q_network"]["action_dim"])
    rewards = torch.randn(batch_size, 1)
    next_states = torch.randn(batch_size, MODEL_CONFIG["q_network"]["state_dim"])
    dones = torch.zeros(batch_size, 1)
    
    # Test update
    loss = model.update(states, actions, rewards, next_states, dones)
    assert isinstance(loss, float)
    assert not np.isnan(loss)
    assert not np.isinf(loss) 