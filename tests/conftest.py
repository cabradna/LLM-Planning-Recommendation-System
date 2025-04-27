"""
Pytest configuration file with fixtures for testing.
"""

import pytest
import sys
import os
import torch
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from models.q_network import QNetwork
from models.world_model import WorldModel
from training.agent import DynaQAgent
from data.data_loader import ReplayBuffer, JobRecommendationDataset

@pytest.fixture(scope="function")
def seed_all():
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@pytest.fixture(scope="function")
def device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="function")
def dimensions():
    """Get standard dimensions for testing."""
    return {
        "state_dim": 384,
        "action_dim": 384,
        "hidden_dims": [256, 128],
        "input_dim": 768  # state_dim + action_dim
    }

@pytest.fixture(scope="function")
def q_network(dimensions, device, seed_all):
    """Create a Q-Network for testing."""
    return QNetwork(
        state_dim=dimensions["state_dim"],
        action_dim=dimensions["action_dim"],
        hidden_dims=dimensions["hidden_dims"],
        dropout_rate=0.1
    ).to(device)

@pytest.fixture(scope="function")
def world_model(dimensions, device, seed_all):
    """Create a World Model for testing."""
    return WorldModel(
        input_dim=dimensions["input_dim"],
        hidden_dims=dimensions["hidden_dims"],
        dropout_rate=0.1
    ).to(device)

@pytest.fixture(scope="function")
def dyna_q_agent(dimensions, device, seed_all):
    """Create a DynaQ Agent for testing."""
    return DynaQAgent(
        state_dim=dimensions["state_dim"],
        action_dim=dimensions["action_dim"],
        device=device
    )

@pytest.fixture(scope="function")
def replay_buffer():
    """Create a replay buffer for testing."""
    return ReplayBuffer(capacity=100)

@pytest.fixture(scope="function")
def dummy_experience_batch(dimensions, device):
    """Create a batch of dummy experiences for testing."""
    batch_size = 4
    
    states = torch.randn(batch_size, dimensions["state_dim"], device=device)
    actions = torch.randn(batch_size, dimensions["action_dim"], device=device)
    rewards = torch.randn(batch_size, 1, device=device)
    next_states = torch.randn(batch_size, dimensions["state_dim"], device=device)
    
    return states, actions, rewards, next_states

@pytest.fixture(scope="function")
def job_recommendation_dataset(dimensions, device, dummy_experience_batch):
    """Create a dataset for testing."""
    states, actions, rewards, next_states = dummy_experience_batch
    
    # Convert to lists for dataset creation
    states_list = [state for state in states]
    actions_list = [action for action in actions]
    rewards_list = [reward.item() for reward in rewards]
    next_states_list = [next_state for next_state in next_states]
    
    return JobRecommendationDataset(
        states=states_list,
        actions=actions_list,
        rewards=rewards_list,
        next_states=next_states_list
    ) 