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
from src.config.config import (
    ENV_CONFIG,
    MODEL_CONFIG,
    TRAINING_CONFIG,
    DB_CONFIG
)

@pytest.fixture
def device():
    """Fixture for device (CPU/GPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def sample_state():
    """Fixture for sample state tensor"""
    batch_size = 32
    state_dim = MODEL_CONFIG["q_network"]["state_dim"]
    return torch.randn(batch_size, state_dim)

@pytest.fixture
def sample_action():
    """Fixture for sample action tensor"""
    batch_size = 32
    action_dim = MODEL_CONFIG["q_network"]["action_dim"]
    return torch.randn(batch_size, action_dim)

@pytest.fixture
def sample_batch():
    """Fixture for sample training batch"""
    batch_size = 32
    state_dim = MODEL_CONFIG["q_network"]["state_dim"]
    action_dim = MODEL_CONFIG["q_network"]["action_dim"]
    
    return {
        "states": torch.randn(batch_size, state_dim),
        "actions": torch.randn(batch_size, action_dim),
        "rewards": torch.randn(batch_size, 1),
        "next_states": torch.randn(batch_size, state_dim),
        "dones": torch.zeros(batch_size, 1)
    }

@pytest.fixture
def env():
    """Fixture for environment"""
    from src.environments.job_env import JobRecommendationEnv
    return JobRecommendationEnv(
        db_config=DB_CONFIG,
        max_jobs_per_episode=ENV_CONFIG["max_jobs_per_episode"],
        reward_scaling=ENV_CONFIG["reward_scaling"],
        normalize_rewards=ENV_CONFIG["normalize_rewards"],
        use_tensor_cache=ENV_CONFIG["use_tensor_cache"],
        cache_device=ENV_CONFIG["cache_device"]
    )

@pytest.fixture
def model():
    """Fixture for NeuralDynaQ model"""
    from src.models.neural_dyna_q import NeuralDynaQ
    return NeuralDynaQ(
        state_dim=MODEL_CONFIG["q_network"]["state_dim"],
        action_dim=MODEL_CONFIG["q_network"]["action_dim"],
        hidden_dims=MODEL_CONFIG["q_network"]["hidden_dims"],
        dropout_rate=MODEL_CONFIG["q_network"]["dropout_rate"],
        activation=MODEL_CONFIG["q_network"]["activation"],
        gamma=TRAINING_CONFIG["gamma"],
        lr=TRAINING_CONFIG["lr"],
        device=TRAINING_CONFIG["device"]
    ) 