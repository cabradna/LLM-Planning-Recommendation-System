import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import numpy as np
from src.environments.job_env import JobRecommendationEnv
from src.data.database import DatabaseConnector
from src.config.config import ENV_CONFIG, DB_CONFIG, MODEL_CONFIG
import torch

def test_environment_initialization():
    """Test environment initialization with config parameters"""
    db_connector = DatabaseConnector()
    env = JobRecommendationEnv(
        db_connector=db_connector,
        reward_strategy="cosine",
        random_seed=ENV_CONFIG["random_seed"]
    )
    
    assert env.reward_strategy == "cosine"
    assert env.db is not None

def test_reset():
    """Test environment reset functionality"""
    db_connector = DatabaseConnector()
    env = JobRecommendationEnv(
        db_connector=db_connector,
        reward_strategy="cosine",
        random_seed=ENV_CONFIG["random_seed"]
    )
    
    # We need a valid applicant ID for testing
    applicant_id = "test_applicant_1"  # This should be a valid ID in your test database
    state = env.reset(applicant_id)
    
    assert isinstance(state, torch.Tensor)
    assert len(env.candidate_jobs) > 0
    assert len(env.job_vectors) > 0

def test_step():
    """Test environment step functionality"""
    db_connector = DatabaseConnector()
    env = JobRecommendationEnv(
        db_connector=db_connector,
        reward_strategy="cosine",
        random_seed=ENV_CONFIG["random_seed"]
    )
    
    # Reset with a valid applicant ID
    applicant_id = "test_applicant_1"  # This should be a valid ID in your test database
    env.reset(applicant_id)
    
    # Take a step with a valid action
    action = 0  # First job in the candidate list
    next_state, reward, done, info = env.step(action)
    
    assert isinstance(next_state, torch.Tensor)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert -1.0 <= reward <= 1.0  # Cosine similarity range 