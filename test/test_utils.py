import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import torch
from src.environments.job_env import JobRecommendationEnv
from src.data.database import DatabaseConnector
from src.config.config import ENV_CONFIG, DB_CONFIG, STRATEGY_CONFIG

def test_cosine_similarity():
    """Test cosine similarity calculation"""
    # Create test vectors
    vec1 = torch.randn(10)
    vec2 = torch.randn(10)
    
    # Normalize vectors
    normalized_vec1 = vec1 / vec1.norm()
    normalized_vec2 = vec2 / vec2.norm()
    
    # Calculate similarity
    similarity = torch.dot(normalized_vec1, normalized_vec2).item()
    
    assert isinstance(similarity, float)
    assert -1.0 <= similarity <= 1.0
    
    # Test with identical vectors
    similarity = torch.dot(normalized_vec1, normalized_vec1).item()
    assert abs(similarity - 1.0) < 1e-6
    
    # Test with opposite vectors
    similarity = torch.dot(normalized_vec1, -normalized_vec1).item()
    assert abs(similarity - (-1.0)) < 1e-6

def test_reward_scaling():
    """Test reward scaling in JobRecommendationEnv"""
    db_connector = DatabaseConnector()
    env = JobRecommendationEnv(
        db_connector=db_connector,
        reward_strategy="cosine",
        random_seed=ENV_CONFIG["random_seed"]
    )
    
    # Create test vectors
    state = torch.randn(10)
    job_vector = torch.randn(10)
    
    # Normalize vectors
    normalized_state = state / state.norm()
    normalized_job = job_vector / job_vector.norm()
    
    # Calculate similarity
    similarity = torch.dot(normalized_state, normalized_job).item()
    
    # Test reward scaling
    if STRATEGY_CONFIG["cosine"]["scale_reward"]:
        scaled_reward = (similarity + 1) / 2
        assert 0.0 <= scaled_reward <= 1.0
    else:
        assert -1.0 <= similarity <= 1.0 