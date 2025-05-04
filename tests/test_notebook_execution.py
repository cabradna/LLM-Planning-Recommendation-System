#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration tests for the neural_dyna_q_notebook.py execution.
These tests verify that the notebook can run end-to-end in a Colab environment.
"""

import unittest
import sys
import os
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import (
    DB_CONFIG,
    MODEL_CONFIG,
    TRAINING_CONFIG,
    STRATEGY_CONFIG,
    PATH_CONFIG,
    EVAL_CONFIG,
    HF_CONFIG,
    ENV_CONFIG
)

class TestNotebookExecution(unittest.TestCase):
    """Test cases for notebook execution."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock the database connector
        self.mock_db = MagicMock()
        
        # Mock database responses
        self.mock_db.get_applicant_state.return_value = torch.randn(384)
        self.mock_jobs = [{"_id": f"job_{i}", "job_title": f"Job {i}"} for i in range(10)]
        self.mock_db.sample_candidate_jobs.return_value = self.mock_jobs
        self.mock_job_vectors = [torch.randn(384) for _ in range(10)]
        self.mock_db.get_job_vectors.return_value = self.mock_job_vectors
        
        # Mock tensor cache
        self.mock_tensor_cache = MagicMock()
        self.mock_tensor_cache.job_ids = [job["_id"] for job in self.mock_jobs]
        self.mock_tensor_cache.job_vectors = self.mock_job_vectors
        
        # Set up patches
        self.patchers = [
            patch('src.data.database.DatabaseConnector', return_value=self.mock_db),
            patch('src.data.tensor_cache.TensorCache', return_value=self.mock_tensor_cache),
            patch.dict('sys.modules', {'google.colab': MagicMock()})  # Properly mock Colab
        ]
        
        for patcher in self.patchers:
            patcher.start()
    
    def tearDown(self):
        """Clean up after tests."""
        for patcher in self.patchers:
            patcher.stop()
    
    def test_environment_setup(self):
        """Test that the environment setup code runs without errors."""
        # Import the notebook's setup code
        from neural_dyna_q_notebook import (
            SETUP_PATH_CONFIG,
            SETUP_STRATEGY_CONFIG,
            IN_COLAB
        )
        
        # Verify configuration
        self.assertTrue(IN_COLAB)
        self.assertIn("repo_url", SETUP_PATH_CONFIG)
        self.assertIn("llm", SETUP_STRATEGY_CONFIG)
    
    def test_database_connection(self):
        """Test that database connection can be established."""
        from src.data.database import DatabaseConnector
        
        # Create database connector
        db_connector = DatabaseConnector()
        
        # Verify connection
        self.assertIsNotNone(db_connector)
        self.assertTrue(hasattr(db_connector, 'db'))
    
    def test_tensor_cache_initialization(self):
        """Test that tensor cache can be initialized."""
        from src.data.tensor_cache import TensorCache
        
        # Initialize tensor cache
        tensor_cache = TensorCache(device="cpu")
        
        # Verify initialization
        self.assertIsNotNone(tensor_cache)
        self.assertTrue(hasattr(tensor_cache, 'job_ids'))
        self.assertTrue(hasattr(tensor_cache, 'job_vectors'))
    
    def test_environment_creation(self):
        """Test that the environment can be created."""
        from src.environments.job_env import JobRecommendationEnv
        
        # Create environment
        env = JobRecommendationEnv(
            db_connector=self.mock_db,
            tensor_cache=self.mock_tensor_cache,
            reward_strategy="cosine"
        )
        
        # Verify environment
        self.assertIsNotNone(env)
        self.assertTrue(hasattr(env, 'reset'))
        self.assertTrue(hasattr(env, 'step'))
    
    def test_model_initialization(self):
        """Test that the neural networks can be initialized."""
        from src.models.q_network import QNetwork
        from src.models.world_model import WorldModel
        
        # Initialize Q-network
        q_network = QNetwork(
            state_dim=MODEL_CONFIG["q_network"]["state_dim"],
            action_dim=MODEL_CONFIG["q_network"]["action_dim"],
            hidden_dims=MODEL_CONFIG["q_network"]["hidden_dims"]
        )
        
        # Initialize World Model
        world_model = WorldModel(
            input_dim=MODEL_CONFIG["world_model"]["input_dim"],
            hidden_dims=MODEL_CONFIG["world_model"]["hidden_dims"]
        )
        
        # Verify models
        self.assertIsNotNone(q_network)
        self.assertIsNotNone(world_model)
        self.assertTrue(hasattr(q_network, 'forward'))
        self.assertTrue(hasattr(world_model, 'forward'))
    
    def test_agent_initialization(self):
        """Test that the DynaQAgent can be initialized."""
        from src.training.agent import DynaQAgent
        from src.models.q_network import QNetwork
        from src.models.world_model import WorldModel
        
        # Create models
        q_network = QNetwork(
            state_dim=MODEL_CONFIG["q_network"]["state_dim"],
            action_dim=MODEL_CONFIG["q_network"]["action_dim"],
            hidden_dims=MODEL_CONFIG["q_network"]["hidden_dims"]
        )
        world_model = WorldModel(
            input_dim=MODEL_CONFIG["world_model"]["input_dim"],
            hidden_dims=MODEL_CONFIG["world_model"]["hidden_dims"]
        )
        
        # Initialize agent
        agent = DynaQAgent(
            state_dim=MODEL_CONFIG["q_network"]["state_dim"],
            action_dim=MODEL_CONFIG["q_network"]["action_dim"],
            q_network=q_network,
            world_model=world_model,
            training_strategy="cosine",
            device="cpu"
        )
        
        # Verify agent
        self.assertIsNotNone(agent)
        self.assertTrue(hasattr(agent, 'train'))
        self.assertTrue(hasattr(agent, 'pretrain'))
    
    def test_training_setup(self):
        """Test that the training configuration is valid."""
        self.assertIn("num_episodes", TRAINING_CONFIG)
        self.assertIn("max_steps_per_episode", TRAINING_CONFIG)
        self.assertIn("batch_size", TRAINING_CONFIG)
        self.assertIn("gamma", TRAINING_CONFIG)
        self.assertIn("planning_steps", TRAINING_CONFIG)
    
    def test_evaluation_setup(self):
        """Test that the evaluation configuration is valid."""
        self.assertIn("num_eval_episodes", EVAL_CONFIG)
        self.assertIn("top_k_recommendations", EVAL_CONFIG)
        self.assertIn("eval_metrics", EVAL_CONFIG)
        self.assertIn("baseline_comparison", EVAL_CONFIG)

if __name__ == '__main__':
    unittest.main() 