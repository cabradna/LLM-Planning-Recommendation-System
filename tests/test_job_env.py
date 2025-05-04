"""
Tests for the job recommendation environment.
"""

import unittest
import sys
import os
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.environments.job_env import JobRecommendationEnv, LLMSimulatorEnv
from src.data.database import DatabaseConnector
from src.data.tensor_cache import TensorCache
from config.config import ENV_CONFIG, DB_CONFIG, STRATEGY_CONFIG

class TestJobRecommendationEnv(unittest.TestCase):
    """Test cases for the JobRecommendationEnv class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock database connector
        self.mock_db = MagicMock()
        
        # Set up mock database methods
        self.mock_db.get_applicant_state.return_value = torch.randn(384)
        
        # Sample jobs with IDs 1-10
        self.mock_jobs = [{"_id": f"job_{i}", "job_title": f"Job {i}"} for i in range(10)]
        self.mock_db.sample_candidate_jobs.return_value = self.mock_jobs
        
        # Return random vectors for job embeddings
        self.mock_job_vectors = [torch.randn(384) for _ in range(10)]
        self.mock_db.get_job_vectors.return_value = self.mock_job_vectors
        
        # Create environment with mock database
        self.env = JobRecommendationEnv(
            db_connector=self.mock_db,
            random_seed=42
        )
    
    def test_reset(self):
        """Test environment reset."""
        # Reset environment
        state = self.env.reset("applicant_1")
        
        # Check that database methods were called correctly
        self.mock_db.get_applicant_state.assert_called_once_with("applicant_1")
        self.mock_db.sample_candidate_jobs.assert_called_once()
        self.mock_db.get_job_vectors.assert_called_once()
        
        # Check that state is correct
        self.assertEqual(state.shape, torch.Size([384]))
        self.assertTrue(torch.equal(state, self.mock_db.get_applicant_state.return_value))
        
        # Check that job vectors were stored
        self.assertEqual(len(self.env.job_vectors), len(self.mock_job_vectors))
        
        # Check that job ID mapping was created
        self.assertEqual(len(self.env.job_id_to_idx), len(self.mock_jobs))
    
    def test_step(self):
        """Test environment step function."""
        # Reset environment first
        state = self.env.reset("applicant_1")
        
        # Take a step
        action_idx = 0
        next_state, reward, done, info = self.env.step(action_idx)
        
        # Check that next_state is the same as current_state
        self.assertTrue(torch.equal(next_state, state))
        
        # Check that done is False
        self.assertFalse(done)
        
        # Check that reward is a float
        self.assertIsInstance(reward, float)
        
        # Check that info contains expected keys
        self.assertIn("job_id", info)
        self.assertIn("job_title", info)
        self.assertIn("applicant_id", info)
        
        # Check that info values are correct
        self.assertEqual(info["job_id"], self.mock_jobs[action_idx]["_id"])
        self.assertEqual(info["job_title"], self.mock_jobs[action_idx]["job_title"])
        self.assertEqual(info["applicant_id"], "applicant_1")
    
    def test_get_valid_actions(self):
        """Test getting valid actions."""
        # Reset environment first
        self.env.reset("applicant_1")
        
        # Get valid actions
        valid_actions = self.env.get_valid_actions()
        
        # Check that valid actions match the number of jobs
        self.assertEqual(len(valid_actions), len(self.mock_jobs))
        self.assertEqual(valid_actions, list(range(len(self.mock_jobs))))
    
    def test_get_action_vector(self):
        """Test getting action vector."""
        # Reset environment first
        self.env.reset("applicant_1")
        
        # Get action vector
        action_idx = 0
        action_vector = self.env.get_action_vector(action_idx)
        
        # Check that action vector matches the corresponding job vector
        self.assertTrue(torch.equal(action_vector, self.mock_job_vectors[action_idx]))
        
        # Check that invalid action index raises ValueError
        with self.assertRaises(ValueError):
            self.env.get_action_vector(-1)
        
        with self.assertRaises(ValueError):
            self.env.get_action_vector(len(self.mock_jobs))

class TestLLMSimulatorEnv(unittest.TestCase):
    """Test cases for the LLMSimulatorEnv class."""
    
    def test_inheritance(self):
        """Test that LLMSimulatorEnv inherits from JobRecommendationEnv."""
        # Create mock database
        mock_db = MagicMock()
        
        # Create LLM simulator environment
        env = LLMSimulatorEnv(db_connector=mock_db)
        
        # Check inheritance
        self.assertIsInstance(env, JobRecommendationEnv)
    
    def test_simulate_user_response_overridden(self):
        """Test that simulate_user_response is overridden."""
        # Create mock database
        mock_db = MagicMock()
        
        # Create both environments
        base_env = JobRecommendationEnv(db_connector=mock_db, random_seed=42)
        llm_env = LLMSimulatorEnv(db_connector=mock_db, random_seed=42)
        
        # Check that methods have different code objects
        self.assertNotEqual(
            base_env.simulate_user_response.__code__, 
            llm_env.simulate_user_response.__code__
        )
        
        # Test that the LLM simulator returns a valid response
        job = {"_id": "job_1", "job_title": "Test Job"}
        response = llm_env.simulate_user_response(job)
        
        # Check that response is a valid type
        self.assertIn(response, ["APPLY", "SAVE", "CLICK", "IGNORE"])

if __name__ == '__main__':
    unittest.main() 