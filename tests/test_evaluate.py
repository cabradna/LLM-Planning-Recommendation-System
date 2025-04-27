"""
Tests for the evaluation module.
"""

import unittest
import sys
import os
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from evaluate import (
    evaluate_model_performance,
    compare_baseline_and_pretrained,
    evaluate_cold_start_performance,
    evaluate_performance_over_time,
    set_seed
)
from training.agent import DynaQAgent
from environments.job_env import JobRecommendationEnv

class TestEvaluate(unittest.TestCase):
    """Test cases for the evaluation module."""
    
    def setUp(self):
        """Set up test environment."""
        # Set random seed for reproducibility
        set_seed(42)
        
        # Create mock environment
        self.mock_env = MagicMock(spec=JobRecommendationEnv)
        
        # Create mock agents
        self.baseline_agent = MagicMock(spec=DynaQAgent)
        self.pretrained_agent = MagicMock(spec=DynaQAgent)
        
        # Prepare mock state and actions
        self.state_dim = 384
        self.action_dim = 384
        self.state = torch.randn(self.state_dim)
        self.actions = [torch.randn(self.action_dim) for _ in range(10)]
        
        # Configure mocks
        self.mock_env.reset.return_value = self.state
        self.mock_env.get_valid_actions.return_value = list(range(10))
        self.mock_env.get_action_vector.side_effect = lambda idx: self.actions[idx]
        self.mock_env.step.return_value = (self.state, 0.5, False, {"response": "CLICK"})
        
        # Set up agent mocks
        self.baseline_agent.select_action.return_value = (0, self.actions[0])
        self.pretrained_agent.select_action.return_value = (1, self.actions[1])
        self.baseline_agent.evaluate.return_value = 0.3
        self.pretrained_agent.evaluate.return_value = 0.5
        
        # Sample applicant IDs
        self.applicant_ids = [f"applicant_{i}" for i in range(5)]
    
    def test_evaluate_model_performance(self):
        """Test model performance evaluation."""
        # Call function
        result = evaluate_model_performance(
            agent=self.baseline_agent,
            env=self.mock_env,
            applicant_ids=self.applicant_ids,
            num_episodes=2,
            agent_name="Baseline"
        )
        
        # Check result
        self.assertIsInstance(result, dict)
        self.assertIn("total_reward", result)
        self.assertIn("average_reward", result)
        self.assertIn("rewards", result)
        self.assertIn("apply_rate", result)
        self.assertIn("responses", result)
        
        # Check method calls
        self.assertEqual(self.mock_env.reset.call_count, 2)
        self.assertEqual(self.baseline_agent.select_action.call_count, 2)
        self.assertEqual(self.mock_env.step.call_count, 2)
    
    def test_compare_baseline_and_pretrained(self):
        """Test comparison between baseline and pretrained agents."""
        # Call function
        result = compare_baseline_and_pretrained(
            baseline_agent=self.baseline_agent,
            pretrained_agent=self.pretrained_agent,
            env=self.mock_env,
            applicant_ids=self.applicant_ids,
            num_episodes=2
        )
        
        # Check result
        self.assertIsInstance(result, dict)
        self.assertIn("baseline", result)
        self.assertIn("pretrained", result)
        self.assertIn("improvements", result)
        
        # Check improvements
        improvements = result["improvements"]
        self.assertIn("reward_improvement", improvements)
        self.assertIn("percentage_improvement", improvements)
        self.assertIn("apply_rate_improvement", improvements)
    
    def test_evaluate_cold_start_performance(self):
        """Test cold start performance evaluation."""
        # Call function
        result = evaluate_cold_start_performance(
            baseline_agent=self.baseline_agent,
            pretrained_agent=self.pretrained_agent,
            env=self.mock_env,
            applicant_ids=self.applicant_ids,
            num_episodes=2,
            num_steps=3
        )
        
        # Check result
        self.assertIsInstance(result, dict)
        self.assertIn("baseline_rewards", result)
        self.assertIn("pretrained_rewards", result)
        self.assertIn("baseline_apply_rates", result)
        self.assertIn("pretrained_apply_rates", result)
        self.assertIn("improvement_over_time", result)
        
        # Check improvements over time
        improvements = result["improvement_over_time"]
        self.assertEqual(len(improvements), 3)  # num_steps
    
    def test_evaluate_performance_over_time(self):
        """Test performance evaluation over time."""
        # Call function
        result = evaluate_performance_over_time(
            agent=self.baseline_agent,
            env=self.mock_env,
            applicant_ids=self.applicant_ids,
            steps=[100, 200, 300]
        )
        
        # Check result
        self.assertIsInstance(result, dict)
        self.assertIn("rewards", result)
        self.assertIn("apply_rates", result)
        
        # Check lengths
        self.assertEqual(len(result["rewards"]), 3)  # 3 steps
        self.assertEqual(len(result["apply_rates"]), 3)  # 3 steps
        
        # Check agent evaluate calls
        self.assertEqual(self.baseline_agent.evaluate.call_count, 3)

if __name__ == '__main__':
    unittest.main() 