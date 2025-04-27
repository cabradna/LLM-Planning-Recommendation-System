"""
Tests for the DynaQ Agent.
"""

import unittest
import torch
import sys
import os
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from training.agent import DynaQAgent
from models.q_network import QNetwork
from models.world_model import WorldModel

class TestDynaQAgent(unittest.TestCase):
    """Test cases for the DynaQAgent class."""
    
    def setUp(self):
        """Set up test environment."""
        self.state_dim = 384
        self.action_dim = 384
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create a DynaQ Agent
        self.agent = DynaQAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
    
    def test_initialization(self):
        """Test agent initialization."""
        # Check that networks are created
        self.assertIsInstance(self.agent.q_network, QNetwork)
        self.assertIsInstance(self.agent.target_network, QNetwork)
        self.assertIsInstance(self.agent.world_model, WorldModel)
        
        # Check replay buffer
        self.assertEqual(len(self.agent.replay_buffer), 0)
        
        # Check hyperparameters
        self.assertEqual(self.agent.state_dim, self.state_dim)
        self.assertEqual(self.agent.action_dim, self.action_dim)
    
    def test_select_action(self):
        """Test action selection."""
        batch_size = 1
        num_actions = 5
        
        # Create dummy state tensor
        state = torch.randn(batch_size, self.state_dim, device=self.device)
        
        # Create dummy available actions
        available_actions = [torch.randn(self.action_dim, device=self.device) for _ in range(num_actions)]
        
        # Test in evaluation mode
        action_idx, action_tensor = self.agent.select_action(
            state=state, 
            available_actions=available_actions, 
            eval_mode=True
        )
        
        # Check that action_idx is valid
        self.assertGreaterEqual(action_idx, 0)
        self.assertLess(action_idx, num_actions)
        
        # Check that action_tensor matches the selected action
        self.assertTrue(torch.equal(action_tensor, available_actions[action_idx]))
        
        # Test in training mode
        self.agent.epsilon = 1.0  # Always explore
        action_idx, action_tensor = self.agent.select_action(
            state=state, 
            available_actions=available_actions, 
            eval_mode=False
        )
        
        # Check that action_idx is valid
        self.assertGreaterEqual(action_idx, 0)
        self.assertLess(action_idx, num_actions)
    
    def test_update_epsilon(self):
        """Test epsilon updating."""
        # Set initial epsilon
        initial_epsilon = 1.0
        self.agent.epsilon = initial_epsilon
        self.agent.epsilon_end = 0.1
        self.agent.epsilon_decay = 0.5
        
        # Update epsilon
        self.agent.update_epsilon()
        
        # Check that epsilon decreased
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertEqual(self.agent.epsilon, initial_epsilon * self.agent.epsilon_decay)
    
    def test_store_experience(self):
        """Test experience storage."""
        # Create dummy experience
        state = torch.randn(self.state_dim, device=self.device)
        action = torch.randn(self.action_dim, device=self.device)
        reward = 1.0
        next_state = torch.randn(self.state_dim, device=self.device)
        
        # Initial buffer size
        initial_size = len(self.agent.replay_buffer)
        
        # Store experience
        self.agent.store_experience(state, action, reward, next_state)
        
        # Check that buffer size increased
        self.assertEqual(len(self.agent.replay_buffer), initial_size + 1)
    
    def test_update_target_network(self):
        """Test target network update."""
        # Modify Q-network parameters
        for param in self.agent.q_network.parameters():
            param.data = torch.randn_like(param)
        
        # Parameters should be different before update
        for q_param, target_param in zip(self.agent.q_network.parameters(), self.agent.target_network.parameters()):
            self.assertFalse(torch.allclose(q_param, target_param))
        
        # Update target network
        self.agent.update_target_network()
        
        # Parameters should be the same after update
        for q_param, target_param in zip(self.agent.q_network.parameters(), self.agent.target_network.parameters()):
            self.assertTrue(torch.allclose(q_param, target_param))

if __name__ == '__main__':
    unittest.main() 