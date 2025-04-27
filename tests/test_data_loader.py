"""
Tests for the data loader module.
"""

import unittest
import sys
import os
import torch
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data.data_loader import ReplayBuffer, JobRecommendationDataset, create_pretraining_data_loader

class TestReplayBuffer(unittest.TestCase):
    """Test cases for the ReplayBuffer class."""
    
    def setUp(self):
        """Set up test environment."""
        self.buffer_capacity = 100
        self.replay_buffer = ReplayBuffer(capacity=self.buffer_capacity)
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create dummy experience
        self.state_dim = 384
        self.action_dim = 384
        self.state = torch.randn(self.state_dim)
        self.action = torch.randn(self.action_dim)
        self.reward = 1.0
        self.next_state = torch.randn(self.state_dim)
    
    def test_push(self):
        """Test adding experiences to the buffer."""
        # Initial size should be 0
        self.assertEqual(len(self.replay_buffer), 0)
        
        # Push a single experience
        self.replay_buffer.push(self.state, self.action, self.reward, self.next_state)
        
        # Buffer size should be 1
        self.assertEqual(len(self.replay_buffer), 1)
        
        # Push more experiences
        for _ in range(9):
            self.replay_buffer.push(self.state, self.action, self.reward, self.next_state)
        
        # Buffer size should be 10
        self.assertEqual(len(self.replay_buffer), 10)
    
    def test_capacity(self):
        """Test that buffer respects its capacity."""
        # Fill the buffer to capacity
        for _ in range(self.buffer_capacity + 10):
            self.replay_buffer.push(self.state, self.action, self.reward, self.next_state)
        
        # Buffer size should be equal to capacity
        self.assertEqual(len(self.replay_buffer), self.buffer_capacity)
    
    def test_sample(self):
        """Test sampling from the buffer."""
        # Fill the buffer
        for _ in range(self.buffer_capacity):
            self.replay_buffer.push(
                torch.randn(self.state_dim),
                torch.randn(self.action_dim),
                float(np.random.rand()),
                torch.randn(self.state_dim)
            )
        
        # Sample from buffer
        batch_size = 32
        states, actions, rewards, next_states = self.replay_buffer.sample(batch_size)
        
        # Check shapes
        self.assertEqual(states.shape, (batch_size, self.state_dim))
        self.assertEqual(actions.shape, (batch_size, self.action_dim))
        self.assertEqual(rewards.shape, (batch_size, 1))
        self.assertEqual(next_states.shape, (batch_size, self.state_dim))
        
        # Check types
        self.assertTrue(isinstance(states, torch.Tensor))
        self.assertTrue(isinstance(actions, torch.Tensor))
        self.assertTrue(isinstance(rewards, torch.Tensor))
        self.assertTrue(isinstance(next_states, torch.Tensor))

class TestJobRecommendationDataset(unittest.TestCase):
    """Test cases for the JobRecommendationDataset class."""
    
    def setUp(self):
        """Set up test environment."""
        self.batch_size = 32
        self.state_dim = 384
        self.action_dim = 384
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create dummy data
        self.states = [torch.randn(self.state_dim) for _ in range(self.batch_size)]
        self.actions = [torch.randn(self.action_dim) for _ in range(self.batch_size)]
        self.rewards = [float(np.random.rand()) for _ in range(self.batch_size)]
        self.next_states = [torch.randn(self.state_dim) for _ in range(self.batch_size)]
        
        # Create dataset
        self.dataset = JobRecommendationDataset(
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            next_states=self.next_states
        )
    
    def test_len(self):
        """Test dataset length."""
        self.assertEqual(len(self.dataset), self.batch_size)
    
    def test_getitem(self):
        """Test getting items from the dataset."""
        # Get first item
        state, action, reward, next_state = self.dataset[0]
        
        # Check that item matches the original data
        self.assertTrue(torch.equal(state, self.states[0]))
        self.assertTrue(torch.equal(action, self.actions[0]))
        self.assertEqual(reward, self.rewards[0])
        self.assertTrue(torch.equal(next_state, self.next_states[0]))

class TestDataLoader(unittest.TestCase):
    """Test cases for the data loader functions."""
    
    def setUp(self):
        """Set up test environment."""
        self.batch_size = 32
        self.state_dim = 384
        self.action_dim = 384
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create dummy data
        states = [torch.randn(self.state_dim) for _ in range(100)]
        actions = [torch.randn(self.action_dim) for _ in range(100)]
        rewards = [float(np.random.rand()) for _ in range(100)]
        next_states = [torch.randn(self.state_dim) for _ in range(100)]
        
        # Create dataset
        self.dataset = JobRecommendationDataset(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states
        )
    
    def test_create_pretraining_data_loader_no_validation(self):
        """Test creating data loader without validation split."""
        train_loader, val_loader = create_pretraining_data_loader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            validation_split=0.0
        )
        
        # Check that train loader has all samples
        self.assertEqual(len(train_loader.dataset), len(self.dataset))
        
        # Check that val loader is None
        self.assertIsNone(val_loader)
        
        # Check that batch size is correct
        self.assertEqual(train_loader.batch_size, self.batch_size)
    
    def test_create_pretraining_data_loader_with_validation(self):
        """Test creating data loader with validation split."""
        validation_split = 0.2
        train_loader, val_loader = create_pretraining_data_loader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            validation_split=validation_split
        )
        
        # Check that train and val loaders have correct number of samples
        expected_train_size = int(len(self.dataset) * (1 - validation_split))
        expected_val_size = len(self.dataset) - expected_train_size
        
        self.assertEqual(len(train_loader.dataset), expected_train_size)
        self.assertEqual(len(val_loader.dataset), expected_val_size)
        
        # Check that batch sizes are correct
        self.assertEqual(train_loader.batch_size, self.batch_size)
        self.assertEqual(val_loader.batch_size, self.batch_size)

if __name__ == '__main__':
    unittest.main() 