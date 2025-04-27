"""
Tests for the World Model.
"""

import unittest
import torch
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from models.world_model import WorldModel

class TestWorldModel(unittest.TestCase):
    """Test cases for the WorldModel class."""
    
    def setUp(self):
        """Set up test environment."""
        self.state_dim = 384
        self.action_dim = 384
        self.input_dim = self.state_dim + self.action_dim
        self.hidden_dims = [256, 128]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create a World Model
        self.world_model = WorldModel(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=0.1
        ).to(self.device)
    
    def test_forward_pass(self):
        """Test forward pass through the network."""
        batch_size = 4
        
        # Create dummy state and action tensors
        states = torch.randn(batch_size, self.state_dim, device=self.device)
        actions = torch.randn(batch_size, self.action_dim, device=self.device)
        
        # Forward pass
        predicted_rewards = self.world_model(states, actions)
        
        # Check output shape
        self.assertEqual(predicted_rewards.shape, (batch_size, 1))
        
        # Check output type
        self.assertTrue(isinstance(predicted_rewards, torch.Tensor))
        
        # Check that rewards are finite
        self.assertTrue(torch.isfinite(predicted_rewards).all())
    
    def test_predict_reward(self):
        """Test reward prediction method."""
        batch_size = 4
        
        # Create dummy state and action tensors
        states = torch.randn(batch_size, self.state_dim, device=self.device)
        actions = torch.randn(batch_size, self.action_dim, device=self.device)
        
        # Get predictions from both methods
        forward_rewards = self.world_model(states, actions)
        predicted_rewards = self.world_model.predict_reward(states, actions)
        
        # Check that both methods return the same results
        self.assertTrue(torch.allclose(forward_rewards, predicted_rewards))
    
    def test_save_load(self):
        """Test saving and loading the model."""
        import tempfile
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            model_path = tmp.name
        
        try:
            # Save the model
            self.world_model.save(model_path)
            
            # Load the model
            loaded_model = WorldModel.load(model_path, self.device)
            
            # Check that the loaded model has the same parameters
            for param1, param2 in zip(self.world_model.parameters(), loaded_model.parameters()):
                self.assertTrue(torch.equal(param1, param2))
                
            # Check that the loaded model has the same hyperparameters
            self.assertEqual(loaded_model.input_dim, self.input_dim)
            self.assertEqual(loaded_model.hidden_dims, self.hidden_dims)
            
        finally:
            # Clean up
            os.remove(model_path)

if __name__ == '__main__':
    unittest.main() 