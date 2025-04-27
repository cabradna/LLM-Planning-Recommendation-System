"""
Tests for the Q-Network model.
"""

import unittest
import torch
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from models.q_network import QNetwork

class TestQNetwork(unittest.TestCase):
    """Test cases for the QNetwork class."""
    
    def setUp(self):
        """Set up test environment."""
        self.state_dim = 384
        self.action_dim = 384
        self.hidden_dims = [256, 128]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create a Q-Network
        self.q_network = QNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
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
        q_values = self.q_network(states, actions)
        
        # Check output shape
        self.assertEqual(q_values.shape, (batch_size, 1))
        
        # Check output type
        self.assertTrue(isinstance(q_values, torch.Tensor))
        
        # Check that Q-values are finite
        self.assertTrue(torch.isfinite(q_values).all())
    
    def test_save_load(self):
        """Test saving and loading the model."""
        import tempfile
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            model_path = tmp.name
        
        try:
            # Save the model
            self.q_network.save(model_path)
            
            # Load the model
            loaded_model = QNetwork.load(model_path, self.device)
            
            # Check that the loaded model has the same parameters
            for param1, param2 in zip(self.q_network.parameters(), loaded_model.parameters()):
                self.assertTrue(torch.equal(param1, param2))
                
            # Check that the loaded model has the same hyperparameters
            self.assertEqual(loaded_model.state_dim, self.state_dim)
            self.assertEqual(loaded_model.action_dim, self.action_dim)
            self.assertEqual(loaded_model.hidden_dims, self.hidden_dims)
            
        finally:
            # Clean up
            os.remove(model_path)

if __name__ == '__main__':
    unittest.main() 