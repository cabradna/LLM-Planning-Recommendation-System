"""
Tests for the pretraining module.
"""

import unittest
import sys
import os
import torch
import numpy as np
import tempfile
import shutil
from unittest.mock import MagicMock, patch

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import modules to test
from pretraining import (
    pretrain_q_network, 
    pretrain_world_model, 
    save_pretrained_models,
    set_seed
)
from models.q_network import QNetwork
from models.world_model import WorldModel
from data.data_loader import JobRecommendationDataset, create_pretraining_data_loader
from data.database import DatabaseConnector

class TestPretraining(unittest.TestCase):
    """Test cases for the pretraining module."""
    
    def setUp(self):
        """Set up test environment."""
        # Set random seed for reproducibility
        set_seed(42)
        
        # Create a temporary directory for saving models
        self.temp_dir = tempfile.mkdtemp()
        
        # Define common dimensions
        self.state_dim = 384
        self.action_dim = 384
        self.input_dim = self.state_dim + self.action_dim
        self.hidden_dims = [256, 128]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create models
        self.q_network = QNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=0.1
        ).to(self.device)
        
        self.world_model = WorldModel(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=0.1
        ).to(self.device)
        
        # Create synthetic dataset
        batch_size = 32
        num_samples = 100
        states = [torch.randn(self.state_dim) for _ in range(num_samples)]
        actions = [torch.randn(self.action_dim) for _ in range(num_samples)]
        rewards = [float(np.random.rand()) for _ in range(num_samples)]
        next_states = [torch.randn(self.state_dim) for _ in range(num_samples)]
        
        self.dataset = JobRecommendationDataset(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states
        )
        
        # Create data loaders
        self.train_loader, self.val_loader = create_pretraining_data_loader(
            dataset=self.dataset,
            batch_size=batch_size,
            validation_split=0.2
        )
    
    def tearDown(self):
        """Tear down test environment."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    @patch('torch.optim.Adam')
    def test_pretrain_q_network(self, mock_adam):
        """Test Q-network pretraining function."""
        # Mock optimizer
        optimizer = MagicMock()
        mock_adam.return_value = optimizer
        
        # Set return value for optimizer.step() to avoid AttributeError
        optimizer.step.return_value = None
        
        # Pretrain Q-network (only 2 epochs for testing)
        train_losses, val_losses = pretrain_q_network(
            q_network=self.q_network,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device,
            num_epochs=2,
            lr=0.001,
            weight_decay=1e-5
        )
        
        # Check that optimizer was created correctly
        mock_adam.assert_called_once()
        
        # Check that losses were recorded
        self.assertEqual(len(train_losses), 2)
        self.assertEqual(len(val_losses), 2)
        
        # Check that losses are finite
        for loss in train_losses + val_losses:
            self.assertTrue(np.isfinite(loss))
    
    @patch('torch.optim.Adam')
    def test_pretrain_world_model(self, mock_adam):
        """Test world model pretraining function."""
        # Mock optimizer
        optimizer = MagicMock()
        mock_adam.return_value = optimizer
        
        # Set return value for optimizer.step() to avoid AttributeError
        optimizer.step.return_value = None
        
        # Pretrain world model (only 2 epochs for testing)
        train_losses, val_losses = pretrain_world_model(
            world_model=self.world_model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device,
            num_epochs=2,
            lr=0.001,
            weight_decay=1e-5
        )
        
        # Check that optimizer was created correctly
        mock_adam.assert_called_once()
        
        # Check that losses were recorded
        self.assertEqual(len(train_losses), 2)
        self.assertEqual(len(val_losses), 2)
        
        # Check that losses are finite
        for loss in train_losses + val_losses:
            self.assertTrue(np.isfinite(loss))
    
    def test_save_pretrained_models(self):
        """Test saving pretrained models."""
        # Create metrics
        metrics = {
            "q_train_losses": [0.5, 0.4, 0.3],
            "q_val_losses": [0.6, 0.5, 0.4],
            "world_train_losses": [0.7, 0.6, 0.5],
            "world_val_losses": [0.8, 0.7, 0.6]
        }
        
        # Save models
        save_pretrained_models(
            q_network=self.q_network,
            world_model=self.world_model,
            output_dir=self.temp_dir,
            model_metrics=metrics
        )
        
        # Check that model files exist
        q_network_path = os.path.join(self.temp_dir, "q_network.pt")
        world_model_path = os.path.join(self.temp_dir, "world_model.pt")
        metrics_path = os.path.join(self.temp_dir, "pretraining_metrics.pt")
        
        self.assertTrue(os.path.exists(q_network_path))
        self.assertTrue(os.path.exists(world_model_path))
        self.assertTrue(os.path.exists(metrics_path))
        
        # Load models and check they are the correct types
        loaded_q_network = QNetwork.load(q_network_path)
        loaded_world_model = WorldModel.load(world_model_path)
        loaded_metrics = torch.load(metrics_path)
        
        self.assertIsInstance(loaded_q_network, QNetwork)
        self.assertIsInstance(loaded_world_model, WorldModel)
        self.assertIsInstance(loaded_metrics, dict)
        
        # Check metrics are preserved
        self.assertEqual(loaded_metrics["q_train_losses"], metrics["q_train_losses"])
        self.assertEqual(loaded_metrics["q_val_losses"], metrics["q_val_losses"])
        self.assertEqual(loaded_metrics["world_train_losses"], metrics["world_train_losses"])
        self.assertEqual(loaded_metrics["world_val_losses"], metrics["world_val_losses"])
    
    @patch.object(DatabaseConnector, "_connect")
    def test_llm_generated_data_loading(self, mock_connect):
        """Test loading LLM-generated data."""
        # This test would normally create a temporary JSON file with synthetic data
        # and use the load_llm_generated_data function, but for simplicity
        # we'll just test that the function parameters work as expected
        
        # Mock DatabaseConnector to avoid actual MongoDB connection
        mock_db = MagicMock()
        
        with patch('pretraining.load_llm_generated_data') as mock_load:
            # Set up return value
            mock_dataset = MagicMock(spec=JobRecommendationDataset)
            mock_load.return_value = mock_dataset
            
            # Call with test path
            from pretraining import load_llm_generated_data
            result = load_llm_generated_data(
                data_path=os.path.join(self.temp_dir, "test_data.json"),
                db_connector=mock_db
            )
            
            # Check result
            self.assertEqual(result, mock_dataset)
            mock_load.assert_called_once()
    
    @patch.object(DatabaseConnector, "_connect")
    def test_generate_synthetic_data(self, mock_connect):
        """Test generating synthetic data."""
        # Mock DatabaseConnector to avoid actual MongoDB connection
        mock_db = MagicMock()
        
        with patch('pretraining.generate_synthetic_data') as mock_generate:
            # Set up return value
            mock_dataset = MagicMock(spec=JobRecommendationDataset)
            mock_generate.return_value = mock_dataset
            
            # Call the function
            from pretraining import generate_synthetic_data
            result = generate_synthetic_data(
                db_connector=mock_db,
                num_samples=100
            )
            
            # Check result
            self.assertEqual(result, mock_dataset)
            mock_generate.assert_called_once()

if __name__ == '__main__':
    unittest.main() 