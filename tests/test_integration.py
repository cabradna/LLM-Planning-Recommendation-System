"""
Integration tests for the Dyna-Q job recommender system.

These tests verify that the main components of the system work together correctly.
"""

import unittest
import sys
import os
import torch
import tempfile
import shutil
from unittest.mock import MagicMock, patch

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from models.q_network import QNetwork
from models.world_model import WorldModel
from data.data_loader import JobRecommendationDataset, create_pretraining_data_loader
from training.agent import DynaQAgent
from environments.job_env import JobRecommendationEnv

class TestIntegration(unittest.TestCase):
    """Integration tests for the Dyna-Q job recommender system."""
    
    def setUp(self):
        """Set up test environment."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create a temporary directory for saving models
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock database connector
        self.mock_db = MagicMock()
        
        # Set up mock data
        self.state_dim = 384
        self.action_dim = 384
        
        # Set up mock database methods
        self.mock_db.get_applicant_state.return_value = torch.randn(self.state_dim)
        
        # Sample jobs with IDs 1-10
        self.mock_jobs = [{"original_job_id": f"job_{i}", "job_title": f"Job {i}"} for i in range(10)]
        self.mock_db.sample_candidate_jobs.return_value = self.mock_jobs
        
        # Return random vectors for job embeddings
        self.mock_job_vectors = [torch.randn(self.action_dim) for _ in range(10)]
        self.mock_db.get_job_vectors.return_value = self.mock_job_vectors
    
    def tearDown(self):
        """Tear down test environment."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_pretraining_workflow(self):
        """Test pretraining workflow."""
        # Create models
        q_network = QNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=[128, 64],
            dropout_rate=0.1
        )
        
        world_model = WorldModel(
            input_dim=self.state_dim + self.action_dim,
            hidden_dims=[128, 64],
            dropout_rate=0.1
        )
        
        # Create synthetic dataset
        num_samples = 100
        states = [torch.randn(self.state_dim) for _ in range(num_samples)]
        actions = [torch.randn(self.action_dim) for _ in range(num_samples)]
        rewards = [float(torch.rand(1).item()) for _ in range(num_samples)]
        next_states = [torch.randn(self.state_dim) for _ in range(num_samples)]
        
        dataset = JobRecommendationDataset(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states
        )
        
        # Create data loaders
        train_loader, val_loader = create_pretraining_data_loader(
            dataset=dataset,
            batch_size=32,
            validation_split=0.2
        )
        
        # Pretrain models (just one epoch for testing)
        # Mock the pre-training functions to avoid actual computation
        with patch('torch.optim.Adam') as mock_adam:
            optimizer = MagicMock()
            mock_adam.return_value = optimizer
            
            # Pretrain Q-network (simplified)
            for batch_idx, (batch_states, batch_actions, batch_rewards, _) in enumerate(train_loader):
                # Forward pass
                q_values = q_network(batch_states, batch_actions)
                
                # Compute loss
                loss = torch.nn.functional.mse_loss(q_values, batch_rewards.unsqueeze(1))
                
                # Just check that this runs without errors
                self.assertIsInstance(loss.item(), float)
                
                # Only process one batch
                break
            
            # Pretrain world model (simplified)
            for batch_idx, (batch_states, batch_actions, batch_rewards, _) in enumerate(train_loader):
                # Forward pass
                predicted_rewards = world_model(batch_states, batch_actions)
                
                # Compute loss
                loss = torch.nn.functional.mse_loss(predicted_rewards, batch_rewards.unsqueeze(1))
                
                # Just check that this runs without errors
                self.assertIsInstance(loss.item(), float)
                
                # Only process one batch
                break
        
        # Save models
        q_network_path = os.path.join(self.temp_dir, "q_network.pt")
        world_model_path = os.path.join(self.temp_dir, "world_model.pt")
        
        q_network.save(q_network_path)
        world_model.save(world_model_path)
        
        # Check that model files exist
        self.assertTrue(os.path.exists(q_network_path))
        self.assertTrue(os.path.exists(world_model_path))
        
        # Load models
        loaded_q_network = QNetwork.load(q_network_path)
        loaded_world_model = WorldModel.load(world_model_path)
        
        # Check that models are loaded correctly
        self.assertIsInstance(loaded_q_network, QNetwork)
        self.assertIsInstance(loaded_world_model, WorldModel)
    
    def test_training_workflow(self):
        """Test training workflow."""
        # Create environment
        env = JobRecommendationEnv(db_connector=self.mock_db, random_seed=42)
        
        # Create agent
        agent = DynaQAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=[128, 64],
            replay_buffer_size=1000,
            batch_size=32,
            gamma=0.9,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.99,
            target_update_freq=10,
            planning_steps=5
        )
        
        # Mock out actual optimization to speed up test
        with patch.object(agent, 'update_q_network', return_value=0.1), \
             patch.object(agent, 'update_world_model', return_value=0.1):
            
            # Train for a few episodes
            num_episodes = 2
            max_steps = 3
            applicant_ids = ["applicant_1", "applicant_2"]
            
            # Train agent (simplified)
            for episode in range(num_episodes):
                # Get applicant ID
                applicant_id = applicant_ids[episode % len(applicant_ids)]
                
                # Reset environment
                state = env.reset(applicant_id)
                
                for step in range(max_steps):
                    # Get valid actions
                    valid_action_indices = env.get_valid_actions()
                    available_actions = [env.get_action_vector(idx) for idx in valid_action_indices]
                    
                    # Select action
                    action_idx, action = agent.select_action(state, available_actions)
                    
                    # Take step
                    next_state, reward, done, _ = env.step(action_idx)
                    
                    # Store experience
                    agent.store_experience(state, action, reward, next_state)
                    
                    # Perform Q-learning update
                    if len(agent.replay_buffer) >= agent.batch_size:
                        agent.update_q_network(None, None, None, None)
                        
                        # Perform planning steps
                        for _ in range(agent.planning_steps):
                            agent.update_q_network(None, None, None, None)
                    
                    # Update epsilon
                    agent.update_epsilon()
                    
                    # Update state
                    state = next_state
                    
                    if done:
                        break
                
                # Update target network
                if episode % agent.target_update_freq == 0:
                    agent.update_target_network()
            
            # Save models
            agent.save_models(self.temp_dir, num_episodes)
            
            # Check that model files exist
            q_network_path = os.path.join(self.temp_dir, f"q_network_{num_episodes}.pt")
            world_model_path = os.path.join(self.temp_dir, f"world_model_{num_episodes}.pt")
            
            self.assertTrue(os.path.exists(q_network_path))
            self.assertTrue(os.path.exists(world_model_path))
            
            # Load agent
            loaded_agent = DynaQAgent.load_models(self.temp_dir, num_episodes)
            
            # Check that agent is loaded correctly
            self.assertIsInstance(loaded_agent, DynaQAgent)
            self.assertIsInstance(loaded_agent.q_network, QNetwork)
            self.assertIsInstance(loaded_agent.world_model, WorldModel)

if __name__ == '__main__':
    unittest.main() 