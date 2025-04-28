"""
Dyna-Q Agent implementation for job recommendation.

This module implements the Dyna-Q algorithm, combining direct reinforcement learning
with model-based planning for efficient learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
from pathlib import Path

# Import configuration and modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config.config import TRAINING_CONFIG, PATH_CONFIG

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.q_network import QNetwork
from models.world_model import WorldModel
from data.data_loader import ReplayBuffer
from environments.job_env import JobRecommendationEnv, LLMSimulatorEnv, HybridEnv

logger = logging.getLogger(__name__)

class DynaQAgent:
    """
    Dyna-Q Agent for job recommendation.
    
    Implements the Dyna-Q algorithm, which combines:
    1. Direct reinforcement learning (Q-learning)
    2. Model-based planning (world model learning and simulation)
    
    The agent supports multiple training strategies as described in the documentation:
    1. Cosine Similarity: Direct semantic matching between applicant and job embeddings
    2. LLM Feedback Only: Using LLM to simulate realistic user feedback
    3. Hybrid Approach: Pre-training with cosine similarity, fine-tuning with LLM feedback
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 q_network: Optional[QNetwork] = None,
                 target_network: Optional[QNetwork] = None,
                 world_model: Optional[WorldModel] = None,
                 training_strategy: str = "cosine",
                 device: Optional[torch.device] = None,
                 target_applicant_id: Optional[str] = None,
                 **kwargs):
        """
        Initialize the Dyna-Q agent.
        
        Args:
            state_dim: Dimension of the state vector.
            action_dim: Dimension of the action vector.
            q_network: Q-Network for value function approximation. If None, creates a new one.
            target_network: Target Q-Network for stable learning. If None, creates a copy of q_network.
            world_model: World model for environment dynamics prediction. If None, creates a new one.
            training_strategy: Strategy for training ("cosine", "llm", or "hybrid").
            device: Device to run the models on.
            target_applicant_id: ID of the applicant this agent is specialized for.
            **kwargs: Additional parameters to override config values.
        """
        # Set device
        self.device = device or TRAINING_CONFIG["device"]
        
        # Set training strategy
        self.training_strategy = training_strategy
        logger.info(f"Initializing agent with training strategy: {training_strategy}")
        
        # Set target applicant ID
        self.target_applicant_id = target_applicant_id
        if target_applicant_id:
            logger.info(f"Creating agent specialized for applicant: {target_applicant_id}")
        
        # Get config parameters (overridden by kwargs if provided)
        self.gamma = kwargs.get("gamma", TRAINING_CONFIG["gamma"])
        self.lr = kwargs.get("lr", TRAINING_CONFIG["lr"])
        self.batch_size = kwargs.get("batch_size", TRAINING_CONFIG["batch_size"])
        self.target_update_freq = kwargs.get("target_update_freq", TRAINING_CONFIG["target_update_freq"])
        self.planning_steps = kwargs.get("planning_steps", TRAINING_CONFIG["planning_steps"])
        
        # Exploration parameters
        self.epsilon_start = kwargs.get("epsilon_start", TRAINING_CONFIG["epsilon_start"])
        self.epsilon_end = kwargs.get("epsilon_end", TRAINING_CONFIG["epsilon_end"])
        self.epsilon_decay = kwargs.get("epsilon_decay", TRAINING_CONFIG["epsilon_decay"])
        self.epsilon = self.epsilon_start
        
        # Hybrid strategy parameters
        self.cosine_weight = kwargs.get("cosine_weight", 1.0)  # Start with full cosine weight if hybrid
        self.cosine_annealing = kwargs.get("cosine_annealing", False)  # Whether to gradually reduce cosine weight
        self.cosine_anneal_steps = kwargs.get("cosine_anneal_steps", 1000)  # Steps to anneal cosine weight
        self.cosine_anneal_end = kwargs.get("cosine_anneal_end", 0.0)  # Final cosine weight after annealing
        
        # Create or use provided Q-Network
        self.q_network = q_network if q_network is not None else QNetwork(
            state_dim=state_dim,
            action_dim=action_dim
        ).to(self.device)
        
        # Create or use provided target network
        if target_network is not None:
            self.target_network = target_network
        else:
            self.target_network = QNetwork(
                state_dim=state_dim,
                action_dim=action_dim
            ).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Create or use provided world model
        input_dim = state_dim + action_dim
        self.world_model = world_model if world_model is not None else WorldModel(
            input_dim=input_dim
        ).to(self.device)
        
        # Create optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.world_optimizer = optim.Adam(self.world_model.parameters(), lr=self.lr)
        
        # Loss functions
        self.q_criterion = nn.MSELoss()
        self.world_criterion = nn.MSELoss()
        
        # Experience replay buffer for real experiences
        self.replay_buffer = ReplayBuffer(TRAINING_CONFIG["replay_buffer_size"])
        
        # Training metrics
        self.steps_done = 0
        
        logger.info(f"Initialized DynaQAgent with state_dim={state_dim}, action_dim={action_dim}")
    
    def select_action(self, state: torch.Tensor, available_actions: List[torch.Tensor],
                      eval_mode: bool = False) -> Tuple[int, torch.Tensor]:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state tensor.
            available_actions: List of available action tensors.
            eval_mode: If True, use greedy policy (epsilon=0).
            
        Returns:
            Tuple containing:
                - int: Index of the selected action
                - torch.Tensor: Selected action tensor
        """
        # Move state to device
        state = state.to(self.device)
        
        # Use epsilon-greedy policy
        epsilon = 0.0 if eval_mode else self.epsilon
        
        if random.random() > epsilon:
            # Exploit: Select action with highest Q-value
            with torch.no_grad():
                q_values = []
                for action in available_actions:
                    action = action.to(self.device)
                    q_value = self.q_network(state.unsqueeze(0), action.unsqueeze(0))
                    q_values.append(q_value.item())
                
                # Get action with maximum Q-value
                max_idx = np.argmax(q_values)
                selected_action = available_actions[max_idx]
                
                return max_idx, selected_action
        else:
            # Explore: Select random action
            idx = random.randrange(len(available_actions))
            return idx, available_actions[idx]
    
    def update_epsilon(self) -> None:
        """
        Update exploration rate (epsilon) using decay schedule.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def update_cosine_weight(self) -> None:
        """
        Update cosine weight for hybrid training strategy.
        Gradually shifts from cosine similarity to LLM feedback during training.
        """
        if not self.cosine_annealing or self.training_strategy != "hybrid":
            return
            
        # Linear annealing of cosine weight from initial value to end value
        if self.steps_done < self.cosine_anneal_steps:
            progress = self.steps_done / self.cosine_anneal_steps
            self.cosine_weight = self.cosine_weight - (progress * (self.cosine_weight - self.cosine_anneal_end))
            logger.debug(f"Updated cosine_weight to {self.cosine_weight}")
    
    def store_experience(self, state: torch.Tensor, action: torch.Tensor, 
                          reward: float, next_state: torch.Tensor) -> None:
        """
        Store a transition in the replay buffer.
        
        Args:
            state: Current state tensor.
            action: Action tensor.
            reward: Reward received.
            next_state: Next state tensor.
        """
        self.replay_buffer.push(state, action, reward, next_state)
    
    def update_q_network(self, states: torch.Tensor, actions: torch.Tensor,
                          rewards: torch.Tensor, next_states: torch.Tensor) -> float:
        """
        Update the Q-network using a batch of experiences.
        
        Args:
            states: Batch of state tensors.
            actions: Batch of action tensors.
            rewards: Batch of reward values.
            next_states: Batch of next state tensors.
            
        Returns:
            float: Loss value.
        """
        # Move inputs to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states, actions)
        
        # Compute next Q values
        next_q_values = torch.zeros(rewards.size(), device=self.device)
        
        # This is simplified - in practice, you'd need to evaluate Q for multiple candidate actions
        # and take the max. For this simplified version, we're assuming the environment provides
        # a limited set of actions, and the next state has the same action space as the current state.
        with torch.no_grad():
            next_q_values = rewards + self.gamma * self.target_network(next_states, actions)
        
        # Compute loss
        loss = self.q_criterion(current_q_values, next_q_values)
        
        # Update Q-network
        self.q_optimizer.zero_grad()
        loss.backward()
        # Optional: gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.q_optimizer.step()
        
        return loss.item()
    
    def update_world_model(self, states: torch.Tensor, actions: torch.Tensor, 
                            rewards: torch.Tensor) -> float:
        """
        Update the world model using a batch of experiences.
        
        Args:
            states: Batch of state tensors.
            actions: Batch of action tensors.
            rewards: Batch of reward values.
            
        Returns:
            float: Loss value.
        """
        # Move inputs to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        
        # Predict rewards
        predicted_rewards = self.world_model(states, actions)
        
        # Compute loss
        loss = self.world_criterion(predicted_rewards, rewards)
        
        # Update world model
        self.world_optimizer.zero_grad()
        loss.backward()
        self.world_optimizer.step()
        
        return loss.item()
    
    def update_target_network(self) -> None:
        """
        Update the target network by copying weights from the main Q-network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
        logger.debug("Updated target network")
    
    def plan(self, env: JobRecommendationEnv, planning_steps: Optional[int] = None) -> Dict[str, float]:
        """
        Perform planning steps using the world model.
        
        Args:
            env: Environment to get valid actions from.
            planning_steps: Number of planning steps to perform. If None, uses self.planning_steps.
            
        Returns:
            Dict[str, float]: Dictionary of metrics.
        """
        planning_steps = planning_steps or self.planning_steps
        
        # Skip planning if no planning steps or buffer too small
        if planning_steps <= 0 or len(self.replay_buffer) < self.batch_size:
            return {'planning_loss': 0.0}
        
        total_loss = 0.0
        
        # Plan for multiple steps
        for _ in range(planning_steps):
            # Sample states and actions from replay buffer for planning
            # For simplicity, we'll use a batch from the buffer
            states, actions, _, _ = self.replay_buffer.sample(self.batch_size)
            
            # Predict rewards using world model
            states = states.to(self.device)
            actions = actions.to(self.device)
            predicted_rewards = self.world_model(states, actions)
            
            # Compute targets for Q-learning
            with torch.no_grad():
                # Next states are the same as current states in this environment
                # (static state transitions)
                next_states = states
                
                # Use target network to estimate future value
                next_q_values = self.target_network(next_states, actions)
                
                # Compute targets using predicted rewards
                targets = predicted_rewards + self.gamma * next_q_values
            
            # Compute current Q-values
            current_q_values = self.q_network(states, actions)
            
            # Compute loss
            loss = self.q_criterion(current_q_values, targets)
            
            # Update Q-network
            self.q_optimizer.zero_grad()
            loss.backward()
            self.q_optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / planning_steps if planning_steps > 0 else 0.0
        return {'planning_loss': avg_loss}
    
    def train_step(self, env: JobRecommendationEnv, applicant_id: str, 
                   max_steps: int = 10) -> Dict[str, float]:
        """
        Train the agent for a single episode.
        
        Args:
            env: Environment to train in.
            applicant_id: ID of the applicant for this episode.
            max_steps: Maximum number of steps per episode.
            
        Returns:
            Dict[str, float]: Dictionary of metrics.
        """
        metrics = {
            'episode_reward': 0.0,
            'q_network_loss': 0.0,
            'world_model_loss': 0.0,
            'planning_loss': 0.0
        }
        
        # Reset environment
        state = env.reset(applicant_id)
        
        # Configure environment for the current training strategy
        if self.training_strategy == "hybrid" and isinstance(env, HybridEnv):
            env.set_cosine_weight(self.cosine_weight)
        
        # Training loop
        for step in range(max_steps):
            # Get available actions
            valid_action_indices = env.get_valid_actions()
            available_actions = [env.get_action_vector(idx) for idx in valid_action_indices]
            
            # Select action
            action_idx, action_vector = self.select_action(state, available_actions)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action_idx)
            
            # Log info
            logger.debug(f"Step {step}: Action={action_idx}, Reward={reward:.4f}")
            
            # Store experience in replay buffer
            self.store_experience(state, action_vector, reward, next_state)
            
            # Update metrics
            metrics['episode_reward'] += reward
            
            # Update networks if enough samples
            if len(self.replay_buffer) >= TRAINING_CONFIG["min_replay_buffer_size"]:
                # Sample batch from replay buffer
                states_batch, actions_batch, rewards_batch, next_states_batch = self.replay_buffer.sample(self.batch_size)
                
                # Update Q-network
                q_loss = self.update_q_network(states_batch, actions_batch, rewards_batch, next_states_batch)
                metrics['q_network_loss'] += q_loss
                
                # Update world model
                world_loss = self.update_world_model(states_batch, actions_batch, rewards_batch)
                metrics['world_model_loss'] += world_loss
                
                # Planning with world model
                planning_metrics = self.plan(env)
                metrics['planning_loss'] += planning_metrics['planning_loss']
                
                # Update target network periodically
                self.steps_done += 1
                if self.steps_done % self.target_update_freq == 0:
                    self.update_target_network()
                
                # Update exploration rate
                self.update_epsilon()
                
                # Update cosine weight for hybrid strategy
                self.update_cosine_weight()
            
            # Update state
            state = next_state
            
            # End episode if done
            if done:
                break
        
        # Compute averages
        num_steps = max_steps if step == max_steps - 1 else step + 1
        if num_steps > 0:
            metrics['q_network_loss'] /= num_steps
            metrics['world_model_loss'] /= num_steps
            metrics['planning_loss'] /= num_steps
        
        return metrics
    
    def train(self, env: JobRecommendationEnv, num_episodes: int, max_steps_per_episode: int,
              applicant_ids: List[str], eval_frequency: int = 100, save_frequency: int = 1000,
              model_dir: Optional[str] = None, switch_to_llm_episode: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Train the agent for multiple episodes on a single applicant.
        
        Following the Dyna-Q training philosophy, this method trains a specialized model
        for a single target applicant. The applicant's profile information remains constant
        throughout the training run, focusing the learning entirely on one chosen applicant.
        
        Args:
            env: Environment to train in.
            num_episodes: Number of episodes to train for.
            max_steps_per_episode: Maximum number of steps per episode.
            applicant_ids: List of applicant IDs. Only the first ID will be used for training
                          unless it's explicitly a comparative experiment.
            eval_frequency: Frequency of evaluation (episodes).
            save_frequency: Frequency of model saving (episodes).
            model_dir: Directory to save models to.
            switch_to_llm_episode: Episode number to switch from cosine to LLM rewards (for hybrid training).
            
        Returns:
            Dict[str, List[float]]: Dictionary of metrics over episodes.
        """
        # Initialize metrics tracking
        history = {
            'episode_reward': [],
            'q_network_loss': [],
            'world_model_loss': [],
            'planning_loss': [],
            'eval_reward': []
        }
        
        # Ensure model directory exists
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        
        # Select the target applicant - use the first in the list
        # This follows the single-applicant philosophy from the documentation
        target_applicant_id = applicant_ids[0]
        logger.info(f"Training specialized agent for applicant: {target_applicant_id}")
        
        # Save the applicant information in the model metadata
        self.target_applicant_id = target_applicant_id
        
        # Training loop - all episodes use the same applicant
        for episode in range(num_episodes):
            # Handle strategy switching for hybrid approach
            if self.training_strategy == "hybrid" and switch_to_llm_episode is not None:
                if episode == switch_to_llm_episode:
                    logger.info(f"Episode {episode}: Switching from cosine similarity to LLM feedback")
                    # Override cosine weight to use only LLM feedback
                    self.cosine_weight = 0.0
                    if isinstance(env, HybridEnv):
                        env.set_cosine_weight(0.0)
            
            # Train for one episode using the target applicant
            episode_metrics = self.train_step(env, target_applicant_id, max_steps_per_episode)
            
            # Update history
            for key, value in episode_metrics.items():
                if key in history:
                    history[key].append(value)
            
            # Log progress
            logger.info(f"Episode {episode}: Reward={episode_metrics['episode_reward']:.4f}, "
                       f"Q-Loss={episode_metrics['q_network_loss']:.4f}, "
                       f"World-Loss={episode_metrics['world_model_loss']:.4f}")
            
            # Evaluate periodically
            if eval_frequency > 0 and episode % eval_frequency == 0:
                # Evaluate using the same target applicant
                eval_reward = self.evaluate(env, 5, max_steps_per_episode, [target_applicant_id])
                history['eval_reward'].append(eval_reward)
                logger.info(f"Evaluation at episode {episode}: Avg Reward={eval_reward:.4f}")
            
            # Save models periodically
            if model_dir and save_frequency > 0 and episode > 0 and episode % save_frequency == 0:
                self.save_models(model_dir, episode)
                logger.info(f"Saved models at episode {episode}")
        
        # Final evaluation
        final_eval_reward = self.evaluate(env, 10, max_steps_per_episode, [target_applicant_id])
        history['eval_reward'].append(final_eval_reward)
        logger.info(f"Final evaluation: Avg Reward={final_eval_reward:.4f}")
        
        # Save final models
        if model_dir:
            self.save_models(model_dir, num_episodes)
            logger.info(f"Saved final models at episode {num_episodes}")
        
        return history

    def evaluate(self, env: JobRecommendationEnv, num_episodes: int, 
                  max_steps_per_episode: int, applicant_ids: List[str]) -> float:
        """
        Evaluate the agent's performance for a specific applicant.
        
        Args:
            env: Environment to evaluate in.
            num_episodes: Number of episodes to evaluate for.
            max_steps_per_episode: Maximum number of steps per episode.
            applicant_ids: List of applicant IDs. Should contain the target applicant this agent was trained for.
                           If multiple IDs are provided and no target_applicant_id is set, the first ID will be used.
            
        Returns:
            float: Average reward per episode.
        """
        total_reward = 0.0
        
        # Determine which applicant to evaluate on
        # If agent has a target_applicant_id, use that
        # Otherwise, use the first in the provided list
        applicant_id = self.target_applicant_id if hasattr(self, "target_applicant_id") and self.target_applicant_id else applicant_ids[0]
        logger.info(f"Evaluating agent on applicant: {applicant_id}")
        
        # Evaluation loop for the specific applicant
        for episode in range(num_episodes):
            # Reset environment with the specific applicant
            state = env.reset(applicant_id)
            episode_reward = 0.0
            
            # Episode loop
            for step in range(max_steps_per_episode):
                # Get available actions
                valid_action_indices = env.get_valid_actions()
                available_actions = [env.get_action_vector(idx) for idx in valid_action_indices]
                
                # Select action (greedy policy for evaluation)
                action_idx, _ = self.select_action(state, available_actions, eval_mode=True)
                
                # Take action in environment
                next_state, reward, done, _ = env.step(action_idx)
                
                # Update reward
                episode_reward += reward
                
                # Update state
                state = next_state
                
                # End episode if done
                if done:
                    break
            
            # Update total reward
            total_reward += episode_reward
            logger.debug(f"Evaluation episode {episode} reward: {episode_reward:.4f}")
        
        # Compute average reward
        avg_reward = total_reward / num_episodes if num_episodes > 0 else 0.0
        
        return avg_reward
    
    def save_models(self, model_dir: str, episode: int) -> None:
        """
        Save the models.
        
        Args:
            model_dir: Directory to save the models to.
            episode: Current episode number.
        """
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save Q-network
        q_network_path = os.path.join(model_dir, f"q_network_{episode}.pt")
        self.q_network.save(q_network_path)
        
        # Save world model
        world_model_path = os.path.join(model_dir, f"world_model_{episode}.pt")
        self.world_model.save(world_model_path)
        
        # Save agent config
        config_path = os.path.join(model_dir, f"agent_config_{episode}.json")
        import json
        with open(config_path, 'w') as f:
            json.dump({
                "training_strategy": self.training_strategy,
                "state_dim": self.q_network.state_dim,
                "action_dim": self.q_network.action_dim,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_end": self.epsilon_end,
                "cosine_weight": self.cosine_weight if hasattr(self, "cosine_weight") else None,
                "steps_done": self.steps_done,
                "episode": episode,
                "target_applicant_id": getattr(self, "target_applicant_id", None)
            }, f)
    
    @classmethod
    def load_models(cls, model_dir: str, episode: Optional[int] = None, 
                    device: Optional[torch.device] = None) -> 'DynaQAgent':
        """
        Load models from files.
        
        Args:
            model_dir: Directory to load the models from.
            episode: Episode number to load. If None, loads the latest.
            device: Device to load the models on.
            
        Returns:
            DynaQAgent: Loaded agent.
        """
        # Find latest episode if not specified
        if episode is None:
            import glob
            config_files = glob.glob(os.path.join(model_dir, "agent_config_*.json"))
            if not config_files:
                raise ValueError(f"No agent config files found in {model_dir}")
            
            # Extract episode numbers
            episodes = [int(f.split("_")[-1].split(".")[0]) for f in config_files]
            episode = max(episodes)
            logger.info(f"Loading latest models from episode {episode}")
        
        # Load agent config
        import json
        config_path = os.path.join(model_dir, f"agent_config_{episode}.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load models
        q_network_path = os.path.join(model_dir, f"q_network_{episode}.pt")
        q_network = QNetwork.load(q_network_path, device)
        
        world_model_path = os.path.join(model_dir, f"world_model_{episode}.pt")
        world_model = WorldModel.load(world_model_path, device)
        
        # Create target network
        target_network = QNetwork(
            state_dim=config["state_dim"],
            action_dim=config["action_dim"]
        ).to(device if device else torch.device('cpu'))
        target_network.load_state_dict(q_network.state_dict())
        
        # Create agent
        agent = cls(
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            q_network=q_network,
            target_network=target_network,
            world_model=world_model,
            training_strategy=config.get("training_strategy", "cosine"),
            device=device,
            gamma=config.get("gamma", TRAINING_CONFIG["gamma"]),
            epsilon=config.get("epsilon", TRAINING_CONFIG["epsilon_end"]),
            target_applicant_id=config.get("target_applicant_id", None)
        )
        
        # Set additional parameters
        agent.steps_done = config.get("steps_done", 0)
        if "cosine_weight" in config and config["cosine_weight"] is not None:
            agent.cosine_weight = config["cosine_weight"]
        
        # Set target applicant ID if available in config
        if "target_applicant_id" in config and config["target_applicant_id"] is not None:
            agent.target_applicant_id = config["target_applicant_id"]
            logger.info(f"Model was trained for applicant: {agent.target_applicant_id}")
        
        return agent
    
    def switch_training_strategy(self, new_strategy: str) -> None:
        """
        Switch the training strategy during training.
        
        Args:
            new_strategy: New training strategy ("cosine", "llm", or "hybrid").
        """
        if new_strategy not in ["cosine", "llm", "hybrid"]:
            raise ValueError(f"Invalid training strategy: {new_strategy}")
        
        self.training_strategy = new_strategy
        logger.info(f"Switched training strategy to {new_strategy}")
        
        # Reset cosine weight if switching to hybrid
        if new_strategy == "hybrid":
            self.cosine_weight = 1.0  # Start with full cosine weight
        
        # Could add additional logic here for more complex strategy switching 