"""
Evaluator utility for assessing model performance.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
import json
from collections import defaultdict

# Import configuration and modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config.config import EVAL_CONFIG, PATH_CONFIG

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from training.agent import DynaQAgent
from environments.job_env import JobRecommendationEnv
from utils.visualizer import Visualizer

logger = logging.getLogger(__name__)

class Evaluator:
    """
    Evaluator for assessing the performance of Dyna-Q agents.
    
    Compares baseline and pretrained agents on various metrics.
    """
    
    def __init__(self, results_dir: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            results_dir: Directory to save evaluation results.
        """
        self.results_dir = results_dir or os.path.join(os.path.dirname(__file__), 
                                                     PATH_CONFIG["results_dir"])
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize visualizer
        self.visualizer = Visualizer(results_dir=self.results_dir)
        
        logger.info(f"Initialized evaluator with results directory: {self.results_dir}")
    
    def evaluate_agent(self, agent: DynaQAgent, env: JobRecommendationEnv, 
                       applicant_ids: List[str], num_episodes: int = EVAL_CONFIG["num_eval_episodes"],
                       max_steps_per_episode: int = 100) -> Dict[str, Any]:
        """
        Evaluate a single agent.
        
        Args:
            agent: Agent to evaluate.
            env: Environment to evaluate in.
            applicant_ids: List of applicant IDs to evaluate on.
            num_episodes: Number of episodes to evaluate for.
            max_steps_per_episode: Maximum number of steps per episode.
            
        Returns:
            Dict[str, Any]: Evaluation results.
        """
        # Metrics to track
        episode_rewards = []
        episode_steps = []
        response_counts = defaultdict(int)
        
        logger.info(f"Evaluating agent for {num_episodes} episodes")
        
        # Evaluation loop
        for episode in range(num_episodes):
            # Select an applicant
            applicant_id = np.random.choice(applicant_ids)
            
            # Reset environment
            state = env.reset(applicant_id)
            episode_reward = 0
            step_count = 0
            
            # Episode loop
            for step in range(max_steps_per_episode):
                # Get available actions
                valid_action_indices = env.get_valid_actions()
                available_actions = [env.get_action_vector(idx) for idx in valid_action_indices]
                
                # Select action (greedy policy)
                action_idx, _ = agent.select_action(state, available_actions, eval_mode=True)
                
                # Take step
                next_state, reward, done, info = env.step(action_idx)
                
                # Update state and metrics
                state = next_state
                episode_reward += reward
                step_count += 1
                
                # Track response
                response = info.get("response", "")
                if response:
                    response_counts[response] += 1
                
                # Break if episode is done
                if done:
                    break
            
            # Record episode metrics
            episode_rewards.append(episode_reward)
            episode_steps.append(step_count)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                logger.info(f"Evaluated {episode+1}/{num_episodes} episodes. "
                           f"Avg reward: {np.mean(episode_rewards):.4f}")
        
        # Calculate metrics
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_steps = np.mean(episode_steps)
        
        # Calculate apply rate
        total_steps = sum(episode_steps)
        apply_rate = response_counts.get("APPLY", 0) / total_steps if total_steps > 0 else 0
        
        # Format response counts
        response_percentages = {resp: count / total_steps * 100 for resp, count in response_counts.items()}
        
        # Compile results
        results = {
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "avg_steps": avg_steps,
            "apply_rate": apply_rate,
            "response_counts": dict(response_counts),
            "response_percentages": response_percentages,
            "episode_rewards": episode_rewards
        }
        
        logger.info(f"Evaluation complete. Average reward: {avg_reward:.4f}")
        return results
    
    def compare_agents(self, baseline_agent: DynaQAgent, pretrained_agent: DynaQAgent,
                       env: JobRecommendationEnv, applicant_ids: List[str],
                       num_episodes: int = EVAL_CONFIG["num_eval_episodes"],
                       max_steps_per_episode: int = 100) -> Dict[str, Any]:
        """
        Compare a baseline and pretrained agent.
        
        Args:
            baseline_agent: Baseline agent.
            pretrained_agent: Pretrained agent.
            env: Environment to evaluate in.
            applicant_ids: List of applicant IDs to evaluate on.
            num_episodes: Number of episodes to evaluate for.
            max_steps_per_episode: Maximum number of steps per episode.
            
        Returns:
            Dict[str, Any]: Comparison results.
        """
        # Evaluate both agents
        logger.info("Evaluating baseline agent...")
        baseline_results = self.evaluate_agent(
            baseline_agent, env, applicant_ids, num_episodes, max_steps_per_episode
        )
        
        logger.info("Evaluating pretrained agent...")
        pretrained_results = self.evaluate_agent(
            pretrained_agent, env, applicant_ids, num_episodes, max_steps_per_episode
        )
        
        # Calculate improvement percentages
        avg_reward_improvement = (
            (pretrained_results["avg_reward"] - baseline_results["avg_reward"]) / 
            abs(baseline_results["avg_reward"]) * 100 if baseline_results["avg_reward"] != 0 else float('inf')
        )
        
        apply_rate_improvement = (
            (pretrained_results["apply_rate"] - baseline_results["apply_rate"]) / 
            baseline_results["apply_rate"] * 100 if baseline_results["apply_rate"] > 0 else float('inf')
        )
        
        # Compile comparison results
        comparison = {
            "baseline": baseline_results,
            "pretrained": pretrained_results,
            "improvements": {
                "avg_reward": avg_reward_improvement,
                "apply_rate": apply_rate_improvement
            }
        }
        
        # Create visualizations
        self.visualize_comparison(comparison)
        
        # Save results to file
        self.save_results(comparison, "agent_comparison.json")
        
        logger.info(f"Agent comparison complete. Reward improvement: {avg_reward_improvement:.2f}%")
        return comparison
    
    def evaluate_performance_over_time(self, agent: DynaQAgent, env: JobRecommendationEnv,
                                       applicant_ids: List[str], 
                                       num_episodes: int = EVAL_CONFIG["num_eval_episodes"],
                                       max_steps_per_episode: int = 100,
                                       eval_interval: int = 10) -> Dict[str, List[float]]:
        """
        Evaluate agent performance over time (throughout training).
        
        Args:
            agent: Agent to evaluate.
            env: Environment to evaluate in.
            applicant_ids: List of applicant IDs to evaluate on.
            num_episodes: Total number of episodes.
            max_steps_per_episode: Maximum number of steps per episode.
            eval_interval: Interval between evaluations.
            
        Returns:
            Dict[str, List[float]]: Performance metrics over time.
        """
        # Metrics to track
        rewards_over_time = []
        apply_rates_over_time = []
        steps = []
        
        logger.info(f"Evaluating performance over {num_episodes} episodes")
        
        # Evaluation loop
        for episode in range(0, num_episodes, eval_interval):
            # Run a mini-evaluation
            eval_results = self.evaluate_agent(
                agent, env, applicant_ids, 
                num_episodes=max(5, eval_interval // 2),  # Smaller number of episodes for efficiency
                max_steps_per_episode=max_steps_per_episode
            )
            
            # Record metrics
            rewards_over_time.append(eval_results["avg_reward"])
            apply_rates_over_time.append(eval_results["apply_rate"])
            steps.append(episode)
            
            logger.info(f"Episode {episode}: Avg reward = {eval_results['avg_reward']:.4f}, "
                       f"Apply rate = {eval_results['apply_rate']:.4f}")
        
        # Compile results
        results = {
            "steps": steps,
            "rewards": rewards_over_time,
            "apply_rates": apply_rates_over_time
        }
        
        # Save results
        self.save_results(results, "performance_over_time.json")
        
        # Create visualization
        self.visualizer.plot_learning_curves(
            {"Agent": {"episode_rewards": rewards_over_time}},
            title="Performance Over Time",
            filename="performance_over_time.png"
        )
        
        logger.info("Performance over time evaluation complete")
        return results
    
    def visualize_comparison(self, comparison: Dict[str, Any]) -> None:
        """
        Visualize comparison between baseline and pretrained agents.
        
        Args:
            comparison: Comparison results from compare_agents.
        """
        # Plot reward comparison
        self.visualizer.plot_evaluation_results(
            comparison["baseline"]["episode_rewards"],
            comparison["pretrained"]["episode_rewards"],
            title="Baseline vs. Pretrained Agent Rewards",
            filename="reward_comparison.png"
        )
        
        # Plot reward distribution
        self.visualizer.plot_reward_histogram(
            {
                "Baseline": comparison["baseline"]["episode_rewards"],
                "Pretrained": comparison["pretrained"]["episode_rewards"]
            },
            title="Reward Distribution Comparison",
            filename="reward_distribution.png"
        )
        
        # Plot cumulative rewards
        self.visualizer.plot_cumulative_rewards(
            {
                "Baseline": comparison["baseline"]["episode_rewards"],
                "Pretrained": comparison["pretrained"]["episode_rewards"]
            },
            title="Cumulative Rewards Comparison",
            filename="cumulative_rewards.png"
        )
        
        logger.info("Comparison visualizations created")
    
    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """
        Save evaluation results to file.
        
        Args:
            results: Results to save.
            filename: Filename to save to.
        """
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        # Save to file
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """
        Load evaluation results from file.
        
        Args:
            filename: Filename to load from.
            
        Returns:
            Dict[str, Any]: Loaded results.
        """
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Results loaded from {filepath}")
        return results 