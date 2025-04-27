"""
Visualizer for plotting training metrics and evaluation results.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os
import logging
import torch

# Import configuration
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config.config import PATH_CONFIG

logger = logging.getLogger(__name__)

class Visualizer:
    """
    Visualizer for plotting training metrics and evaluation results.
    """
    
    def __init__(self, results_dir: Optional[str] = None):
        """
        Initialize the visualizer.
        
        Args:
            results_dir: Directory to save visualizations.
        """
        self.results_dir = results_dir or os.path.join(os.path.dirname(__file__), 
                                                     PATH_CONFIG["results_dir"])
        os.makedirs(self.results_dir, exist_ok=True)
        logger.info(f"Initialized visualizer with results directory: {self.results_dir}")
    
    def plot_training_metrics(self, metrics: Dict[str, List[float]], 
                               title: str = "Training Metrics", 
                               filename: str = "training_metrics.png") -> None:
        """
        Plot training metrics.
        
        Args:
            metrics: Dictionary of metric names and values.
            title: Plot title.
            filename: Filename to save the plot.
        """
        # Create figure with subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics), sharex=True)
        
        # If only one metric, axes is not a list
        if n_metrics == 1:
            axes = [axes]
        
        # Plot each metric
        for i, (metric_name, values) in enumerate(metrics.items()):
            # Skip empty metrics
            if not values:
                continue
                
            axes[i].plot(values)
            axes[i].set_title(f"{metric_name.replace('_', ' ').title()}")
            axes[i].grid(True)
            
            # Add smoothed line for noisy metrics
            if len(values) > 10 and metric_name in ["episode_rewards", "q_losses", "world_losses"]:
                smoothed = np.convolve(values, np.ones(10)/10, mode='valid')
                axes[i].plot(range(9, len(values)), smoothed, 'r-', linewidth=2)
                axes[i].legend(['Raw', 'Smoothed (Moving Avg)'])
            
        # Set x-label for the bottom subplot
        axes[-1].set_xlabel("Episodes")
        
        # Adjust layout
        plt.tight_layout()
        fig.suptitle(title, fontsize=16, y=1.02)
        
        # Save figure
        plt.savefig(os.path.join(self.results_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training metrics plot to {filename}")
    
    def plot_evaluation_results(self, baseline_rewards: List[float], 
                                 pretrained_rewards: List[float],
                                 x_values: Optional[List[int]] = None,
                                 title: str = "Baseline vs. Pretrained Agent Performance",
                                 filename: str = "evaluation_comparison.png") -> None:
        """
        Plot comparison between baseline and pretrained agent.
        
        Args:
            baseline_rewards: Rewards for the baseline agent.
            pretrained_rewards: Rewards for the pretrained agent.
            x_values: X-axis values (e.g., episode numbers or steps).
            title: Plot title.
            filename: Filename to save the plot.
        """
        # Create x-values if not provided
        if x_values is None:
            x_values = list(range(1, min(len(baseline_rewards), len(pretrained_rewards)) + 1))
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot rewards
        plt.plot(x_values, baseline_rewards[:len(x_values)], 'b-', linewidth=2, label='Baseline Agent')
        plt.plot(x_values, pretrained_rewards[:len(x_values)], 'r-', linewidth=2, label='Pretrained Agent (LLM)')
        
        # Add annotations
        plt.title(title, fontsize=14)
        plt.xlabel("Episodes", fontsize=12)
        plt.ylabel("Average Reward", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Save figure
        plt.savefig(os.path.join(self.results_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved evaluation comparison plot to {filename}")
    
    def plot_cumulative_rewards(self, rewards_dict: Dict[str, List[float]], 
                                 title: str = "Cumulative Rewards",
                                 filename: str = "cumulative_rewards.png") -> None:
        """
        Plot cumulative rewards for different agents.
        
        Args:
            rewards_dict: Dictionary mapping agent names to reward lists.
            title: Plot title.
            filename: Filename to save the plot.
        """
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot cumulative rewards for each agent
        for agent_name, rewards in rewards_dict.items():
            cumulative_rewards = np.cumsum(rewards)
            plt.plot(cumulative_rewards, linewidth=2, label=agent_name)
        
        # Add annotations
        plt.title(title, fontsize=14)
        plt.xlabel("Episodes", fontsize=12)
        plt.ylabel("Cumulative Reward", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Save figure
        plt.savefig(os.path.join(self.results_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved cumulative rewards plot to {filename}")
    
    def plot_reward_histogram(self, rewards_dict: Dict[str, List[float]], 
                               bins: int = 20,
                               title: str = "Reward Distribution",
                               filename: str = "reward_histogram.png") -> None:
        """
        Plot histogram of rewards for different agents.
        
        Args:
            rewards_dict: Dictionary mapping agent names to reward lists.
            bins: Number of histogram bins.
            title: Plot title.
            filename: Filename to save the plot.
        """
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot histogram for each agent
        for agent_name, rewards in rewards_dict.items():
            plt.hist(rewards, bins=bins, alpha=0.5, label=agent_name)
        
        # Add annotations
        plt.title(title, fontsize=14)
        plt.xlabel("Reward", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Save figure
        plt.savefig(os.path.join(self.results_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved reward histogram plot to {filename}")
    
    def visualize_q_values(self, q_network: torch.nn.Module, 
                            sample_states: List[torch.Tensor],
                            sample_actions: List[torch.Tensor],
                            state_labels: Optional[List[str]] = None,
                            action_labels: Optional[List[str]] = None,
                            title: str = "Q-Value Heatmap",
                            filename: str = "q_value_heatmap.png") -> None:
        """
        Visualize Q-values for sample states and actions.
        
        Args:
            q_network: Q-Network model.
            sample_states: List of state tensors.
            sample_actions: List of action tensors.
            state_labels: Labels for states.
            action_labels: Labels for actions.
            title: Plot title.
            filename: Filename to save the plot.
        """
        if not state_labels:
            state_labels = [f"State {i}" for i in range(len(sample_states))]
        
        if not action_labels:
            action_labels = [f"Action {i}" for i in range(len(sample_actions))]
        
        # Compute Q-values for all state-action pairs
        q_values = np.zeros((len(sample_states), len(sample_actions)))
        
        with torch.no_grad():
            for i, state in enumerate(sample_states):
                for j, action in enumerate(sample_actions):
                    q_values[i, j] = q_network(state.unsqueeze(0), action.unsqueeze(0)).item()
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot heatmap
        plt.imshow(q_values, cmap='viridis')
        plt.colorbar(label='Q-Value')
        
        # Add annotations
        plt.title(title, fontsize=14)
        plt.xlabel("Actions", fontsize=12)
        plt.ylabel("States", fontsize=12)
        plt.xticks(range(len(action_labels)), action_labels, rotation=45, ha="right")
        plt.yticks(range(len(state_labels)), state_labels)
        
        # Add text annotations with actual values
        for i in range(len(sample_states)):
            for j in range(len(sample_actions)):
                plt.text(j, i, f"{q_values[i, j]:.2f}", 
                        ha="center", va="center", 
                        color="white" if q_values[i, j] < np.mean(q_values) else "black")
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved Q-value heatmap to {filename}")
    
    def plot_learning_curves(self, data_dict: Dict[str, Dict[str, List[float]]],
                              metric: str = "episode_rewards",
                              title: str = "Learning Curves",
                              filename: str = "learning_curves.png") -> None:
        """
        Plot learning curves for different configurations.
        
        Args:
            data_dict: Nested dictionary mapping config names to metric dictionaries.
            metric: Name of the metric to plot.
            title: Plot title.
            filename: Filename to save the plot.
        """
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot learning curve for each configuration
        for config_name, metrics in data_dict.items():
            if metric in metrics:
                values = metrics[metric]
                plt.plot(values, linewidth=2, label=config_name)
                
                # Add smoothed line
                if len(values) > 10:
                    smoothed = np.convolve(values, np.ones(10)/10, mode='valid')
                    plt.plot(range(9, len(values)), smoothed, '--', linewidth=1)
        
        # Add annotations
        plt.title(title, fontsize=14)
        plt.xlabel("Episodes", fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Save figure
        plt.savefig(os.path.join(self.results_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved learning curves plot to {filename}") 