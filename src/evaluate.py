"""
Evaluation script for the Dyna-Q job recommender model.

This script handles evaluation of the trained models, comparing baseline
and pretrained agents to assess the impact of LLM-based pretraining.
"""

import os
import sys
import logging
import argparse
import torch
import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import MODEL_CONFIG, TRAINING_CONFIG, EVAL_CONFIG, PATH_CONFIG

# Import modules
from data.database import DatabaseConnector
from environments.job_env import JobRecommendationEnv, LLMSimulatorEnv
from training.agent import DynaQAgent
from utils.evaluator import Evaluator
from utils.visualizer import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Set deterministic backend for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_model_performance(agent: DynaQAgent, env: JobRecommendationEnv, 
                              applicant_ids: List[str], num_episodes: int = EVAL_CONFIG["num_eval_episodes"],
                              agent_name: str = "Agent") -> Dict[str, Any]:
    """
    Evaluate a single agent's performance.
    
    Args:
        agent: Agent to evaluate.
        env: Environment to evaluate in.
        applicant_ids: List of applicant IDs to evaluate on.
        num_episodes: Number of episodes to evaluate for.
        agent_name: Name of the agent for reporting.
        
    Returns:
        Dict[str, Any]: Evaluation results.
    """
    # Create evaluator
    evaluator = Evaluator()
    
    # Evaluate agent
    results = evaluator.evaluate_agent(
        agent=agent,
        env=env,
        applicant_ids=applicant_ids,
        num_episodes=num_episodes
    )
    
    # Log results
    logger.info(f"Evaluation results for {agent_name}:")
    logger.info(f"  Average reward: {results['avg_reward']:.4f}")
    logger.info(f"  Apply rate: {results['apply_rate']:.4f}")
    logger.info(f"  Response percentages: {results['response_percentages']}")
    
    return results

def compare_baseline_and_pretrained(baseline_agent: DynaQAgent, pretrained_agent: DynaQAgent, 
                                  env: JobRecommendationEnv, applicant_ids: List[str],
                                  num_episodes: int = EVAL_CONFIG["num_eval_episodes"]) -> Dict[str, Any]:
    """
    Compare baseline and pretrained agents.
    
    Args:
        baseline_agent: Baseline agent.
        pretrained_agent: Pretrained agent.
        env: Environment to evaluate in.
        applicant_ids: List of applicant IDs to evaluate on.
        num_episodes: Number of episodes to evaluate for.
        
    Returns:
        Dict[str, Any]: Comparison results.
    """
    # Create evaluator
    evaluator = Evaluator()
    
    # Compare agents
    comparison = evaluator.compare_agents(
        baseline_agent=baseline_agent,
        pretrained_agent=pretrained_agent,
        env=env,
        applicant_ids=applicant_ids,
        num_episodes=num_episodes
    )
    
    # Log results
    logger.info("Comparison Results:")
    logger.info(f"  Reward improvement: {comparison['improvements']['avg_reward']:.2f}%")
    logger.info(f"  Apply rate improvement: {comparison['improvements']['apply_rate']:.2f}%")
    
    return comparison

def evaluate_cold_start_performance(baseline_agent: DynaQAgent, pretrained_agent: DynaQAgent,
                                  env: JobRecommendationEnv, applicant_ids: List[str],
                                  num_episodes: int = 10, num_steps: int = 10) -> Dict[str, Any]:
    """
    Evaluate cold-start performance by comparing cumulative rewards in the initial steps.
    
    Args:
        baseline_agent: Baseline agent.
        pretrained_agent: Pretrained agent.
        env: Environment to evaluate in.
        applicant_ids: List of applicant IDs to evaluate on.
        num_episodes: Number of episodes to evaluate for.
        num_steps: Number of steps per episode.
        
    Returns:
        Dict[str, Any]: Cold-start evaluation results.
    """
    logger.info(f"Evaluating cold-start performance for {num_episodes} episodes, {num_steps} steps each")
    
    # Initialize reward tracking
    baseline_episode_rewards = []
    pretrained_episode_rewards = []
    
    # Evaluate on multiple episodes
    for episode in range(num_episodes):
        # Select a random applicant
        applicant_id = random.choice(applicant_ids)
        
        # Evaluate baseline agent
        env.reset(applicant_id)
        baseline_reward = 0
        for step in range(num_steps):
            valid_action_indices = env.get_valid_actions()
            available_actions = [env.get_action_vector(idx) for idx in valid_action_indices]
            action_idx, _ = baseline_agent.select_action(env.current_state, available_actions, eval_mode=True)
            _, reward, _, _ = env.step(action_idx)
            baseline_reward += reward
        baseline_episode_rewards.append(baseline_reward)
        
        # Evaluate pretrained agent (reuse same applicant for fair comparison)
        env.reset(applicant_id)
        pretrained_reward = 0
        for step in range(num_steps):
            valid_action_indices = env.get_valid_actions()
            available_actions = [env.get_action_vector(idx) for idx in valid_action_indices]
            action_idx, _ = pretrained_agent.select_action(env.current_state, available_actions, eval_mode=True)
            _, reward, _, _ = env.step(action_idx)
            pretrained_reward += reward
        pretrained_episode_rewards.append(pretrained_reward)
        
        logger.info(f"Episode {episode+1}: Baseline reward = {baseline_reward:.2f}, "
                   f"Pretrained reward = {pretrained_reward:.2f}")
    
    # Calculate metrics
    avg_baseline_reward = np.mean(baseline_episode_rewards)
    avg_pretrained_reward = np.mean(pretrained_episode_rewards)
    improvement = ((avg_pretrained_reward - avg_baseline_reward) / 
                   abs(avg_baseline_reward) * 100 if avg_baseline_reward != 0 else float('inf'))
    
    # Compile results
    results = {
        "baseline_episode_rewards": baseline_episode_rewards,
        "pretrained_episode_rewards": pretrained_episode_rewards,
        "avg_baseline_reward": avg_baseline_reward,
        "avg_pretrained_reward": avg_pretrained_reward,
        "improvement": improvement
    }
    
    # Create visualization
    visualizer = Visualizer()
    visualizer.plot_evaluation_results(
        baseline_rewards=baseline_episode_rewards,
        pretrained_rewards=pretrained_episode_rewards,
        title="Cold-Start Performance Comparison",
        filename="cold_start_comparison.png"
    )
    
    # Log results
    logger.info(f"Cold-start evaluation results:")
    logger.info(f"  Average baseline reward: {avg_baseline_reward:.4f}")
    logger.info(f"  Average pretrained reward: {avg_pretrained_reward:.4f}")
    logger.info(f"  Improvement: {improvement:.2f}%")
    
    return results

def evaluate_performance_over_time(agent: DynaQAgent, env: JobRecommendationEnv, 
                                 applicant_ids: List[str], steps: List[int]) -> Dict[str, List[float]]:
    """
    Evaluate agent performance over increasing amounts of experience.
    
    Args:
        agent: Agent to evaluate.
        env: Environment to evaluate in.
        applicant_ids: List of applicant IDs to evaluate on.
        steps: List of step numbers to evaluate at.
        
    Returns:
        Dict[str, List[float]]: Performance metrics over time.
    """
    logger.info(f"Evaluating performance over time at steps: {steps}")
    
    # Create evaluator
    evaluator = Evaluator()
    
    # Evaluate performance over time
    results = evaluator.evaluate_performance_over_time(
        agent=agent,
        env=env,
        applicant_ids=applicant_ids,
        num_episodes=max(steps),
        eval_interval=min(steps)
    )
    
    return results

def main(args: argparse.Namespace) -> None:
    """
    Main function for evaluating models.
    
    Args:
        args: Command-line arguments.
    """
    # Set random seed
    set_seed(args.seed)
    
    # Initialize database connector
    db_connector = DatabaseConnector()
    
    # Get sample applicant IDs (simplified - in practice, would fetch from DB)
    # In a real scenario, you would query your MongoDB for a list of applicant IDs
    # For now, we use placeholder IDs
    sample_applicant_ids = [f"applicant_{i}" for i in range(100)]
    
    # Create evaluation environment
    if args.use_llm_simulator:
        env = LLMSimulatorEnv(db_connector=db_connector, random_seed=args.seed)
    else:
        env = JobRecommendationEnv(db_connector=db_connector, random_seed=args.seed)
    
    # Load models
    device = torch.device(args.device)
    
    # Load baseline agent
    baseline_path = args.baseline_path or os.path.join(
        os.path.dirname(__file__), PATH_CONFIG["model_dir"], "baseline"
    )
    logger.info(f"Loading baseline agent from {baseline_path}")
    baseline_agent = DynaQAgent.load_models(
        model_dir=baseline_path,
        episode=args.baseline_episode,
        device=device
    )
    
    # Load pretrained agent
    pretrained_path = args.pretrained_path or os.path.join(
        os.path.dirname(__file__), PATH_CONFIG["model_dir"], "pretrained"
    )
    logger.info(f"Loading pretrained agent from {pretrained_path}")
    pretrained_agent = DynaQAgent.load_models(
        model_dir=pretrained_path,
        episode=args.pretrained_episode,
        device=device
    )
    
    # Run evaluations based on command-line arguments
    if args.evaluate_baseline:
        logger.info("Evaluating baseline agent")
        evaluate_model_performance(
            agent=baseline_agent,
            env=env,
            applicant_ids=sample_applicant_ids,
            num_episodes=args.num_episodes,
            agent_name="Baseline Agent"
        )
    
    if args.evaluate_pretrained:
        logger.info("Evaluating pretrained agent")
        evaluate_model_performance(
            agent=pretrained_agent,
            env=env,
            applicant_ids=sample_applicant_ids,
            num_episodes=args.num_episodes,
            agent_name="Pretrained Agent"
        )
    
    if args.compare:
        logger.info("Comparing baseline and pretrained agents")
        compare_baseline_and_pretrained(
            baseline_agent=baseline_agent,
            pretrained_agent=pretrained_agent,
            env=env,
            applicant_ids=sample_applicant_ids,
            num_episodes=args.num_episodes
        )
    
    if args.cold_start:
        logger.info("Evaluating cold-start performance")
        evaluate_cold_start_performance(
            baseline_agent=baseline_agent,
            pretrained_agent=pretrained_agent,
            env=env,
            applicant_ids=sample_applicant_ids,
            num_episodes=args.cold_start_episodes,
            num_steps=args.cold_start_steps
        )
    
    if args.performance_over_time:
        logger.info("Evaluating performance over time")
        evaluate_performance_over_time(
            agent=pretrained_agent,
            env=env,
            applicant_ids=sample_applicant_ids,
            steps=list(range(0, args.num_episodes, args.eval_interval))
        )
    
    # Close database connection
    db_connector.close()
    
    logger.info("Evaluation complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Dyna-Q job recommender models")
    
    # Model paths
    parser.add_argument("--baseline_path", type=str, help="Path to baseline model directory")
    parser.add_argument("--pretrained_path", type=str, help="Path to pretrained model directory")
    parser.add_argument("--baseline_episode", type=int, default=TRAINING_CONFIG["num_episodes"], 
                        help="Episode number for baseline model")
    parser.add_argument("--pretrained_episode", type=int, default=TRAINING_CONFIG["num_episodes"], 
                        help="Episode number for pretrained model")
    
    # Evaluation options
    parser.add_argument("--evaluate_baseline", action="store_true", help="Evaluate baseline agent")
    parser.add_argument("--evaluate_pretrained", action="store_true", help="Evaluate pretrained agent")
    parser.add_argument("--compare", action="store_true", help="Compare baseline and pretrained agents")
    parser.add_argument("--cold_start", action="store_true", help="Evaluate cold-start performance")
    parser.add_argument("--performance_over_time", action="store_true", help="Evaluate performance over time")
    
    # Evaluation parameters
    parser.add_argument("--num_episodes", type=int, default=EVAL_CONFIG["num_eval_episodes"], 
                        help="Number of evaluation episodes")
    parser.add_argument("--cold_start_episodes", type=int, default=10, 
                        help="Number of episodes for cold-start evaluation")
    parser.add_argument("--cold_start_steps", type=int, default=10, 
                        help="Number of steps per episode for cold-start evaluation")
    parser.add_argument("--eval_interval", type=int, default=100, 
                        help="Interval for performance over time evaluation")
    parser.add_argument("--use_llm_simulator", action="store_true", 
                        help="Use LLM simulator for evaluation")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run evaluation on")
    
    args = parser.parse_args()
    
    # If no evaluation option specified, enable all
    if not (args.evaluate_baseline or args.evaluate_pretrained or args.compare or 
            args.cold_start or args.performance_over_time):
        args.evaluate_baseline = True
        args.evaluate_pretrained = True
        args.compare = True
        args.cold_start = True
    
    main(args) 