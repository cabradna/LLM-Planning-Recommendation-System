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
from config.config import MODEL_CONFIG, TRAINING_CONFIG, EVAL_CONFIG, PATH_CONFIG, STRATEGY_CONFIG, ENV_CONFIG

# Import modules
from data.database import DatabaseConnector
from data.tensor_cache import TensorCache
from environments.job_env import JobRecommendationEnv, LLMSimulatorEnv, HybridEnv
from models.q_network import QNetwork
from models.world_model import WorldModel
from training.agent import DynaQAgent
from utils.visualizer import Visualizer
from utils.evaluator import Evaluator
from train import train_baseline_agent, train_pretrained_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluate.log"),
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
    
    # Visualize comparison
    visualizer = Visualizer()
    visualizer.plot_evaluation_results(
        baseline_rewards=comparison['baseline']['episode_rewards'],
        pretrained_rewards=comparison['pretrained']['episode_rewards'],
        title="Evaluation Results: Baseline vs. Pretrained"
    )
    
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
        env.reset(applicant_id=applicant_id)
        baseline_reward = 0
        for step in range(num_steps):
            valid_job_indices = env.get_valid_actions()
            if not valid_job_indices:
                logger.warning(f"No valid actions available for applicant {applicant_id} at step {step}")
                break
                
            action_tensors = [env.tensor_cache.get_job_vector_by_index(idx) for idx in valid_job_indices]
            action_idx, _ = baseline_agent.select_action(env.current_state, action_tensors, eval_mode=True)
            job_idx = valid_job_indices[action_idx]
            _, reward, done, _ = env.step(job_idx)
            baseline_reward += reward
            if done:
                break
        baseline_episode_rewards.append(baseline_reward)
        
        # Evaluate pretrained agent (reuse same applicant for fair comparison)
        env.reset(applicant_id=applicant_id)
        pretrained_reward = 0
        for step in range(num_steps):
            valid_job_indices = env.get_valid_actions()
            if not valid_job_indices:
                logger.warning(f"No valid actions available for applicant {applicant_id} at step {step}")
                break
                
            action_tensors = [env.tensor_cache.get_job_vector_by_index(idx) for idx in valid_job_indices]
            action_idx, _ = pretrained_agent.select_action(env.current_state, action_tensors, eval_mode=True)
            job_idx = valid_job_indices[action_idx]
            _, reward, done, _ = env.step(job_idx)
            pretrained_reward += reward
            if done:
                break
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
    
    # Visualize results
    visualizer = Visualizer()
    visualizer.plot_training_curves(
        train_values=results['rewards'],
        title="Agent Performance Over Time",
        xlabel="Episode",
        ylabel="Reward",
        save_path="performance_over_time.png"
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
    
    # Initialize TensorCache
    tensor_cache = TensorCache(device=args.device)
    logger.info(f"Initializing TensorCache on {args.device}")
    
    # Get sample applicant IDs
    try:
        collection = db_connector.db[db_connector.collections["candidates_text"]]
        applicant_ids = [doc["_id"] for doc in collection.find({}, {"_id": 1}).limit(args.num_applicants)]
        if not applicant_ids:
            raise ValueError("No applicant IDs found in the database")
        target_applicant_id = applicant_ids[0]
        logger.info(f"Selected target applicant ID: {target_applicant_id}")
    except Exception as e:
        logger.error(f"Error selecting target applicant: {e}")
        sys.exit(1)
    
    # Copy data from database to TensorCache
    tensor_cache.copy_from_database(
        db_connector=db_connector,
        applicant_ids=applicant_ids
    )
    stats = tensor_cache.cache_stats()
    logger.info(f"Cache initialized with {stats['job_count']} jobs and {stats['applicant_state_count']} applicant states")
    
    # Create evaluation environment based on reward strategy
    if args.reward_strategy == "cosine":
        env = JobRecommendationEnv(
            tensor_cache=tensor_cache,
            reward_scheme=STRATEGY_CONFIG["llm"]["response_mapping"],
            reward_strategy="cosine",
            random_seed=ENV_CONFIG["random_seed"]
        )
    elif args.reward_strategy == "llm":
        env = LLMSimulatorEnv(
            tensor_cache=tensor_cache,
            reward_scheme=STRATEGY_CONFIG["llm"]["response_mapping"],
            random_seed=ENV_CONFIG["random_seed"]
        )
        
        # Setup LLM if needed
        if args.use_llm_simulator:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                from huggingface_hub import login
                
                # Get token path from config and read token
                from config.config import HF_CONFIG
                token_path = HF_CONFIG["token_path"]
                if not os.path.isabs(token_path):
                    token_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), token_path)
                    
                with open(token_path, "r") as f:
                    token = f.read().strip()
                
                # Login to Hugging Face
                login(token=token)
                logger.info(f"Successfully logged in to Hugging Face")
                
                # Load model and tokenizer
                model_id = STRATEGY_CONFIG["llm"]["model_id"]
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=STRATEGY_CONFIG["llm"]["quantization"]["load_in_4bit"],
                    bnb_4bit_quant_type=STRATEGY_CONFIG["llm"]["quantization"]["quant_type"],
                    bnb_4bit_use_nested_quant=STRATEGY_CONFIG["llm"]["quantization"]["use_nested_quant"],
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                
                logger.info(f"Loading LLM model {model_id}")
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                llm_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto"
                )
                
                # Setup LLM in environment
                env = env.setup_llm(llm_model, tokenizer, device="auto")
                logger.info(f"LLM model set up in environment")
            except Exception as e:
                logger.error(f"Failed to set up LLM: {e}")
                sys.exit(1)
    elif args.reward_strategy == "hybrid":
        env = HybridEnv(
            tensor_cache=tensor_cache,
            reward_scheme=STRATEGY_CONFIG["llm"]["response_mapping"],
            cosine_weight=STRATEGY_CONFIG["hybrid"]["initial_cosine_weight"],
            random_seed=ENV_CONFIG["random_seed"]
        )
        
        # Setup LLM if needed
        if args.use_llm_simulator:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                from huggingface_hub import login
                
                # Get token path from config and read token
                from config.config import HF_CONFIG
                token_path = HF_CONFIG["token_path"]
                if not os.path.isabs(token_path):
                    token_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), token_path)
                    
                with open(token_path, "r") as f:
                    token = f.read().strip()
                
                # Login to Hugging Face
                login(token=token)
                logger.info(f"Successfully logged in to Hugging Face")
                
                # Load model and tokenizer
                model_id = STRATEGY_CONFIG["llm"]["model_id"]
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=STRATEGY_CONFIG["llm"]["quantization"]["load_in_4bit"],
                    bnb_4bit_quant_type=STRATEGY_CONFIG["llm"]["quantization"]["quant_type"],
                    bnb_4bit_use_nested_quant=STRATEGY_CONFIG["llm"]["quantization"]["use_nested_quant"],
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                
                logger.info(f"Loading LLM model {model_id}")
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                llm_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto"
                )
                
                # Setup LLM in environment
                env = env.setup_llm(llm_model, tokenizer, device="auto")
                logger.info(f"LLM model set up in environment")
            except Exception as e:
                logger.error(f"Failed to set up LLM: {e}")
                sys.exit(1)
    else:
        logger.error(f"Invalid reward strategy: {args.reward_strategy}")
        sys.exit(1)
    
    logger.info(f"Created {env.__class__.__name__} with '{args.reward_strategy}' reward strategy")
    
    # Load agents
    device = torch.device(args.device)
    
    # Load baseline agent
    baseline_path = args.baseline_path or os.path.join(
        PATH_CONFIG["model_dir"], "baseline"
    )
    logger.info(f"Loading baseline agent from {baseline_path}")
    baseline_agent = DynaQAgent.load_models(
        model_dir=baseline_path,
        episode=args.baseline_episode,
        device=device
    )
    
    # Set tensor_cache for baseline agent
    baseline_agent.tensor_cache = tensor_cache
    
    # Load pretrained agent
    pretrained_path = args.pretrained_path or os.path.join(
        PATH_CONFIG["model_dir"], "pretrained"
    )
    logger.info(f"Loading pretrained agent from {pretrained_path}")
    pretrained_agent = DynaQAgent.load_models(
        model_dir=pretrained_path,
        episode=args.pretrained_episode,
        device=device
    )
    
    # Set tensor_cache for pretrained agent
    pretrained_agent.tensor_cache = tensor_cache
    
    # Run evaluations based on command-line arguments
    if args.evaluate_baseline:
        logger.info("Evaluating baseline agent")
        evaluate_model_performance(
            agent=baseline_agent,
            env=env,
            applicant_ids=applicant_ids,
            num_episodes=args.num_episodes,
            agent_name="Baseline Agent"
        )
    
    if args.evaluate_pretrained:
        logger.info("Evaluating pretrained agent")
        evaluate_model_performance(
            agent=pretrained_agent,
            env=env,
            applicant_ids=applicant_ids,
            num_episodes=args.num_episodes,
            agent_name="Pretrained Agent"
        )
    
    if args.compare:
        logger.info("Comparing baseline and pretrained agents")
        compare_baseline_and_pretrained(
            baseline_agent=baseline_agent,
            pretrained_agent=pretrained_agent,
            env=env,
            applicant_ids=applicant_ids,
            num_episodes=args.num_episodes
        )
    
    if args.cold_start:
        logger.info("Evaluating cold-start performance")
        evaluate_cold_start_performance(
            baseline_agent=baseline_agent,
            pretrained_agent=pretrained_agent,
            env=env,
            applicant_ids=applicant_ids,
            num_episodes=args.cold_start_episodes,
            num_steps=args.cold_start_steps
        )
    
    if args.performance_over_time:
        logger.info("Evaluating performance over time")
        evaluate_performance_over_time(
            agent=pretrained_agent,
            env=env,
            applicant_ids=applicant_ids,
            steps=list(range(0, args.num_episodes, args.eval_interval))
        )
    
    # Close database connection
    db_connector.close()
    
    # Clean up resources
    if tensor_cache is not None and hasattr(tensor_cache, 'clear'):
        tensor_cache.clear()
        logger.info("Tensor cache cleared")
        
    # Free PyTorch memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache emptied")
    
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
    
    # Reward strategy
    parser.add_argument("--reward_strategy", type=str, choices=["cosine", "llm", "hybrid"], default="hybrid",
                       help="Reward strategy to use for evaluation")
    
    # Evaluation parameters
    parser.add_argument("--num_episodes", type=int, default=EVAL_CONFIG["num_eval_episodes"], 
                        help="Number of evaluation episodes")
    parser.add_argument("--num_applicants", type=int, default=TRAINING_CONFIG["num_candidates"],
                       help="Number of applicants to use for evaluation")
    parser.add_argument("--cold_start_episodes", type=int, default=10, 
                        help="Number of episodes for cold-start evaluation")
    parser.add_argument("--cold_start_steps", type=int, default=10, 
                        help="Number of steps per episode for cold-start evaluation")
    parser.add_argument("--eval_interval", type=int, default=100, 
                        help="Interval for performance over time evaluation")
    parser.add_argument("--use_llm_simulator", action="store_true", 
                        help="Use LLM simulator for evaluation")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=ENV_CONFIG["random_seed"], help="Random seed")
    parser.add_argument("--device", type=str, default=TRAINING_CONFIG["device"], 
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