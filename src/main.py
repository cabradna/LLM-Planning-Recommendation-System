"""
Main entry point for the Dyna-Q job recommender system.

This script provides a unified command-line interface to access the various
components of the job recommendation system, including data preprocessing,
pretraining, training, and evaluation.
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def show_info() -> None:
    """
    Display information about the Dyna-Q job recommender system.
    """
    logger.info("Dyna-Q Job Recommender System")
    logger.info("============================")
    logger.info("A reinforcement learning-based job recommendation system using the Dyna-Q algorithm.")
    logger.info("")
    logger.info("Key Features:")
    logger.info("- Combines model-free RL learning with model-based planning")
    logger.info("- Uses data from MongoDB database for pretraining to solve the cold-start problem")
    logger.info("- Incorporates neural networks for Q-value and world model approximation")
    logger.info("- Simulates user feedback with response modeling")
    logger.info("")
    logger.info("Available Commands:")
    logger.info("  pretraining  - Pretrain models using database data")
    logger.info("  train        - Train models using user interaction data")
    logger.info("  evaluate     - Evaluate trained models and compare performance")
    logger.info("  simulate     - Run simulations with trained models")
    logger.info("  info         - Display this information")
    logger.info("")
    logger.info("For more details on each command, run: python main.py <command> --help")

def run_pretraining(args: List[str]) -> None:
    """
    Run the pretraining script with the given arguments.
    
    Args:
        args: Command-line arguments to pass to the pretraining script.
    """
    logger.info("Running pretraining...")
    from pretraining import main
    
    # Parse pretraining-specific arguments
    parser = argparse.ArgumentParser(description="Pretrain Dyna-Q job recommender models")
    
    # Data options
    parser.add_argument("--method", type=str, choices=["cosine", "llm", "hybrid"], default="cosine",
                      help="Method for pretraining: cosine (baseline), llm, or hybrid")
    parser.add_argument("--cosine_weight", type=float, default=0.3,
                      help="Weight for cosine similarity in hybrid method (0-1)")
    parser.add_argument("--dataset_limit", type=int, default=None,
                      help="Maximum number of examples to load from database")
    
    # Pretraining options
    parser.add_argument("--pretrain_q_network", action="store_true", help="Pretrain Q-network")
    parser.add_argument("--pretrain_world_model", action="store_true", help="Pretrain world model")
    
    # Model parameters
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate for pretraining")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for pretraining")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs for pretraining")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save pretrained models")
    
    # Parse arguments
    pretraining_args = parser.parse_args(args)
    
    # If no pretraining option specified, enable all
    if not (pretraining_args.pretrain_q_network or pretraining_args.pretrain_world_model):
        pretraining_args.pretrain_q_network = True
        pretraining_args.pretrain_world_model = True
    
    # Run pretraining with parsed arguments
    main(pretraining_args)
    
    logger.info(f"Pretraining complete with method: {pretraining_args.method}")

def run_training(args: List[str]) -> None:
    """
    Run the training script with the given arguments.
    
    Args:
        args: Command-line arguments to pass to the training script.
    """
    logger.info("Running training...")
    from train import main
    
    # Parse training-specific arguments
    parser = argparse.ArgumentParser(description="Train Dyna-Q job recommender models")
    
    # Training options
    parser.add_argument("--train_baseline", action="store_true", help="Train baseline agent")
    parser.add_argument("--train_pretrained", action="store_true", help="Train pretrained agent")
    parser.add_argument("--pretrained_model_path", type=str, help="Path to pretrained model directory")
    
    # Training strategy
    parser.add_argument("--strategy", type=str, choices=["cosine", "llm", "hybrid"], default="cosine",
                      help="Training strategy: cosine (baseline), llm (LLM feedback), or hybrid (combined)")
    parser.add_argument("--cosine_weight", type=float, default=1.0,
                      help="Initial weight for cosine similarity in hybrid method (0-1)")
    parser.add_argument("--switch_episode", type=int, default=None,
                      help="Episode to switch from cosine to LLM feedback in hybrid strategy")
    parser.add_argument("--cosine_annealing", action="store_true",
                      help="Whether to gradually anneal cosine weight in hybrid strategy")
    
    # Training parameters
    parser.add_argument("--num_episodes", type=int, default=None, help="Number of training episodes")
    parser.add_argument("--planning_steps", type=int, default=None, help="Number of planning steps per real step")
    
    # Evaluation options
    parser.add_argument("--compare", action="store_true", help="Compare baseline and pretrained agents")
    
    # Environment options
    parser.add_argument("--use_llm_simulator", action="store_true", help="Use LLM simulator for training")
    parser.add_argument("--llm_model_name", type=str, default=None, help="Name of LLM model to use for simulation")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save trained models")
    
    # Parse arguments
    training_args = parser.parse_args(args)
    
    # If no training option specified, train both
    if not (training_args.train_baseline or training_args.train_pretrained):
        training_args.train_baseline = True
        training_args.train_pretrained = True
    
    # Run training with parsed arguments
    main(training_args)
    
    logger.info("Training complete")

def run_evaluation(args: List[str]) -> None:
    """
    Run the evaluation script with the given arguments.
    
    Args:
        args: Command-line arguments to pass to the evaluation script.
    """
    logger.info("Running evaluation...")
    from evaluate import main
    
    # Parse evaluation-specific arguments
    parser = argparse.ArgumentParser(description="Evaluate Dyna-Q job recommender models")
    
    # Model paths
    parser.add_argument("--baseline_path", type=str, help="Path to baseline model directory")
    parser.add_argument("--pretrained_path", type=str, help="Path to pretrained model directory")
    parser.add_argument("--baseline_episode", type=int, default=None, help="Episode number for baseline model")
    parser.add_argument("--pretrained_episode", type=int, default=None, help="Episode number for pretrained model")
    
    # Evaluation options
    parser.add_argument("--evaluate_baseline", action="store_true", help="Evaluate baseline agent")
    parser.add_argument("--evaluate_pretrained", action="store_true", help="Evaluate pretrained agent")
    parser.add_argument("--compare", action="store_true", help="Compare baseline and pretrained agents")
    parser.add_argument("--cold_start", action="store_true", help="Evaluate cold-start performance")
    parser.add_argument("--performance_over_time", action="store_true", help="Evaluate performance over time")
    
    # Evaluation parameters
    parser.add_argument("--num_episodes", type=int, default=None, help="Number of evaluation episodes")
    parser.add_argument("--cold_start_episodes", type=int, default=10, help="Number of episodes for cold-start evaluation")
    parser.add_argument("--cold_start_steps", type=int, default=10, help="Number of steps per episode for cold-start evaluation")
    parser.add_argument("--eval_interval", type=int, default=100, help="Interval for performance over time evaluation")
    parser.add_argument("--use_llm_simulator", action="store_true", help="Use LLM simulator for evaluation")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Parse arguments
    evaluation_args = parser.parse_args(args)
    
    # If no evaluation option specified, enable all
    if not (evaluation_args.evaluate_baseline or evaluation_args.evaluate_pretrained or 
            evaluation_args.compare or evaluation_args.cold_start or evaluation_args.performance_over_time):
        evaluation_args.evaluate_baseline = True
        evaluation_args.evaluate_pretrained = True
        evaluation_args.compare = True
        evaluation_args.cold_start = True
    
    # Run evaluation with parsed arguments
    main(evaluation_args)
    
    logger.info("Evaluation complete")

def run_simulation(args: List[str]) -> None:
    """
    Run a simulation with a trained model.
    
    Args:
        args: Command-line arguments for the simulation.
    """
    logger.info("Running simulation...")
    
    # Parse simulation-specific arguments
    parser = argparse.ArgumentParser(description="Run simulation with trained models")
    
    # Model options
    parser.add_argument("--model_path", type=str, required=True, help="Path to model directory")
    parser.add_argument("--episode", type=int, default=None, help="Episode number to load")
    
    # Simulation parameters
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of simulation episodes")
    parser.add_argument("--candidate_id", type=str, default=None, help="Specific candidate ID to simulate")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Parse arguments
    simulation_args = parser.parse_args(args)
    
    # Import necessary modules (delayed import to avoid circular dependencies)
    import torch
    import random
    import numpy as np
    from data.database import DatabaseConnector
    from environments.job_env import JobRecommendationEnv, LLMSimulatorEnv
    from training.agent import DynaQAgent
    
    # Set random seed
    torch.manual_seed(simulation_args.seed)
    torch.cuda.manual_seed_all(simulation_args.seed)
    np.random.seed(simulation_args.seed)
    random.seed(simulation_args.seed)
    
    # Initialize database connector
    db_connector = DatabaseConnector()
    
    # Create environment
    env = JobRecommendationEnv(db_connector=db_connector, random_seed=simulation_args.seed)
    
    # Load agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DynaQAgent.load_models(
        model_dir=simulation_args.model_path,
        episode=simulation_args.episode,
        device=device
    )
    
    logger.info(f"Running simulation with model from {simulation_args.model_path}")
    
    # Get candidate IDs
    if simulation_args.candidate_id:
        # Use the specified candidate ID
        candidate_ids = [simulation_args.candidate_id]
    else:
        # Fetch a sample of candidate IDs from the database
        candidates_collection = db_connector.db[db_connector.collections["candidates_embeddings"]]
        candidates = list(candidates_collection.find().limit(simulation_args.num_episodes))
        candidate_ids = [candidate.get("original_candidate_id") for candidate in candidates if candidate.get("original_candidate_id")]
        
        if not candidate_ids:
            logger.error("No candidate IDs found in the database")
            return
    
    # Run simulation for each candidate
    total_reward = 0
    episode_rewards = []
    
    for episode, candidate_id in enumerate(candidate_ids[:simulation_args.num_episodes]):
        logger.info(f"Episode {episode+1}/{len(candidate_ids[:simulation_args.num_episodes])}: Candidate {candidate_id}")
        
        try:
            # Get candidate state
            state = db_connector.get_applicant_state(candidate_id)
            
            # Sample candidate jobs
            candidate_jobs = db_connector.sample_candidate_jobs(n=20)
            job_ids = [job.get("original_job_id") for job in candidate_jobs if job.get("original_job_id")]
            
            if not job_ids:
                logger.warning(f"No job IDs found for candidate {candidate_id}, skipping")
                continue
            
            # Get job vectors
            job_vectors = db_connector.get_job_vectors(job_ids)
            
            if not job_vectors:
                logger.warning(f"No job vectors found for candidate {candidate_id}, skipping")
                continue
            
            # Initialize environment for this candidate
            env.reset(candidate_id=candidate_id, available_jobs=job_ids)
            
            # Interactive mode
            if simulation_args.interactive:
                print(f"\nCandidate ID: {candidate_id}")
                print("-" * 50)
                
                # Get candidate details
                candidates_text_collection = db_connector.db["candidates_text"]
                candidate_text = candidates_text_collection.find_one({"original_candidate_id": candidate_id})
                
                if candidate_text:
                    print(f"Candidate Bio: {candidate_text.get('bio', 'N/A')}")
                    print(f"Candidate Skills: {', '.join(candidate_text.get('skills', []))}")
                    print("-" * 50)
                
                step = 0
                episode_reward = 0
                done = False
                
                while not done and step < 10:  # Limit to 10 steps for interactive mode
                    # Agent selects action
                    job_idx = agent.select_action(state, job_vectors)
                    job_id = job_ids[job_idx]
                    
                    # Get job details
                    job_details = db_connector.get_job_details(job_id)
                    
                    if job_details:
                        print(f"\nStep {step+1}: Recommended Job")
                        print(f"Job Title: {job_details.get('job_title', 'N/A')}")
                        print(f"Description: {job_details.get('description', 'N/A')[:200]}...")
                        print("-" * 50)
                    
                    # Get user feedback
                    valid_feedback = False
                    while not valid_feedback:
                        feedback = input("Your feedback (APPLY: 1.0, SAVE: 0.5, CLICK: 0.0, IGNORE: -0.1): ").strip().upper()
                        
                        feedback_map = {
                            "APPLY": 1.0,
                            "SAVE": 0.5,
                            "CLICK": 0.0,
                            "IGNORE": -0.1,
                            "1.0": 1.0,
                            "0.5": 0.5,
                            "0.0": 0.0,
                            "-0.1": -0.1
                        }
                        
                        if feedback in feedback_map:
                            reward = feedback_map[feedback]
                            valid_feedback = True
                        else:
                            print("Invalid feedback. Please use APPLY, SAVE, CLICK, IGNORE or the corresponding values.")
                    
                    # Take action
                    next_state, reward, done, _ = env.step(job_idx)
                    
                    # Update agent
                    agent.update(state, job_vectors[job_idx], reward, next_state)
                    
                    # Update state
                    state = next_state
                    episode_reward += reward
                    step += 1
                    
                    print(f"Reward: {reward}, Cumulative Reward: {episode_reward}")
                    print("-" * 50)
                    
                    if done:
                        print("Episode complete!")
                        break
                
                # Continue to next candidate?
                if episode < len(candidate_ids) - 1:
                    continue_sim = input("\nContinue to next candidate? (y/n): ").strip().lower()
                    if continue_sim != 'y':
                        break
            
            # Automated mode
            else:
                # Run episode
                state = env.reset(candidate_id=candidate_id, available_jobs=job_ids)
                episode_reward = 0
                done = False
                step = 0
                
                while not done and step < 20:  # Limit steps per episode
                    # Select action
                    job_idx = agent.select_action(state, job_vectors)
                    
                    # Take action
                    next_state, reward, done, _ = env.step(job_idx)
                    
                    # Update agent
                    agent.update(state, job_vectors[job_idx], reward, next_state)
                    
                    # Update state
                    state = next_state
                    episode_reward += reward
                    step += 1
                
                logger.info(f"Episode {episode+1} complete: Reward = {episode_reward:.2f}")
                episode_rewards.append(episode_reward)
                total_reward += episode_reward
        except Exception as e:
            logger.error(f"Error in episode {episode+1} with candidate {candidate_id}: {e}")
            continue
    
    # Report results
    if not simulation_args.interactive and episode_rewards:
        avg_reward = total_reward / len(episode_rewards)
        logger.info(f"Simulation complete: Average reward = {avg_reward:.2f}")
        
    # Close database connection
    db_connector.close()
    
    logger.info("Simulation complete")

def main() -> None:
    """
    Main entry point for the Dyna-Q job recommender system.
    """
    # Create argument parser
    parser = argparse.ArgumentParser(description="Dyna-Q Job Recommender System", add_help=False)
    parser.add_argument("command", choices=["pretraining", "train", "evaluate", "simulate", "info"],
                        help="Command to execute", nargs="?", default="info")
    
    # Parse known arguments
    args, remaining = parser.parse_known_args()
    
    # Execute the specified command
    if args.command == "pretraining":
        run_pretraining(remaining)
    elif args.command == "train":
        run_training(remaining)
    elif args.command == "evaluate":
        run_evaluation(remaining)
    elif args.command == "simulate":
        run_simulation(remaining)
    elif args.command == "info":
        show_info()
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 