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
import torch
import numpy as np
import random

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from config.config import HF_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, PATH_CONFIG, STRATEGY_CONFIG, ENV_CONFIG

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

# Login to Hugging Face
try:
    token_path = HF_CONFIG["token_path"]
    
    # Ensure an absolute path by resolving relative to project root
    if not os.path.isabs(token_path):
        token_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), token_path))
    
    with open(token_path, "r") as f:
        token = f.read().strip()
    
    from huggingface_hub import login
    login(token=token)
    logger.info("Successfully logged in to Hugging Face")
except Exception as e:
    logger.error(f"Failed to login to Hugging Face: {e}")
    logger.warning("Hugging Face login failed - LLM functionality may be limited")

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
    parser.add_argument("--strategy", type=str, choices=["cosine", "llm", "hybrid"], default="cosine",
                      help="Method for pretraining: cosine (baseline), llm, or hybrid")
    parser.add_argument("--cosine_weight", type=float, default=STRATEGY_CONFIG["hybrid"]["initial_cosine_weight"],
                      help="Weight for cosine similarity in hybrid method (0-1)")
    parser.add_argument("--dataset_limit", type=int, default=None,
                      help="Maximum number of examples to load from database")
    
    # Pretraining options
    parser.add_argument("--pretrain_q_network", action="store_true", help="Pretrain Q-network")
    parser.add_argument("--pretrain_world_model", action="store_true", help="Pretrain world model")
    
    # Model parameters
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate for pretraining")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for pretraining")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs for pretraining")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=ENV_CONFIG["random_seed"], help="Random seed")
    parser.add_argument("--device", type=str, default=TRAINING_CONFIG["device"], help="Device to use for training")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save pretrained models")
    parser.add_argument("--num_applicants", type=int, default=TRAINING_CONFIG["num_candidates"],
                      help="Number of applicants to use for pretraining")
    
    # Parse arguments
    pretraining_args = parser.parse_args(args)
    
    # If no pretraining option specified, enable all
    if not (pretraining_args.pretrain_q_network or pretraining_args.pretrain_world_model):
        pretraining_args.pretrain_q_network = True
        pretraining_args.pretrain_world_model = True
    
    # Run pretraining with parsed arguments
    main(pretraining_args)
    
    logger.info(f"Pretraining complete with strategy: {pretraining_args.strategy}")

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
    parser.add_argument("--applicant_id", type=str, help="Target applicant ID to train for")
    
    # Training strategy
    parser.add_argument("--strategy", type=str, choices=["cosine", "llm", "hybrid"], default="hybrid",
                      help="Training strategy: cosine (baseline), llm (LLM feedback), or hybrid (combined)")
    parser.add_argument("--cosine_weight", type=float, default=STRATEGY_CONFIG["hybrid"]["initial_cosine_weight"],
                      help="Initial weight for cosine similarity in hybrid method (0-1)")
    parser.add_argument("--final_cosine_weight", type=float, default=STRATEGY_CONFIG["hybrid"]["final_cosine_weight"],
                      help="Final weight for cosine similarity in hybrid method (0-1)")
    parser.add_argument("--annealing_episodes", type=int, default=STRATEGY_CONFIG["hybrid"]["annealing_episodes"],
                      help="Number of episodes to anneal cosine weight in hybrid strategy")
    
    # Training parameters
    parser.add_argument("--num_episodes", type=int, default=TRAINING_CONFIG["num_episodes"], 
                      help="Number of training episodes")
    parser.add_argument("--max_steps_per_episode", type=int, default=TRAINING_CONFIG["max_steps_per_episode"],
                      help="Maximum steps per episode")
    parser.add_argument("--planning_steps", type=int, default=TRAINING_CONFIG["planning_steps"], 
                      help="Number of planning steps per real step")
    parser.add_argument("--batch_size", type=int, default=TRAINING_CONFIG["batch_size"],
                      help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=TRAINING_CONFIG["lr"],
                      help="Learning rate for training")
    parser.add_argument("--gamma", type=float, default=TRAINING_CONFIG["gamma"],
                      help="Discount factor for RL")
    
    # Evaluation options
    parser.add_argument("--eval_frequency", type=int, default=TRAINING_CONFIG["eval_frequency"],
                      help="Frequency of evaluation during training")
    parser.add_argument("--num_eval_episodes", type=int, default=TRAINING_CONFIG["num_eval_episodes"],
                      help="Number of episodes for evaluation during training")
    
    # Environment options
    parser.add_argument("--use_llm_simulator", action="store_true", help="Use LLM simulator for training")
    parser.add_argument("--llm_model_name", type=str, default=STRATEGY_CONFIG["llm"]["model_id"], 
                      help="Name of LLM model to use for simulation")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=ENV_CONFIG["random_seed"], help="Random seed")
    parser.add_argument("--device", type=str, default=TRAINING_CONFIG["device"], help="Device to use for training")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save trained models")
    parser.add_argument("--num_applicants", type=int, default=TRAINING_CONFIG["num_candidates"],
                      help="Number of applicants to use for training")
    
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
    
    # Reward strategy
    parser.add_argument("--reward_strategy", type=str, choices=["cosine", "llm", "hybrid"], default="hybrid",
                       help="Reward strategy to use for evaluation")
    
    # Evaluation parameters
    parser.add_argument("--num_episodes", type=int, default=TRAINING_CONFIG["num_eval_episodes"], 
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
    parser.add_argument("--applicant_id", type=str, default=None, help="Specific applicant ID to simulate")
    parser.add_argument("--num_recommendations", type=int, default=5, help="Number of recommendations to generate")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    # Strategy
    parser.add_argument("--strategy", type=str, choices=["cosine", "llm", "hybrid"], default="hybrid",
                       help="Reward strategy to use for simulation")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=ENV_CONFIG["random_seed"], help="Random seed")
    parser.add_argument("--device", type=str, default=TRAINING_CONFIG["device"], help="Device to use for simulation")
    
    # Parse arguments
    simulation_args = parser.parse_args(args)
    
    # Set random seed
    set_seed(simulation_args.seed)
    
    # Import necessary modules
    import torch
    from data.database import DatabaseConnector
    from data.tensor_cache import TensorCache
    from environments.job_env import JobRecommendationEnv, LLMSimulatorEnv, HybridEnv
    from training.agent import DynaQAgent
    from utils.visualizer import Visualizer
    
    # Initialize database connector
    db_connector = DatabaseConnector()
    
    # Initialize TensorCache
    tensor_cache = TensorCache(device=simulation_args.device)
    logger.info(f"Initializing TensorCache on {simulation_args.device}")
    
    # Get applicant IDs
    if simulation_args.applicant_id:
        # Use the specified applicant ID
        applicant_ids = [simulation_args.applicant_id]
        logger.info(f"Using specified applicant ID: {simulation_args.applicant_id}")
    else:
        # Fetch applicant IDs from the database
        try:
            collection = db_connector.db[db_connector.collections["candidates_text"]]
            applicant_ids = [doc["_id"] for doc in collection.find({}, {"_id": 1}).limit(10)]
            if not applicant_ids:
                raise ValueError("No applicant IDs found in the database")
            logger.info(f"Selected applicant IDs from database: {applicant_ids[0]} (plus {len(applicant_ids)-1} more)")
        except Exception as e:
            logger.error(f"Error selecting applicants: {e}")
            return
    
    # Copy data from database to TensorCache
    tensor_cache.copy_from_database(
        db_connector=db_connector,
        applicant_ids=applicant_ids
    )
    stats = tensor_cache.cache_stats()
    logger.info(f"Cache initialized with {stats['job_count']} jobs and {stats['applicant_state_count']} applicant states")
    
    # Create simulation environment based on reward strategy
    if simulation_args.strategy == "cosine":
        env = JobRecommendationEnv(
            tensor_cache=tensor_cache,
            reward_scheme=STRATEGY_CONFIG["llm"]["response_mapping"],
            reward_strategy="cosine",
            random_seed=simulation_args.seed
        )
        logger.info("Created JobRecommendationEnv with cosine reward strategy")
    elif simulation_args.strategy == "llm":
        env = LLMSimulatorEnv(
            tensor_cache=tensor_cache,
            reward_scheme=STRATEGY_CONFIG["llm"]["response_mapping"],
            random_seed=simulation_args.seed
        )
        
        # Setup LLM
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
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
            return
        
        logger.info("Created LLMSimulatorEnv with LLM reward strategy")
    elif simulation_args.strategy == "hybrid":
        env = HybridEnv(
            tensor_cache=tensor_cache,
            reward_scheme=STRATEGY_CONFIG["llm"]["response_mapping"],
            cosine_weight=STRATEGY_CONFIG["hybrid"]["initial_cosine_weight"],
            random_seed=simulation_args.seed
        )
        
        # Setup LLM
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
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
            return
        
        logger.info("Created HybridEnv with hybrid reward strategy")
    else:
        logger.error(f"Invalid reward strategy: {simulation_args.strategy}")
        return
    
    # Load agent
    device = torch.device(simulation_args.device)
    try:
        agent = DynaQAgent.load_models(
            model_dir=simulation_args.model_path,
            episode=simulation_args.episode,
            device=device
        )
        # Set tensor_cache for agent
        agent.tensor_cache = tensor_cache
        logger.info(f"Loaded agent from {simulation_args.model_path}")
    except Exception as e:
        logger.error(f"Failed to load agent: {e}")
        return
    
    # Initialize visualizer
    visualizer = Visualizer()
    
    # Target applicant for recommendations
    target_applicant_id = applicant_ids[0]
    
    # Interactive mode
    if simulation_args.interactive:
        print(f"\nInteractive Simulation Mode for Applicant ID: {target_applicant_id}")
        print("=" * 80)
        
        # Get applicant details
        try:
            candidate_text = db_connector.get_applicant_details(target_applicant_id)
            if candidate_text:
                print(f"Applicant Bio: {candidate_text.get('bio', 'N/A')}")
                print(f"Applicant Skills: {', '.join(candidate_text.get('skills', []))}")
                print("=" * 80)
        except Exception as e:
            print(f"Error retrieving applicant details: {e}")
        
        # Reset environment for target applicant
        state = env.reset(applicant_id=target_applicant_id)
        
        # Generate and display recommendations
        recommendations = []
        
        for i in range(simulation_args.num_recommendations):
            valid_job_indices = env.get_valid_actions()
            
            if not valid_job_indices:
                print("No more valid job recommendations available.")
                break
            
            # Get job tensors
            action_tensors = [tensor_cache.get_job_vector_by_index(idx) for idx in valid_job_indices]
            
            # Select action
            action_idx, _ = agent.select_action(state, action_tensors, eval_mode=True)
            job_idx = valid_job_indices[action_idx]
            
            # Get job details
            try:
                job_id = tensor_cache.get_job_id(job_idx)
                job_details = tensor_cache.get_job_metadata(job_id)
                
                # Get Q-value
                with torch.no_grad():
                    state_tensor = state.unsqueeze(0) if state.dim() == 1 else state
                    job_tensor = tensor_cache.get_job_vector_by_index(job_idx).unsqueeze(0)
                    q_value = agent.q_network(state_tensor, job_tensor).item()
                
                recommendations.append({
                    "rank": i + 1,
                    "job_id": job_id,
                    "job_title": job_details.get("job_title", "N/A"),
                    "q_value": q_value,
                    "job_details": job_details
                })
                
                # Display recommendation
                print(f"\nRecommendation #{i+1}: [Q-Value: {q_value:.4f}]")
                print(f"Job Title: {job_details.get('job_title', 'N/A')}")
                description = job_details.get('description', 'N/A')
                print(f"Description: {description[:200]}..." if len(description) > 200 else f"Description: {description}")
                if 'technical_skills' in job_details:
                    skills = job_details['technical_skills']
                    if isinstance(skills, list):
                        print(f"Technical Skills: {', '.join(map(str, skills))}")
                    else:
                        print(f"Technical Skills: {skills}")
                print("-" * 80)
                
                # Get user feedback
                if not simulation_args.interactive:
                    # Take environment step
                    next_state, reward, done, _ = env.step(job_idx)
                    state = next_state
                    
                    print(f"Automatic feedback - Reward: {reward:.2f}")
                    
                    if done:
                        print("Episode complete!")
                        break
                else:
                    valid_feedback = False
                    while not valid_feedback:
                        feedback = input("Your feedback (APPLY: 1.0, SAVE: 0.5, CLICK: 0.0, IGNORE: -0.1, NEXT: skip): ").strip().upper()
                        
                        if feedback == "NEXT":
                            print("Skipping to next recommendation...")
                            break
                        
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
                            reward_value = feedback_map[feedback]
                            # Take environment step (this affects future recommendations)
                            next_state, reward, done, _ = env.step(job_idx)
                            
                            # Optionally update the agent (if learning during simulation)
                            # agent.update(state, tensor_cache.get_job_vector_by_index(job_idx), reward, next_state)
                            
                            state = next_state
                            print(f"Feedback recorded: {feedback} (Reward: {reward_value})")
                            
                            valid_feedback = True
                        else:
                            print("Invalid feedback. Please use APPLY, SAVE, CLICK, IGNORE or the corresponding values.")
                    
                    if valid_feedback and done:
                        print("Episode complete!")
                        break
            except Exception as e:
                print(f"Error processing recommendation: {e}")
                continue
        
        print("\nSimulation complete!")
    
    # Non-interactive mode
    else:
        logger.info(f"Running non-interactive simulation for applicant: {target_applicant_id}")
        
        # Reset environment for target applicant
        state = env.reset(applicant_id=target_applicant_id)
        
        # Generate recommendations
        recommended_jobs = []
        recommendation_scores = []
        valid_job_indices = list(range(len(tensor_cache)))
        
        for _ in range(min(simulation_args.num_recommendations, len(valid_job_indices))):
            action_tensors = [tensor_cache.get_job_vector_by_index(idx) for idx in valid_job_indices]
            
            if not action_tensors:
                logger.warning("No more actions available to recommend.")
                break
                
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, device=device, dtype=torch.float32)
                
            action_idx, _ = agent.select_action(state, action_tensors, eval_mode=True) 
            
            selected_cache_idx = valid_job_indices[action_idx]
            job_id = tensor_cache.get_job_id(selected_cache_idx)
            
            with torch.no_grad():
                state_tensor = state.unsqueeze(0) if state.dim() == 1 else state 
                job_tensor = tensor_cache.get_job_vector_by_index(selected_cache_idx).unsqueeze(0)
                q_value = agent.q_network(state_tensor, job_tensor).item()
            
            recommended_jobs.append(job_id)
            recommendation_scores.append(q_value)
            
            # Take step in environment
            next_state, reward, done, _ = env.step(selected_cache_idx)
            state = next_state
            
            # Remove recommended job from valid jobs
            valid_job_indices.pop(action_idx)
            
            if done:
                logger.info("Episode complete during recommendation generation")
                break
        
        # Log and display recommendations
        logger.info(f"Generated {len(recommended_jobs)} recommendations for applicant {target_applicant_id}")
        
        print(f"\n=== Top Job Recommendations for Applicant: {target_applicant_id} ===\n")
        for i, (job_id, score) in enumerate(zip(recommended_jobs, recommendation_scores)):
            print(f"Recommendation #{i+1}: [Q-Value: {score:.4f}]")
            print(f"Job ID: {job_id}")

            try:
                job_details = tensor_cache.get_job_metadata(job_id)
                print(f"Title: {job_details.get('job_title', 'N/A')}")
                description = job_details.get('description', 'N/A')
                print(f"Description: {description[:100]}..." if len(description) > 100 else f"Description: {description}")
                if 'technical_skills' in job_details:
                    skills = job_details['technical_skills']
                    if isinstance(skills, list):
                        print(f"Technical Skills: {', '.join(map(str, skills))}")
                    else:
                        print(f"Technical Skills: {skills}")
            except Exception as e:
                logger.error(f"Error retrieving job details: {e}")
                print(f"Error retrieving job details from cache: {e}")
            print("-" * 50)
    
    # Clean up resources
    db_connector.close()
    
    if tensor_cache is not None and hasattr(tensor_cache, 'clear'):
        tensor_cache.clear()
        logger.info("Tensor cache cleared")
        
    # Free PyTorch memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache emptied")
    
    logger.info("Simulation complete")

def main() -> None:
    """
    Main entry point for the Dyna-Q job recommender system.
    """
    # Set random seed for reproducibility
    set_seed(ENV_CONFIG["random_seed"])
    
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