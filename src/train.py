"""
Main training script for the Dyna-Q job recommender model.
"""

import os
import sys
import logging
import argparse
import torch
import numpy as np
import random
from typing import List, Dict, Optional, Any

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import MODEL_CONFIG, TRAINING_CONFIG, PATH_CONFIG

# Import modules
from data.database import DatabaseConnector
from environments.job_env import JobRecommendationEnv, LLMSimulatorEnv
from models.q_network import QNetwork
from models.world_model import WorldModel
from training.agent import DynaQAgent
from utils.visualizer import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
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

def train_baseline_agent(db_connector: DatabaseConnector, 
                         sample_applicant_ids: List[str],
                         num_episodes: int = TRAINING_CONFIG["num_episodes"],
                         use_llm_simulator: bool = False,
                         seed: int = 42) -> DynaQAgent:
    """
    Train a baseline agent without pretraining for a single applicant.
    
    Args:
        db_connector: Database connector for data retrieval.
        sample_applicant_ids: List of applicant IDs. Only the first ID will be used as the target.
        num_episodes: Number of training episodes.
        use_llm_simulator: Whether to use the LLM simulator for user feedback.
        seed: Random seed.
        
    Returns:
        DynaQAgent: Trained baseline agent.
    """
    # Set random seed
    set_seed(seed)
    
    # Select target applicant (first in the list)
    target_applicant_id = sample_applicant_ids[0]
    logger.info(f"Training baseline agent specialized for applicant: {target_applicant_id}")
    
    # Create environment
    if use_llm_simulator:
        env = LLMSimulatorEnv(db_connector=db_connector, random_seed=seed)
    else:
        env = JobRecommendationEnv(db_connector=db_connector, random_seed=seed)
    
    # Create agent
    agent = DynaQAgent(
        state_dim=MODEL_CONFIG["q_network"]["state_dim"],
        action_dim=MODEL_CONFIG["q_network"]["action_dim"],
        device=TRAINING_CONFIG["device"],
        target_applicant_id=target_applicant_id
    )
    
    # Train agent
    model_dir = os.path.join(os.path.dirname(__file__), PATH_CONFIG["model_dir"], "baseline")
    os.makedirs(model_dir, exist_ok=True)
    
    # Training loop
    metrics = agent.train(
        env=env,
        num_episodes=num_episodes,
        max_steps_per_episode=TRAINING_CONFIG["max_steps_per_episode"],
        applicant_ids=[target_applicant_id],  # Pass only the target applicant
        eval_frequency=TRAINING_CONFIG["eval_frequency"],
        save_frequency=TRAINING_CONFIG["save_frequency"],
        model_dir=model_dir
    )
    
    # Visualize training metrics
    visualizer = Visualizer()
    visualizer.plot_training_metrics(
        metrics=metrics,
        title=f"Baseline Agent Training Metrics (Applicant: {target_applicant_id})",
        filename=f"baseline_training_metrics_{target_applicant_id}.png"
    )
    
    return agent

def train_pretrained_agent(db_connector: DatabaseConnector,
                           sample_applicant_ids: List[str],
                           pretrained_model_path: str,
                           num_episodes: int = TRAINING_CONFIG["num_episodes"],
                           use_llm_simulator: bool = False,
                           seed: int = 43) -> DynaQAgent:
    """
    Train an agent with pretraining for a single applicant.
    
    Args:
        db_connector: Database connector for data retrieval.
        sample_applicant_ids: List of applicant IDs. Only the first ID will be used as the target.
        pretrained_model_path: Path to pretrained model.
        num_episodes: Number of training episodes.
        use_llm_simulator: Whether to use the LLM simulator for user feedback.
        seed: Random seed.
        
    Returns:
        DynaQAgent: Trained pretrained agent.
    """
    # Set random seed
    set_seed(seed)
    
    # Select target applicant (first in the list)
    target_applicant_id = sample_applicant_ids[0]
    logger.info(f"Training pretrained agent specialized for applicant: {target_applicant_id}")
    
    # Create environment
    if use_llm_simulator:
        env = LLMSimulatorEnv(db_connector=db_connector, random_seed=seed)
    else:
        env = JobRecommendationEnv(db_connector=db_connector, random_seed=seed)
    
    # Load pretrained models
    q_network = QNetwork.load(
        os.path.join(pretrained_model_path, "q_network.pt"),
        device=TRAINING_CONFIG["device"]
    )
    
    world_model = WorldModel.load(
        os.path.join(pretrained_model_path, "world_model.pt"),
        device=TRAINING_CONFIG["device"]
    )
    
    # Create agent with pretrained models
    agent = DynaQAgent(
        state_dim=q_network.state_dim,
        action_dim=q_network.action_dim,
        q_network=q_network,
        world_model=world_model,
        device=TRAINING_CONFIG["device"],
        target_applicant_id=target_applicant_id
    )
    
    # Train agent (fine-tuning)
    model_dir = os.path.join(os.path.dirname(__file__), PATH_CONFIG["model_dir"], "pretrained")
    os.makedirs(model_dir, exist_ok=True)
    
    # Fine-tuning with a smaller learning rate
    agent.lr = TRAINING_CONFIG["lr"] * 0.1
    agent.q_optimizer = torch.optim.Adam(agent.q_network.parameters(), lr=agent.lr)
    agent.world_optimizer = torch.optim.Adam(agent.world_model.parameters(), lr=agent.lr)
    
    # Training loop
    metrics = agent.train(
        env=env,
        num_episodes=num_episodes,
        max_steps_per_episode=TRAINING_CONFIG["max_steps_per_episode"],
        applicant_ids=[target_applicant_id],  # Pass only the target applicant
        eval_frequency=TRAINING_CONFIG["eval_frequency"],
        save_frequency=TRAINING_CONFIG["save_frequency"],
        model_dir=model_dir
    )
    
    # Visualize training metrics
    visualizer = Visualizer()
    visualizer.plot_training_metrics(
        metrics=metrics,
        title=f"Pretrained Agent Training Metrics (Applicant: {target_applicant_id})",
        filename=f"pretrained_training_metrics_{target_applicant_id}.png"
    )
    
    return agent

def compare_agents(baseline_agent: DynaQAgent, pretrained_agent: DynaQAgent,
                   db_connector: DatabaseConnector, sample_applicant_ids: List[str],
                   num_eval_episodes: int = 100) -> None:
    """
    Compare baseline and pretrained agents.
    
    Args:
        baseline_agent: Baseline agent.
        pretrained_agent: Pretrained agent.
        db_connector: Database connector for data retrieval.
        sample_applicant_ids: List of applicant IDs for evaluation.
        num_eval_episodes: Number of evaluation episodes.
    """
    # Create environment for evaluation (same for both agents)
    env = JobRecommendationEnv(db_connector=db_connector, random_seed=100)
    
    # Import evaluator here to avoid circular imports
    from utils.evaluator import Evaluator
    
    # Initialize evaluator
    evaluator = Evaluator()
    
    # Extract the target applicant IDs from the agents
    baseline_applicant_id = getattr(baseline_agent, "target_applicant_id", sample_applicant_ids[0])
    pretrained_applicant_id = getattr(pretrained_agent, "target_applicant_id", sample_applicant_ids[0])
    
    logger.info(f"Comparing agents: baseline (applicant: {baseline_applicant_id}) vs. pretrained (applicant: {pretrained_applicant_id})")
    
    # Compare agents using their respective target applicants
    comparison = evaluator.compare_agents(
        baseline_agent=baseline_agent,
        pretrained_agent=pretrained_agent,
        env=env,
        applicant_ids=[baseline_applicant_id, pretrained_applicant_id],
        num_episodes=num_eval_episodes
    )
    
    # Log comparison results
    logger.info(f"Comparison Results: {comparison['improvements']}")

def main(args: argparse.Namespace) -> None:
    """
    Main function for training and evaluating agents.
    
    Args:
        args: Command-line arguments.
    """
    # Initialize database connector
    db_connector = DatabaseConnector()
    
    # Get sample applicant IDs (simplified - in practice, would fetch from DB)
    # In a real scenario, you would query your MongoDB for a list of applicant IDs
    # For now, we use placeholder IDs
    sample_applicant_ids = [f"applicant_{i}" for i in range(100)]
    
    # Train baseline agent
    if args.train_baseline:
        logger.info("Training baseline agent...")
        baseline_agent = train_baseline_agent(
            db_connector=db_connector,
            sample_applicant_ids=sample_applicant_ids,
            num_episodes=args.num_episodes,
            use_llm_simulator=args.use_llm_simulator,
            seed=args.seed
        )
    else:
        # Load existing baseline agent
        baseline_path = os.path.join(os.path.dirname(__file__), PATH_CONFIG["model_dir"], "baseline")
        logger.info(f"Loading baseline agent from {baseline_path}...")
        baseline_agent = DynaQAgent.load_models(baseline_path, args.episode_to_load, TRAINING_CONFIG["device"])
    
    # Train or load pretrained agent
    if args.train_pretrained:
        if not args.pretrained_model_path:
            logger.error("Pretrained model path must be provided to train pretrained agent")
            return
            
        logger.info("Training pretrained agent...")
        pretrained_agent = train_pretrained_agent(
            db_connector=db_connector,
            sample_applicant_ids=sample_applicant_ids,
            pretrained_model_path=args.pretrained_model_path,
            num_episodes=args.num_episodes,
            use_llm_simulator=args.use_llm_simulator,
            seed=args.seed + 1
        )
    else:
        # Load existing pretrained agent
        pretrained_path = os.path.join(os.path.dirname(__file__), PATH_CONFIG["model_dir"], "pretrained")
        logger.info(f"Loading pretrained agent from {pretrained_path}...")
        pretrained_agent = DynaQAgent.load_models(pretrained_path, args.episode_to_load, TRAINING_CONFIG["device"])
    
    # Compare agents
    if args.compare:
        logger.info("Comparing agents...")
        compare_agents(
            baseline_agent=baseline_agent,
            pretrained_agent=pretrained_agent,
            db_connector=db_connector,
            sample_applicant_ids=sample_applicant_ids,
            num_eval_episodes=args.num_eval_episodes
        )
    
    # Close database connection
    db_connector.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Dyna-Q job recommender agents")
    
    # Training arguments
    parser.add_argument("--train_baseline", action="store_true", help="Train baseline agent")
    parser.add_argument("--train_pretrained", action="store_true", help="Train pretrained agent")
    parser.add_argument("--pretrained_model_path", type=str, help="Path to pretrained models")
    parser.add_argument("--num_episodes", type=int, default=TRAINING_CONFIG["num_episodes"], help="Number of training episodes")
    parser.add_argument("--use_llm_simulator", action="store_true", help="Use LLM simulator for training")
    
    # Evaluation arguments
    parser.add_argument("--compare", action="store_true", help="Compare baseline and pretrained agents")
    parser.add_argument("--num_eval_episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--episode_to_load", type=int, default=TRAINING_CONFIG["num_episodes"], help="Episode number to load models from")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    main(args) 