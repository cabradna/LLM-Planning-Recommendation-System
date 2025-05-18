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
from config.config import MODEL_CONFIG, TRAINING_CONFIG, PATH_CONFIG, STRATEGY_CONFIG, ENV_CONFIG

# Import modules
from data.database import DatabaseConnector
from data.tensor_cache import TensorCache
from environments.job_env import JobRecommendationEnv, LLMSimulatorEnv, HybridEnv
from models.q_network import QNetwork
from models.world_model import WorldModel
from training.agent import DynaQAgent
from utils.visualizer import Visualizer
from utils.evaluator import Evaluator

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
                         tensor_cache: TensorCache,
                         target_applicant_id: str,
                         strategy: str = "cosine",
                         num_episodes: int = TRAINING_CONFIG["num_episodes"],
                         max_steps_per_episode: int = TRAINING_CONFIG["max_steps_per_episode"],
                         learning_rate: float = TRAINING_CONFIG["lr"],
                         batch_size: int = TRAINING_CONFIG["batch_size"],
                         gamma: float = TRAINING_CONFIG["gamma"],
                         planning_steps: int = TRAINING_CONFIG["planning_steps"],
                         eval_frequency: int = TRAINING_CONFIG["eval_frequency"],
                         num_eval_episodes: int = TRAINING_CONFIG["num_eval_episodes"],
                         device: str = TRAINING_CONFIG["device"],
                         save_path: Optional[str] = None,
                         cosine_weight: float = STRATEGY_CONFIG["hybrid"]["initial_cosine_weight"],
                         final_cosine_weight: float = STRATEGY_CONFIG["hybrid"]["final_cosine_weight"],
                         annealing_episodes: int = STRATEGY_CONFIG["hybrid"]["annealing_episodes"],
                         seed: int = ENV_CONFIG["random_seed"]) -> DynaQAgent:
    """
    Train a baseline agent without pretraining for a single applicant.
    
    Args:
        db_connector: Database connector for data retrieval.
        tensor_cache: Cache for tensors.
        target_applicant_id: Target applicant ID.
        strategy: Training strategy (cosine, llm, hybrid).
        num_episodes: Number of training episodes.
        max_steps_per_episode: Maximum steps per episode.
        learning_rate: Learning rate for training.
        batch_size: Batch size for training.
        gamma: Discount factor.
        planning_steps: Number of planning steps per real step.
        eval_frequency: Frequency of evaluation during training.
        num_eval_episodes: Number of episodes for evaluation during training.
        device: Device to run training on.
        save_path: Path to save trained agent.
        cosine_weight: Initial cosine weight for hybrid strategy.
        final_cosine_weight: Final cosine weight for hybrid strategy.
        annealing_episodes: Number of episodes to anneal cosine weight in hybrid strategy.
        seed: Random seed.
        
    Returns:
        DynaQAgent: Trained baseline agent.
    """
    # Set random seed
    set_seed(seed)
    device = torch.device(device)
    
    logger.info(f"Training baseline agent with '{strategy}' strategy for applicant: {target_applicant_id}")
    
    # Create environment based on strategy
    if strategy == "cosine":
        env = JobRecommendationEnv(
            tensor_cache=tensor_cache,
            reward_scheme=STRATEGY_CONFIG["llm"]["response_mapping"],
            reward_strategy="cosine",
            random_seed=seed
        )
        logger.info("Created JobRecommendationEnv with cosine reward strategy")
    elif strategy == "llm":
        env = LLMSimulatorEnv(
            tensor_cache=tensor_cache,
            reward_scheme=STRATEGY_CONFIG["llm"]["response_mapping"],
            random_seed=seed
        )
        
        # Setup LLM
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
            raise
            
        logger.info("Created LLMSimulatorEnv with LLM reward strategy")
    elif strategy == "hybrid":
        env = HybridEnv(
            tensor_cache=tensor_cache,
            reward_scheme=STRATEGY_CONFIG["llm"]["response_mapping"],
            cosine_weight=cosine_weight,
            random_seed=seed
        )
        
        # Setup LLM
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
            raise
            
        logger.info("Created HybridEnv with hybrid reward strategy")
    else:
        logger.error(f"Invalid strategy: {strategy}")
        raise ValueError(f"Invalid strategy: {strategy}")
    
    # Create agent
    agent = DynaQAgent(
        state_dim=MODEL_CONFIG["q_network"]["state_dim"],
        action_dim=MODEL_CONFIG["q_network"]["action_dim"],
        device=device,
        target_applicant_id=target_applicant_id,
        tensor_cache=tensor_cache,
        training_strategy=strategy
    )
    
    # Configure agent parameters
    agent.lr = learning_rate
    agent.batch_size = batch_size
    agent.gamma = gamma
    agent.planning_steps = planning_steps
    
    if strategy == "hybrid":
        agent.cosine_weight = cosine_weight
        agent.final_cosine_weight = final_cosine_weight
        agent.annealing_episodes = annealing_episodes
    
    # Train agent
    model_dir = save_path or os.path.join(PATH_CONFIG["model_dir"], "baseline")
    os.makedirs(model_dir, exist_ok=True)
    
    # Define evaluator
    evaluator = Evaluator()
    
    # Training loop
    metrics = agent.train(
        env=env,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        applicant_ids=[target_applicant_id],  # Pass only the target applicant
        eval_frequency=eval_frequency,
        save_frequency=eval_frequency,
        model_dir=model_dir,
        evaluator=evaluator,
        num_eval_episodes=num_eval_episodes
    )
    
    # Visualize training metrics
    visualizer = Visualizer()
    visualizer.plot_training_metrics(
        metrics={
            'q_losses': metrics.get('q_network_loss', []),
            'world_losses': metrics.get('world_model_loss', []),
            'episode_rewards': metrics.get('episode_reward', [])
        },
        title=f"Baseline Agent Training Metrics (Applicant: {target_applicant_id}, Strategy: {strategy})"
    )
    
    return agent

def train_pretrained_agent(db_connector: DatabaseConnector,
                           tensor_cache: TensorCache,
                           target_applicant_id: str,
                           pretrained_model_path: str,
                           strategy: str = "hybrid",
                           num_episodes: int = TRAINING_CONFIG["num_episodes"],
                           max_steps_per_episode: int = TRAINING_CONFIG["max_steps_per_episode"],
                           learning_rate: float = TRAINING_CONFIG["lr"] * 0.1,  # Lower LR for fine-tuning
                           batch_size: int = TRAINING_CONFIG["batch_size"],
                           gamma: float = TRAINING_CONFIG["gamma"],
                           planning_steps: int = TRAINING_CONFIG["planning_steps"],
                           eval_frequency: int = TRAINING_CONFIG["eval_frequency"],
                           num_eval_episodes: int = TRAINING_CONFIG["num_eval_episodes"],
                           device: str = TRAINING_CONFIG["device"],
                           save_path: Optional[str] = None,
                           cosine_weight: float = STRATEGY_CONFIG["hybrid"]["initial_cosine_weight"],
                           final_cosine_weight: float = STRATEGY_CONFIG["hybrid"]["final_cosine_weight"],
                           annealing_episodes: int = STRATEGY_CONFIG["hybrid"]["annealing_episodes"],
                           seed: int = ENV_CONFIG["random_seed"] + 1) -> DynaQAgent:
    """
    Train an agent with pretraining for a single applicant.
    
    Args:
        db_connector: Database connector for data retrieval.
        tensor_cache: Cache for tensors.
        target_applicant_id: Target applicant ID.
        pretrained_model_path: Path to pretrained model.
        strategy: Training strategy (cosine, llm, hybrid).
        num_episodes: Number of training episodes.
        max_steps_per_episode: Maximum steps per episode.
        learning_rate: Learning rate for training.
        batch_size: Batch size for training.
        gamma: Discount factor.
        planning_steps: Number of planning steps per real step.
        eval_frequency: Frequency of evaluation during training.
        num_eval_episodes: Number of episodes for evaluation during training.
        device: Device to run training on.
        save_path: Path to save trained agent.
        cosine_weight: Initial cosine weight for hybrid strategy.
        final_cosine_weight: Final cosine weight for hybrid strategy.
        annealing_episodes: Number of episodes to anneal cosine weight in hybrid strategy.
        seed: Random seed.
        
    Returns:
        DynaQAgent: Trained pretrained agent.
    """
    # Set random seed
    set_seed(seed)
    device = torch.device(device)
    
    logger.info(f"Training pretrained agent with '{strategy}' strategy for applicant: {target_applicant_id}")
    
    # Create environment based on strategy
    if strategy == "cosine":
        env = JobRecommendationEnv(
            tensor_cache=tensor_cache,
            reward_scheme=STRATEGY_CONFIG["llm"]["response_mapping"],
            reward_strategy="cosine",
            random_seed=seed
        )
        logger.info("Created JobRecommendationEnv with cosine reward strategy")
    elif strategy == "llm":
        env = LLMSimulatorEnv(
            tensor_cache=tensor_cache,
            reward_scheme=STRATEGY_CONFIG["llm"]["response_mapping"],
            random_seed=seed
        )
        
        # Setup LLM
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
            raise
            
        logger.info("Created LLMSimulatorEnv with LLM reward strategy")
    elif strategy == "hybrid":
        env = HybridEnv(
            tensor_cache=tensor_cache,
            reward_scheme=STRATEGY_CONFIG["llm"]["response_mapping"],
            cosine_weight=cosine_weight,
            random_seed=seed
        )
        
        # Setup LLM
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
            raise
            
        logger.info("Created HybridEnv with hybrid reward strategy")
    else:
        logger.error(f"Invalid strategy: {strategy}")
        raise ValueError(f"Invalid strategy: {strategy}")
    
    # Load pretrained models
    try:
        logger.info(f"Loading pretrained models from {pretrained_model_path}")
        q_network_file = os.path.join(pretrained_model_path, "q_network_pretrained.pt")
        world_model_file = os.path.join(pretrained_model_path, "world_model_pretrained.pt")
        
        if not os.path.exists(q_network_file) or not os.path.exists(world_model_file):
            logger.warning(f"Pretrained model files not found at {pretrained_model_path}")
            logger.warning("Checking for standard model filenames instead...")
            q_network_file = os.path.join(pretrained_model_path, "q_network.pt")
            world_model_file = os.path.join(pretrained_model_path, "world_model.pt")
        
        # Initialize models
        q_network = QNetwork(
            state_dim=MODEL_CONFIG["q_network"]["state_dim"],
            action_dim=MODEL_CONFIG["q_network"]["action_dim"],
            hidden_dims=MODEL_CONFIG["q_network"]["hidden_dims"],
            dropout_rate=MODEL_CONFIG["q_network"]["dropout_rate"],
            activation=MODEL_CONFIG["q_network"]["activation"]
        ).to(device)
        
        world_model = WorldModel(
            input_dim=MODEL_CONFIG["world_model"]["input_dim"],
            state_dim=MODEL_CONFIG["q_network"]["state_dim"],
            hidden_dims=MODEL_CONFIG["world_model"]["hidden_dims"],
            dropout_rate=MODEL_CONFIG["world_model"]["dropout_rate"],
            activation=MODEL_CONFIG["world_model"]["activation"]
        ).to(device)
        
        # Load state dictionaries
        q_network.load_state_dict(torch.load(q_network_file, map_location=device))
        world_model.load_state_dict(torch.load(world_model_file, map_location=device))
        
        logger.info("Pretrained models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading pretrained models: {e}")
        raise
    
    # Create agent with pretrained models
    agent = DynaQAgent(
        state_dim=q_network.state_dim,
        action_dim=q_network.action_dim,
        q_network=q_network,
        world_model=world_model,
        device=device,
        target_applicant_id=target_applicant_id,
        tensor_cache=tensor_cache,
        training_strategy=strategy
    )
    
    # Configure agent parameters
    agent.lr = learning_rate
    agent.batch_size = batch_size
    agent.gamma = gamma
    agent.planning_steps = planning_steps
    
    # Initialize optimizers with specified learning rate
    agent.q_optimizer = torch.optim.Adam(agent.q_network.parameters(), lr=agent.lr)
    agent.world_optimizer = torch.optim.Adam(agent.world_model.parameters(), lr=agent.lr)
    
    if strategy == "hybrid":
        agent.cosine_weight = cosine_weight
        agent.final_cosine_weight = final_cosine_weight
        agent.annealing_episodes = annealing_episodes
    
    # Train agent (fine-tuning)
    model_dir = save_path or os.path.join(PATH_CONFIG["model_dir"], "pretrained")
    os.makedirs(model_dir, exist_ok=True)
    
    # Define evaluator
    evaluator = Evaluator()
    
    # Training loop
    metrics = agent.train(
        env=env,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        applicant_ids=[target_applicant_id],  # Pass only the target applicant
        eval_frequency=eval_frequency,
        save_frequency=eval_frequency,
        model_dir=model_dir,
        evaluator=evaluator,
        num_eval_episodes=num_eval_episodes
    )
    
    # Visualize training metrics
    visualizer = Visualizer()
    visualizer.plot_training_metrics(
        metrics={
            'q_losses': metrics.get('q_network_loss', []),
            'world_losses': metrics.get('world_model_loss', []),
            'episode_rewards': metrics.get('episode_reward', [])
        },
        title=f"Pretrained Agent Training Metrics (Applicant: {target_applicant_id}, Strategy: {strategy})"
    )
    
    return agent

def compare_agents(baseline_agent: DynaQAgent, pretrained_agent: DynaQAgent,
                   db_connector: DatabaseConnector, tensor_cache: TensorCache, target_applicant_id: str,
                   num_eval_episodes: int = TRAINING_CONFIG["num_eval_episodes"],
                   strategy: str = "hybrid", seed: int = ENV_CONFIG["random_seed"]) -> None:
    """
    Compare baseline and pretrained agents.
    
    Args:
        baseline_agent: Baseline agent.
        pretrained_agent: Pretrained agent.
        db_connector: Database connector for data retrieval.
        tensor_cache: Cache for tensors.
        target_applicant_id: Target applicant ID.
        num_eval_episodes: Number of evaluation episodes.
        strategy: Reward strategy for evaluation.
        seed: Random seed.
    """
    # Create environment for evaluation (same for both agents)
    if strategy == "cosine":
        env = JobRecommendationEnv(
            tensor_cache=tensor_cache,
            reward_scheme=STRATEGY_CONFIG["llm"]["response_mapping"],
            reward_strategy="cosine",
            random_seed=seed
        )
    elif strategy == "llm":
        env = LLMSimulatorEnv(
            tensor_cache=tensor_cache,
            reward_scheme=STRATEGY_CONFIG["llm"]["response_mapping"],
            random_seed=seed
        )
        
        # Setup LLM if not already set up
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
            raise
    elif strategy == "hybrid":
        env = HybridEnv(
            tensor_cache=tensor_cache,
            reward_scheme=STRATEGY_CONFIG["llm"]["response_mapping"],
            cosine_weight=STRATEGY_CONFIG["hybrid"]["initial_cosine_weight"],
            random_seed=seed
        )
        
        # Setup LLM if not already set up
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
            raise
    else:
        logger.error(f"Invalid strategy: {strategy}")
        raise ValueError(f"Invalid strategy: {strategy}")
    
    # Initialize evaluator
    evaluator = Evaluator()
    
    # Compare agents
    comparison = evaluator.compare_agents(
        baseline_agent=baseline_agent,
        pretrained_agent=pretrained_agent,
        env=env,
        applicant_ids=[target_applicant_id],
        num_episodes=num_eval_episodes
    )
    
    # Log comparison results
    logger.info("Comparison Results:")
    logger.info(f"  Reward improvement: {comparison['improvements']['avg_reward']:.2f}%")
    logger.info(f"  Apply rate improvement: {comparison['improvements']['apply_rate']:.2f}%")
    
    # Visualize comparison
    visualizer = Visualizer()
    visualizer.plot_evaluation_results(
        baseline_rewards=comparison['baseline']['episode_rewards'],
        pretrained_rewards=comparison['pretrained']['episode_rewards'],
        title="Comparison: Baseline vs. Pretrained Agent"
    )

def main(args: argparse.Namespace) -> None:
    """
    Main function for training and evaluating agents.
    
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
    
    # Get applicant IDs
    try:
        if args.applicant_id:
            # Use the specified applicant ID
            applicant_ids = [args.applicant_id]
            target_applicant_id = args.applicant_id
            logger.info(f"Using specified applicant ID: {target_applicant_id}")
        else:
            # Fetch a list of applicant IDs from the database
            collection = db_connector.db[db_connector.collections["candidates_text"]]
            applicant_ids = [doc["_id"] for doc in collection.find({}, {"_id": 1}).limit(args.num_applicants)]
            if not applicant_ids:
                logger.error("No applicant IDs found in the database")
                raise ValueError("No applicant IDs found in the database")
            target_applicant_id = applicant_ids[0]
            logger.info(f"Selected target applicant ID: {target_applicant_id}")
    except Exception as e:
        logger.error(f"Error selecting target applicant: {e}")
        raise
    
    # Copy data from database to TensorCache
    tensor_cache.copy_from_database(
        db_connector=db_connector,
        applicant_ids=applicant_ids
    )
    stats = tensor_cache.cache_stats()
    logger.info(f"Cache initialized with {stats['job_count']} jobs and {stats['applicant_state_count']} applicant states")
    
    # Train baseline agent
    if args.train_baseline:
        logger.info(f"Training baseline agent with strategy: {args.strategy}")
        baseline_agent = train_baseline_agent(
            db_connector=db_connector,
            tensor_cache=tensor_cache,
            target_applicant_id=target_applicant_id,
            strategy=args.strategy,
            num_episodes=args.num_episodes,
            max_steps_per_episode=args.max_steps_per_episode,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=args.gamma,
            planning_steps=args.planning_steps,
            eval_frequency=args.eval_frequency,
            num_eval_episodes=args.num_eval_episodes,
            device=args.device,
            save_path=args.save_path,
            cosine_weight=args.cosine_weight,
            final_cosine_weight=args.final_cosine_weight,
            annealing_episodes=args.annealing_episodes,
            seed=args.seed
        )
    else:
        # If not training baseline, we may still need to load it for comparison
        logger.info("Skipping baseline agent training")
        baseline_agent = None
    
    # Train or load pretrained agent
    if args.train_pretrained:
        if not args.pretrained_model_path:
            logger.error("Pretrained model path must be provided to train pretrained agent")
            return
            
        logger.info(f"Training pretrained agent with strategy: {args.strategy}")
        pretrained_agent = train_pretrained_agent(
            db_connector=db_connector,
            tensor_cache=tensor_cache,
            target_applicant_id=target_applicant_id,
            pretrained_model_path=args.pretrained_model_path,
            strategy=args.strategy,
            num_episodes=args.num_episodes,
            max_steps_per_episode=args.max_steps_per_episode,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=args.gamma,
            planning_steps=args.planning_steps,
            eval_frequency=args.eval_frequency,
            num_eval_episodes=args.num_eval_episodes,
            device=args.device,
            save_path=args.save_path,
            cosine_weight=args.cosine_weight,
            final_cosine_weight=args.final_cosine_weight,
            annealing_episodes=args.annealing_episodes,
            seed=args.seed + 1  # Use a different seed for the pretrained agent
        )
    else:
        # If not training pretrained, we may still need to load it for comparison
        logger.info("Skipping pretrained agent training")
        pretrained_agent = None
    
    # Compare agents if both are available
    if baseline_agent and pretrained_agent and args.compare:
        logger.info("Comparing agents...")
        compare_agents(
            baseline_agent=baseline_agent,
            pretrained_agent=pretrained_agent,
            db_connector=db_connector,
            tensor_cache=tensor_cache,
            target_applicant_id=target_applicant_id,
            num_eval_episodes=args.num_eval_episodes,
            strategy=args.strategy,
            seed=args.seed
        )
    
    # Clean up resources
    db_connector.close()
    
    if tensor_cache is not None and hasattr(tensor_cache, 'clear'):
        tensor_cache.clear()
        logger.info("Tensor cache cleared")
        
    # Free PyTorch memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache emptied")
    
    logger.info("Training complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Dyna-Q job recommender agents")
    
    # Training arguments
    parser.add_argument("--train_baseline", action="store_true", help="Train baseline agent")
    parser.add_argument("--train_pretrained", action="store_true", help="Train pretrained agent")
    parser.add_argument("--pretrained_model_path", type=str, help="Path to pretrained models")
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
    
    # Evaluation parameters
    parser.add_argument("--compare", action="store_true", help="Compare baseline and pretrained agents")
    parser.add_argument("--eval_frequency", type=int, default=TRAINING_CONFIG["eval_frequency"],
                      help="Frequency of evaluation during training")
    parser.add_argument("--num_eval_episodes", type=int, default=TRAINING_CONFIG["num_eval_episodes"],
                      help="Number of episodes for evaluation during training")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=ENV_CONFIG["random_seed"], help="Random seed")
    parser.add_argument("--device", type=str, default=TRAINING_CONFIG["device"], help="Device to use for training")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save trained models")
    parser.add_argument("--num_applicants", type=int, default=TRAINING_CONFIG["num_candidates"],
                      help="Number of applicants to use for training")
    
    args = parser.parse_args()
    
    # If no training option specified, train both
    if not (args.train_baseline or args.train_pretrained):
        args.train_baseline = True
        args.train_pretrained = True
    
    main(args) 