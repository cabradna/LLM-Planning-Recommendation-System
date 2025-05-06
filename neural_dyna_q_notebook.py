#!/usr/bin/env python
# coding: utf-8

# %% [markdown]
# # Neural Dyna-Q for Job Recommendation: Model-Based RL Approach
# 
# **Author:** LLM Lab Team  
# **Date:** 2023-10-15

# %% [markdown]
# ## 1. Introduction and Overview
# 
# This notebook demonstrates a Neural Dyna-Q approach to personalized job recommendation. By combining deep reinforcement learning with model-based planning, we build a system that learns to recommend jobs that match candidate skills and preferences.
# 
# ### 1.1 Core Algorithm Concepts
# 
# The Neural Dyna-Q algorithm integrates these key components:
# 
# - **Q-Learning**: Learning a value function that predicts the utility of job recommendations
# - **World Model**: A neural network that models the environment dynamics
# - **Planning**: Using the world model to simulate experiences and improve the value function
# - **Exploration Strategy**: Balancing exploration and exploitation with epsilon-greedy policy
# 
# ### 1.2 Key Features of Implementation
# 
# - **Single-Applicant Specialization**: Each agent is trained for one specific applicant
# - **Neural Networks**: Deep learning for value function and environment dynamics approximation
# - **Multiple Reward Generation Strategies**: Cosine similarity, LLM feedback, and hybrid approaches
# - **Dyna-Q Algorithm**: Combines direct RL with model-based planning
# 
# ### 1.3 Expected Outcomes
# 
# By the end of this notebook, we'll have:
# 1. A trained job recommendation agent for a specific candidate
# 2. Visualizations of the learning process
# 3. A list of personalized job recommendations
# 4. Insights into the model's performance

# %% [markdown]
# ## 2. Environment Setup
# 
# First, we'll set up our environment, install necessary packages, and import libraries. If running in Google Colab, we'll clone the repository and install dependencies.

# %% 
# Check if running in Google Colab
# Import necessary libraries for setup
import os
import sys
import subprocess
from pathlib import Path

# Configuration settings for the setup process, particularly for Colab environments.
SETUP_PATH_CONFIG = {
    "repo_url": "https://github.com/cabradna/LLM-Planning-Recommendation-System.git",  # URL of the repository to clone.
    "branch": "local_tensors"  # Specific branch to use for tensor-based implementation
}

# Configuration for enabling specific strategies during setup (e.g., LLM, hybrid).
SETUP_STRATEGY_CONFIG = {
    "llm": {"enabled": True},  # Enable LLM-related dependency installation.
    "hybrid": {"enabled": False} # Enable hybrid strategy dependency installation.
}

# Determine if the code is running in Google Colab.
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("Running in Google Colab. Cloning repository and installing dependencies...")

    try:
        # Retrieve repository URL from configuration.
        repo_url = SETUP_PATH_CONFIG["repo_url"]
        repo_name = Path(repo_url).stem  # Extract repository name from the URL.

        # Clone the repository.
        print(f"Cloning repository: {repo_url}")
        subprocess.run(["git", "clone", "-q", repo_url], check=True)  # Clone quietly and raise an error if cloning fails.

        # Navigate into the cloned repository directory.
        cloned_repo_path = Path(repo_name)
        if cloned_repo_path.is_dir():
            os.chdir(cloned_repo_path)
            print(f"Changed directory to: {os.getcwd()}")
            
            # Checkout the specific branch
            branch = SETUP_PATH_CONFIG["branch"]
            print(f"Checking out branch: {branch}")
            subprocess.run(["git", "checkout", branch], check=True)
        else:
            raise FileNotFoundError(f"Cloned repository directory '{repo_name}' not found or is not a directory.")

        # Install base requirements from requirements.txt.
        requirements_path = Path("requirements.txt")
        if requirements_path.is_file():
            print("Installing base requirements from requirements.txt...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_path)], check=True)  # Use the correct pip and install quietly.
        else:
            print("Warning: requirements.txt not found. Skipping base requirements installation.")

        # Install the project package in editable mode.
        print("Attempting to install project package in editable mode...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)  # Install the current package in editable mode.

        # Install extra packages if LLM or hybrid strategy is enabled.
        if SETUP_STRATEGY_CONFIG["llm"]["enabled"] or SETUP_STRATEGY_CONFIG["hybrid"]["enabled"]:
            print("Installing LLM-related packages (transformers, accelerate, bitsandbytes)...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "accelerate"], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.45.0"], check=True)
            print("LLM-related packages installed successfully.")

        print("Repository cloned and dependencies installed successfully.")

        # Add the project root to the Python path.
        project_root = Path(os.getcwd()).absolute()
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))  # Insert at the beginning to prioritize project modules.
            print(f"Added {project_root} to sys.path")

    except subprocess.CalledProcessError as e:
        print(f"Setup command failed: {e}")
        print("Please check the repository URL, requirements file, setup.py, and Colab environment.")
        raise  # Re-raise the exception to halt execution.
    except FileNotFoundError as e:
        print(f"Setup error: {e}")
        raise  # Re-raise the exception to halt execution.
else:
    print("Not running in Google Colab. Assuming local environment is set up.")

    try:
        # Determine the project root directory.
        project_root = Path(__file__).resolve().parent.parent
    except NameError:
        # Handle cases where __file__ is not defined (e.g., interactive environments).
        project_root = Path('.').absolute()
        print(f"Warning: __file__ not defined. Assuming project root is CWD: {project_root}")

    # Add the project root to the Python path if it's not already there.
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"Added {project_root} to sys.path for local execution.")
        
    # Verify we're on the correct branch
    try:
        result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
        current_branch = result.stdout.strip()
        if current_branch != SETUP_PATH_CONFIG["branch"]:
            print(f"Warning: Current branch is '{current_branch}', but code expects '{SETUP_PATH_CONFIG['branch']}'")
            print("Please switch to the correct branch for this implementation.")
    except Exception as e:
        print(f"Warning: Could not verify git branch: {e}")


# %%
# Standard imports for data manipulation, visualization, and deep learning
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm.notebook import tqdm  # Use tqdm.notebook for Colab/notebook progress bars
import torch
from pathlib import Path

# --- Configure Logging --- 
import logging
from config.config import PATH_CONFIG # Assuming PATH_CONFIG is defined here

# Define log levels
file_log_level = logging.DEBUG  # Always log DEBUG and above to file
initial_stream_log_level = logging.INFO # Initial level for notebook output

# Ensure log directory exists
log_dir = PATH_CONFIG.get("log_dir", "logs") # Default to ./logs if not in config
if not os.path.isabs(log_dir):
    try:
        project_root_for_log = Path(__file__).resolve().parent.parent
    except NameError:
        project_root_for_log = Path('.').absolute()
    log_dir = os.path.join(project_root_for_log, log_dir)
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'notebook_run.log')

# Get root logger and set level to lowest required (DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG) # Allow all messages >= DEBUG through the root logger

# Remove existing handlers from previous runs (if any)
# This is important if re-running this cell in the notebook
for handler in logger.handlers[:]:
   logger.removeHandler(handler)

# Create formatter
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create File Handler (Level: DEBUG)
file_handler = logging.FileHandler(log_file_path, mode='w') # 'w' to overwrite each run
file_handler.setFormatter(log_formatter)
file_handler.setLevel(file_log_level) # Log DEBUG and up to the file
logger.addHandler(file_handler)

# Create Stream Handler (Level: INFO initially)
# Store the handler instance in a variable so its level can be changed later
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(initial_stream_log_level) # Log INFO and up to the console/notebook
logger.addHandler(stream_handler)

logger.info(f"Logging configured. Root Level: DEBUG. File Level: {logging.getLevelName(file_log_level)}. Stream Level: {logging.getLevelName(initial_stream_log_level)}. File: {log_file_path}")
logger.debug("DEBUG logging is enabled for the root logger and file handler.") # This will appear in file only (unless stream level is changed)

# --- How to change Stream Handler level later --- 
# In a later cell, to see DEBUG messages in notebook output:
# stream_handler.setLevel(logging.DEBUG)
# logger.info("--- Changed Stream Handler level to DEBUG ---")

# To switch Stream Handler back to INFO:
# stream_handler.setLevel(logging.INFO)
# logger.info("--- Changed Stream Handler level back to INFO ---")
# --- End Logging Configuration ---

# Dynamically add the project root to the Python path
# This allows importing modules from the project's source code, regardless of the current working directory.
try:
    # Determine the project root directory based on the location of this script.
    # Assumes the script is located in a subdirectory of the project root.
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    # Handle cases where __file__ is not defined, such as in interactive environments.
    project_root = Path('.').absolute()

# Add the project root to the Python path if it's not already there.
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))  # Prioritize project modules over installed ones

# Import configuration settings from the project's config module
# These settings define various aspects of the experiment, such as database connections,
# model architectures, training parameters, and evaluation metrics.
from config.config import (
    DB_CONFIG,
    MODEL_CONFIG,
    TRAINING_CONFIG,
    STRATEGY_CONFIG,
    PATH_CONFIG,
    EVAL_CONFIG,
    HF_CONFIG,
    ENV_CONFIG,
    PRETRAINING_CONFIG
)

# Set random seeds for reproducibility
# This ensures that the experiment produces consistent results across multiple runs,
# which is essential for debugging and comparing different approaches.
random.seed(ENV_CONFIG["random_seed"])
np.random.seed(ENV_CONFIG["random_seed"])
torch.manual_seed(ENV_CONFIG["random_seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(ENV_CONFIG["random_seed"])

# Determine the device to use for PyTorch computations (CPU or GPU)
# If a GPU is available, it will be used to accelerate training.
device = torch.device(TRAINING_CONFIG["device"])
print(f"Using device: {device}")

# Import project-specific modules
# These modules contain the implementations of the various components of the system,
# such as data loading, environment interaction, model architectures, and training algorithms.
from src.data.database import DatabaseConnector
from src.data.data_loader import JobRecommendationDataset, ReplayBuffer
from src.environments.job_env import JobRecommendationEnv, LLMSimulatorEnv, HybridEnv
from src.models.q_network import QNetwork
from src.models.world_model import WorldModel
from src.training.agent import DynaQAgent
from src.utils.visualizer import Visualizer
from src.utils.evaluator import Evaluator
from src.data.tensor_cache import TensorCache

print("Project modules imported successfully.")
# %% [markdown]
# ## 3. Database Connection
# 
# The system connects to a MongoDB Atlas database containing job postings and candidate information. The database includes collections for:
# - Job text data (descriptions, requirements, etc.)
# - Job embedding vectors (semantic representations)
# - Candidate profiles
# - Candidate embedding vectors
# 
# The connection is established using environment variables for credentials.

# %%
# Get reward strategy settings from config
if not STRATEGY_CONFIG["hybrid"]["enabled"]:
    raise ValueError("Hybrid strategy is not enabled in config")

reward_strategy = "hybrid"  # Default to hybrid strategy
cosine_weight = STRATEGY_CONFIG["hybrid"]["initial_cosine_weight"]

# Initialize database connector
try:
    db_connector = DatabaseConnector()  # Uses default connection string and db name from config
    print("Database connection established successfully.")
except Exception as e:
    print(f"Database connection error: {e}")
    raise

# %% [markdown]
# ## 4. Data Loading, Cache Init, Agent/Env Init, and First Reset
#
# This combined cell handles:
# 1. Selecting the target candidate ID
# 2. Initializing the TensorCache and loading data from DB
# 3. Initializing the appropriate Environment using the cache
# 4. Initializing the QNetwork, WorldModel, and DynaQAgent using the cache
# 5. Initializing Visualizer and Evaluator
# 6. Performing the first environment reset for the target candidate

# %% 
# Get target candidate ID
try:
    collection = db_connector.db[db_connector.collections["candidates_text"]]
    candidate_ids = [doc["_id"] for doc in collection.find({}, {"_id": 1}).limit(TRAINING_CONFIG["num_candidates"])]
    target_candidate_id = candidate_ids[0]
    print(f"Selected target candidate ID: {target_candidate_id}")
except Exception as e:
    print(f"Error selecting target candidate: {e}")
    raise

# Initialize tensor cache
CACHE_CONFIG = {
    "enabled": True,
    "device": TRAINING_CONFIG.get("device", "cuda"),
    "load_all_jobs": True
}

tensor_cache = None # Define outside the if block for broader scope if needed later
env = None
# agent_cosine and agent_hybrid will be defined in their respective training blocks.
# The initial agent definition here is mostly for confirming setup and can be streamlined.

if CACHE_CONFIG["enabled"]:
    print(f"Initializing tensor cache on {CACHE_CONFIG['device']}...")
    tensor_cache = TensorCache(device=CACHE_CONFIG["device"])
    tensor_cache.copy_from_database(
        db_connector=db_connector,
        applicant_ids=[target_candidate_id]
    )
    stats = tensor_cache.cache_stats()
    # Use the specific counts from stats
    print(f"Cache initialized with {stats['job_count']} jobs and {stats['applicant_state_count']} applicant states.") 
    print(f"Initialization time: {stats['initialization_time']:.2f} seconds")
    print(f"Memory device: {stats['device']}")

    # Create environment with cache (NO db_connector)
    # This env will be used by agent_hybrid. agent_cosine.pretrain uses data directly.
    if reward_strategy == "cosine": # This path might be less used if hybrid is the focus
        env = JobRecommendationEnv(
            tensor_cache=tensor_cache,
            reward_scheme=STRATEGY_CONFIG["llm"]["response_mapping"], 
            reward_strategy="cosine",
            random_seed=ENV_CONFIG["random_seed"]
        )
    elif reward_strategy == "llm":
        env = LLMSimulatorEnv(
            tensor_cache=tensor_cache,
            reward_scheme=STRATEGY_CONFIG["llm"]["response_mapping"],
            random_seed=ENV_CONFIG["random_seed"]
        )
    elif reward_strategy == "hybrid": # This is the expected reward_strategy
        env = HybridEnv(
            tensor_cache=tensor_cache,
            reward_scheme=STRATEGY_CONFIG["llm"]["response_mapping"],
            cosine_weight=STRATEGY_CONFIG["hybrid"]["initial_cosine_weight"],
            random_seed=ENV_CONFIG["random_seed"]
        )
    else:
        raise ValueError(f"Unknown reward strategy: {reward_strategy}")
    print(f"Created {env.__class__.__name__} for main training (strategy: {reward_strategy}).")

else:
    raise RuntimeError("TensorCache is disabled in CACHE_CONFIG, but the environment now requires it.")

# Models will be initialized within their respective agent training blocks (9 and 10)
# to ensure fresh weights for independent training.

# Initialize Visualizer & Evaluator
visualizer = Visualizer()
evaluator = Evaluator()
print("Initialized Visualizer and Evaluator.")

# Reset environment for the target candidate (primarily for agent_hybrid training)
try:
    print(f"Resetting environment for applicant: {target_candidate_id} (for main training setup)")
    if tensor_cache and target_candidate_id:
         print(f"Cache check before reset: Applicant '{str(target_candidate_id)}' in applicant_states: {str(target_candidate_id) in tensor_cache.applicant_states}")
         
    initial_state_for_hybrid_env = env.reset(applicant_id=target_candidate_id) # Store for potential use by hybrid agent
    
    if not hasattr(env, 'current_state') or env.current_state is None:
         raise ValueError("Environment reset failed - current_state is None or missing after reset.")
    if initial_state_for_hybrid_env is None:
         raise ValueError("Environment reset failed - returned None state.")
         
    print(f"Environment reset successful. Initial state shape for hybrid env: {initial_state_for_hybrid_env.shape}")
    print(f"Environment setup for main agent complete with {reward_strategy} reward strategy.")
    
except Exception as e:
    print(f"ERROR during env.reset or verification: {e}")
    logger.error("Failed during initial environment reset for main agent", exc_info=True)
    raise

# %% [markdown]
# ## 5. LLM Integration (If Using LLM-Based Reward Strategy)
# 
# If we're using an LLM-based reward strategy (either "llm" or "hybrid"), we need to set up a language model to simulate candidate responses to job recommendations. This helps in generating more realistic rewards based on semantic understanding of both candidate profiles and job descriptions.
# 
# Note: This section will only execute if we're using an LLM-based strategy and running in a Colab environment with sufficient resources.

# %%
# Set up LLM in the environment for reward calculation
# This setup applies to the 'env' instance that will be used by agent_hybrid
if STRATEGY_CONFIG["llm"]["enabled"] and (reward_strategy == "llm" or reward_strategy == "hybrid"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from huggingface_hub import login
    
    # Get token path from config and read token
    token_path = HF_CONFIG["token_path"]
    if not os.path.isabs(token_path):
        token_path = token_path
    with open(token_path, "r") as f:
        token = f.read().strip()
    
    # Login to Hugging Face
    login(token=token)
    print(f"Successfully logged in to Hugging Face using token from {token_path}")
    
    # Load model and tokenizer
    model_id = STRATEGY_CONFIG["llm"]["model_id"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=STRATEGY_CONFIG["llm"]["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=STRATEGY_CONFIG["llm"]["quantization"]["quant_type"],
        bnb_4bit_use_nested_quant=STRATEGY_CONFIG["llm"]["quantization"]["use_nested_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16 
    )
    
    print(f"Loading tokenizer for model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"Loading model {model_id} with 4-bit quantization...")
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    print(f"LLM model loaded successfully: {model_id}")
    
    env = env.setup_llm(llm_model, tokenizer, device="auto") 
    print(f"Ensured LLM is set up in {env.__class__.__name__} for strategy '{reward_strategy}' (used by agent_hybrid)")

# %% [markdown]
# ## 8. Training Setup
# 
# We'll now set up the training process using the DynaQAgent's built-in training interface.
# The agent handles:
# 1. Experience replay buffer
# 2. Q-Network and World Model updates
# 3. Loss tracking and metrics
# 4. Model checkpointing
# 
# This section provides general parameters; specific agents will be configured in their training blocks.

# %%
# General training parameters that will be used by both agents if not overridden
# Print general configs for verification
print(f"Device for training: {device}")
print(f"Default Planning steps: {TRAINING_CONFIG['planning_steps']}")
print(f"Default Batch size: {TRAINING_CONFIG['batch_size']}")
print(f"Default Gamma: {TRAINING_CONFIG['gamma']}")

print("General training setup parameters noted.")

# %% [markdown]
# ## 6. Cosine Strategy Agent Training (Baseline)
# 
# In this phase, we train an agent (`agent_cosine`) using exclusively the cosine similarity for reward generation. This agent will serve as our baseline.

# %%
# Define pretraining parameters from config (used for agent_cosine training)
num_pretraining_samples = PRETRAINING_CONFIG["num_samples"]
num_pretraining_epochs = PRETRAINING_CONFIG["num_epochs"]
cosine_training_batch_size = TRAINING_CONFIG["batch_size"] # Can be specific if needed

# Initialize agent_cosine with fresh networks
print("Initializing agent_cosine for COSINE strategy training...")
q_network_cosine = QNetwork(
    state_dim=MODEL_CONFIG["q_network"]["state_dim"],
    action_dim=MODEL_CONFIG["q_network"]["action_dim"],
    hidden_dims=MODEL_CONFIG["q_network"]["hidden_dims"],
    dropout_rate=MODEL_CONFIG["q_network"]["dropout_rate"],
    activation=MODEL_CONFIG["q_network"]["activation"]
).to(device)

world_model_cosine = WorldModel(
    input_dim=MODEL_CONFIG["world_model"]["input_dim"],
    hidden_dims=MODEL_CONFIG["world_model"]["hidden_dims"],
    dropout_rate=MODEL_CONFIG["world_model"]["dropout_rate"],
    activation=MODEL_CONFIG["world_model"]["activation"]
).to(device)

agent_cosine = DynaQAgent(
    state_dim=MODEL_CONFIG["q_network"]["state_dim"],
    action_dim=MODEL_CONFIG["q_network"]["action_dim"],
    q_network=q_network_cosine,
    world_model=world_model_cosine,
    training_strategy="cosine", # Explicitly cosine
    device=device,
    target_applicant_id=target_candidate_id,
    tensor_cache=tensor_cache
)
# Configure agent_cosine specific training parameters
agent_cosine.planning_steps = TRAINING_CONFIG['planning_steps'] # Or a specific value for cosine agent
agent_cosine.batch_size = cosine_training_batch_size
agent_cosine.gamma = TRAINING_CONFIG['gamma']
print(f"Initialized agent_cosine with training strategy: {agent_cosine.training_strategy}")


print("Starting training for agent_cosine (using pretraining method with cosine data)...")

# Generate pretraining data (cosine-based rewards)
print("Generating cosine-based training data for agent_cosine...")
cosine_training_data = []

try:
    initial_state_tensor_cosine = tensor_cache.get_applicant_state(target_candidate_id)
except KeyError as e:
    print(f"Error: {e}. Applicant not found in cache. Cannot proceed with agent_cosine training.")
    raise

valid_job_indices_cosine = tensor_cache.get_valid_job_indices()
print(f"Found {len(valid_job_indices_cosine)} valid jobs in tensor cache for agent_cosine training")

if not valid_job_indices_cosine:
    raise ValueError("Tensor cache contains no valid jobs. Cannot generate agent_cosine training data.")

num_samples_to_generate_cosine = min(num_pretraining_samples, len(valid_job_indices_cosine))
indices_to_use_cosine = valid_job_indices_cosine[:num_samples_to_generate_cosine]

print("Pre-calculating all cosine similarity rewards for agent_cosine training...")
all_rewards_cosine = tensor_cache.calculate_cosine_similarities(initial_state_tensor_cosine)
if STRATEGY_CONFIG["cosine"]["scale_reward"]:
    all_rewards_cosine = (all_rewards_cosine + 1) / 2 # Scale from [-1, 1] to [0, 1]
print("Cosine rewards calculated for agent_cosine.")
    
print(f"Generating {num_samples_to_generate_cosine} samples for agent_cosine training...")
for cache_job_idx_cosine in tqdm(indices_to_use_cosine, desc="Generating agent_cosine training data"):
    action_tensor_cosine = tensor_cache.get_job_vector_by_index(cache_job_idx_cosine)
    reward_cosine = all_rewards_cosine[cache_job_idx_cosine].item()
    next_state_tensor_cosine = initial_state_tensor_cosine 
    cosine_training_data.append((initial_state_tensor_cosine, action_tensor_cosine, reward_cosine, next_state_tensor_cosine))

print(f"Collected {len(cosine_training_data)} experiences for agent_cosine")

if not cosine_training_data:
    raise RuntimeError("No training data was collected for agent_cosine. Cannot proceed.")

try:
    states_cosine = torch.stack([d[0] for d in cosine_training_data]).to(device)
    actions_cosine = torch.stack([d[1] for d in cosine_training_data]).to(device)
    rewards_cosine = torch.tensor([d[2] for d in cosine_training_data], dtype=torch.float32, device=device)
    next_states_cosine = torch.stack([d[3] for d in cosine_training_data]).to(device)
except RuntimeError as e:
    print(f"Error creating tensor batches for agent_cosine: {e}")
    raise

# Run training for agent_cosine using the agent's pretrain method (as it takes static data)
print("Starting agent_cosine network training (using its .pretrain() method)")
agent_cosine_training_metrics = agent_cosine.pretrain(
    states=states_cosine,
    actions=actions_cosine,
    rewards=rewards_cosine,
    next_states=next_states_cosine,
    num_epochs=num_pretraining_epochs, # Using pretraining epochs for this cosine training
    batch_size=cosine_training_batch_size
)

# Plot agent_cosine training metrics
visualizer.plot_training_metrics(
    metrics={
        'q_losses': agent_cosine_training_metrics['q_losses'],
        'world_losses': agent_cosine_training_metrics['world_losses']
    },
    title="agent_cosine Training Losses (Cosine Strategy)"
)

print("agent_cosine training (using pretraining method) completed successfully.")

# %% [markdown]
# ## 7. Hybrid Strategy Agent Training
# 
# Now we train a separate agent (`agent_hybrid`) using the main experimental strategy (e.g., "hybrid").
# This agent starts with fresh network weights and is trained independently of `agent_cosine`.

# %%
# Training parameters from config for agent_hybrid
num_episodes_hybrid = TRAINING_CONFIG["num_episodes"]
max_steps_per_episode_hybrid = TRAINING_CONFIG["max_steps_per_episode"]

print(f"Starting main training loop for agent_hybrid (strategy: {reward_strategy})...")

# Set up for multiple experiments for agent_hybrid
num_experiments = 5  # Number of experiments to run for agent_hybrid
experiment_results_hybrid = {
    'q_network_loss': [],
    'world_model_loss': [],
    'episode_reward': [],
    'eval_reward': []
}

# Ensure the environment for agent_hybrid is the one configured with LLM if hybrid strategy is used.
# The 'env' variable should already be the HybridEnv instance if configured earlier.
print(f"Using environment of type: {env.__class__.__name__} for agent_hybrid training.")

# agent_hybrid will be defined and potentially re-initialized inside the loop
agent_hybrid = None 

# Create directory for saving agent_hybrid results
results_dir_hybrid = os.path.join(PATH_CONFIG["results_dir"], f"multi_experiment_{reward_strategy}")
os.makedirs(results_dir_hybrid, exist_ok=True)

# Run multiple experiments for agent_hybrid
for exp_idx in range(num_experiments):
    print(f"\nStarting agent_hybrid experiment {exp_idx+1}/{num_experiments}")
    
    # Initialize or Reinitialize agent_hybrid for each experiment to ensure independence
    print(f"Initializing new agent_hybrid instance for experiment {exp_idx+1}...")
    q_network_hybrid_exp = QNetwork(
        state_dim=MODEL_CONFIG["q_network"]["state_dim"],
        action_dim=MODEL_CONFIG["q_network"]["action_dim"],
        hidden_dims=MODEL_CONFIG["q_network"]["hidden_dims"],
        dropout_rate=MODEL_CONFIG["q_network"]["dropout_rate"],
        activation=MODEL_CONFIG["q_network"]["activation"]
    ).to(device)

    world_model_hybrid_exp = WorldModel(
        input_dim=MODEL_CONFIG["world_model"]["input_dim"],
        hidden_dims=MODEL_CONFIG["world_model"]["hidden_dims"],
        dropout_rate=MODEL_CONFIG["world_model"]["dropout_rate"],
        activation=MODEL_CONFIG["world_model"]["activation"]
    ).to(device)
    
    agent_hybrid = DynaQAgent(
        state_dim=MODEL_CONFIG["q_network"]["state_dim"],
        action_dim=MODEL_CONFIG["q_network"]["action_dim"],
        q_network=q_network_hybrid_exp,
        world_model=world_model_hybrid_exp,
        training_strategy=reward_strategy, # This should be 'hybrid' or 'llm'
        device=device,
        target_applicant_id=target_candidate_id,
        tensor_cache=tensor_cache
    )
    # Configure agent_hybrid specific training parameters
    agent_hybrid.planning_steps = TRAINING_CONFIG['planning_steps']
    agent_hybrid.batch_size = TRAINING_CONFIG['batch_size']
    agent_hybrid.gamma = TRAINING_CONFIG['gamma']
    print(f"Initialized agent_hybrid with training strategy: {agent_hybrid.training_strategy}")

    # Run training for this agent_hybrid experiment
    # The env here should be the HybridEnv if reward_strategy is hybrid
    training_metrics_hybrid = agent_hybrid.train(
        env=env, 
        num_episodes=num_episodes_hybrid,
        max_steps_per_episode=max_steps_per_episode_hybrid,
        applicant_ids=[target_candidate_id]
    )
    
    # Store results for this agent_hybrid experiment
    for key in experiment_results_hybrid:
        if key in training_metrics_hybrid:
            experiment_results_hybrid[key].append(training_metrics_hybrid[key])
    
    # Save individual agent_hybrid experiment results
    exp_file_hybrid = os.path.join(results_dir_hybrid, f"experiment_hybrid_{exp_idx+1}.pt")
    torch.save(training_metrics_hybrid, exp_file_hybrid)
    print(f"Saved agent_hybrid experiment {exp_idx+1} results to {exp_file_hybrid}")

# Save aggregated results for agent_hybrid
aggregated_file_hybrid = os.path.join(results_dir_hybrid, "aggregated_results_hybrid.pt")
torch.save(experiment_results_hybrid, aggregated_file_hybrid)
print(f"Saved aggregated agent_hybrid results to {aggregated_file_hybrid}")

# Visualize results for agent_hybrid using the Visualizer
visualizer.plot_experiment_results(
    experiment_results=experiment_results_hybrid,
    title_prefix=f"Dyna-Q Performance ({reward_strategy.upper()} Strategy)",
    filename_prefix=f"multi_experiment_{reward_strategy}"
)

# Plot training metrics for the last agent_hybrid experiment
# Ensure training_metrics_hybrid holds the metrics from the last experiment
if training_metrics_hybrid:
    visualizer.plot_training_metrics(
        metrics={
            'q_losses': training_metrics_hybrid['q_network_loss'],
            'world_losses': training_metrics_hybrid['world_model_loss'],
            'episode_rewards': training_metrics_hybrid['episode_reward']
        },
        title=f"Training Metrics (Last {reward_strategy.upper()} Agent Experiment)"
    )

print(f"Multiple experiments for agent_hybrid ({reward_strategy} strategy) completed successfully.")

# %% [markdown]
# ## 7.1 Loading and Visualizing Saved Experiment Results
# 
# This section provides functionality to load and analyze results from previously run experiments.
# This is useful for revisiting experiment results without having to rerun the experiments.

# %%
def load_experiment_results(results_dir):
    """
    Load saved experiment results for visualization.
    
    Args:
        results_dir: Directory containing experiment results
    
    Returns:
        dict: Loaded experiment results
    """
    # Check if aggregated results exist
    aggregated_file = os.path.join(results_dir, "aggregated_results.pt")
    
    if os.path.exists(aggregated_file):
        print(f"Loading aggregated results from {aggregated_file}")
        return torch.load(aggregated_file)
    else:
        # Try to load individual experiment files
        import glob
        exp_files = glob.glob(os.path.join(results_dir, "experiment_*.pt"))
        
        if not exp_files:
            print(f"No experiment files found in {results_dir}")
            return {}
        
        print(f"Found {len(exp_files)} individual experiment files")
        
        # Load individual files and combine them
        combined_results = {
            'q_network_loss': [],
            'world_model_loss': [],
            'episode_reward': [],
            'eval_reward': []
        }
        
        for file in exp_files:
            exp_data = torch.load(file)
            for key in combined_results:
                if key in exp_data:
                    combined_results[key].append(exp_data[key])
        
        return combined_results

# Example usage (commented out by default)
# results_dir = os.path.join(PATH_CONFIG["results_dir"], "multi_experiment")
# loaded_results = load_experiment_results(results_dir)
# if loaded_results:
#     visualizer.plot_experiment_results(
#         experiment_results=loaded_results,
#         title_prefix="Loaded Results",
#         filename_prefix="loaded_experiment"
#     )

# %% [markdown]
# ## 8. Evaluation
# 
# Finally, we evaluate the trained agents: `agent_cosine` (baseline) and `agent_hybrid`.
# This will:
# 1. Test both agents on a set of evaluation episodes.
# 2. Compare their performance using the `Evaluator`.
# 3. Generate evaluation metrics and visualizations.

# %%
# Evaluation parameters from config
num_eval_episodes = EVAL_CONFIG["num_eval_episodes"]
# baseline_strategy is implicitly "cosine" due to agent_cosine's training

print("Starting evaluation: Comparing agent_cosine vs agent_hybrid...")

# Ensure agent_cosine and agent_hybrid are the trained instances from Block 9 and 10 respectively.
# agent_cosine is fully trained after Block 9.
# agent_hybrid is the instance from the last iteration of the experiment loop in Block 10.

if agent_cosine is None or agent_hybrid is None:
    raise RuntimeError("One or both agents (agent_cosine, agent_hybrid) are not defined. Ensure training blocks were run.")

# The environment used for evaluation should be consistent for both agents.
# Using the 'env' that was set up for the hybrid strategy, as it's more general
# or can be adapted by the agents if their internal logic or the evaluator handles it.
# If agent_cosine strictly requires a simpler env, that would need specific handling.
# However, evaluate_agent primarily uses env.reset, env.get_valid_actions, env.get_action_vector, env.step.
# The reward generation within env.step will apply to both if they use the same env instance.
# For a fair comparison of learned policies, the evaluation environment should be the same.

print(f"Evaluation will use environment of type: {env.__class__.__name__}")

# Use evaluator's compare_agents method
comparison_results = evaluator.compare_agents(
    baseline_agent=agent_cosine,    # Agent trained with cosine strategy in Block 9
    pretrained_agent=agent_hybrid,  # Agent trained with hybrid strategy in Block 10 (final instance)
    env=env,
    applicant_ids=[target_candidate_id],
    num_episodes=num_eval_episodes
)

# Extract the results for visualization
# Note: 'pretrained' in comparison_results refers to agent_hybrid here
evaluation_visualization_data = {
    'agent_hybrid_rewards': comparison_results['pretrained']['episode_rewards'],
    'agent_cosine_rewards': comparison_results['baseline']['episode_rewards']
}

# Plot evaluation results
visualizer.plot_evaluation_results(
    baseline_rewards=evaluation_visualization_data['agent_cosine_rewards'],
    pretrained_rewards=evaluation_visualization_data['agent_hybrid_rewards'],
    title="Evaluation Results: Cosine Agent vs. Hybrid Agent"
)

print("Evaluation completed successfully.")

# %% [markdown]
# ## 9. Generate Job Recommendations
# 
# Now let's use our trained `agent_hybrid` to generate personalized job recommendations for our target candidate. We'll:
# 
# 1. Use the trained Q-network of `agent_hybrid` to evaluate jobs
# 2. Select the top-K jobs with highest Q-values
# 3. Display the recommendations along with job details

# %%
# Define testing parameters
num_recommendations = EVAL_CONFIG["top_k_recommendations"]
# test_epsilon = 0.0 is implicit in agent.select_action with eval_mode=True

# Initialize lists to store recommendations
recommended_jobs_hybrid = []
recommendation_scores_hybrid = []

# Get all valid job indices from tensor_cache
valid_job_indices_reco = list(range(len(tensor_cache))) # Renamed to avoid conflict
print(f"Found {len(valid_job_indices_reco)} valid jobs in tensor cache for recommendations using agent_hybrid")

# Reset the environment to get the initial state (using the main 'env' instance)
state_reco = env.reset(applicant_id=target_candidate_id) # Renamed for clarity

# Create a copy of job indices to work with
remaining_job_indices_reco = valid_job_indices_reco.copy()

if agent_hybrid is None:
    raise RuntimeError("agent_hybrid is not defined. Ensure training block 10 was run.")

print(f"Generating top-{num_recommendations} recommendations using agent_hybrid...")
# Generate top-K recommendations using agent_hybrid
for _ in range(min(num_recommendations, len(valid_job_indices_reco))):
    action_tensors_reco = [tensor_cache.get_job_vector_by_index(idx) for idx in remaining_job_indices_reco]
    
    if not action_tensors_reco:
        print("No more actions available to recommend.")
        break
        
    if not isinstance(state_reco, torch.Tensor):
         state_reco = torch.tensor(state_reco, device=device, dtype=torch.float32)
         
    action_idx_reco, _ = agent_hybrid.select_action(state_reco, action_tensors_reco, eval_mode=True) 
    
    selected_cache_idx_reco = remaining_job_indices_reco[action_idx_reco]
    job_id_reco = tensor_cache.get_job_id(selected_cache_idx_reco)
    
    with torch.no_grad():
        state_tensor_reco = state_reco.unsqueeze(0) if state_reco.dim() == 1 else state_reco 
        job_tensor_reco = tensor_cache.get_job_vector_by_index(selected_cache_idx_reco).unsqueeze(0)
        q_value_reco = agent_hybrid.q_network(state_tensor_reco, job_tensor_reco).item()
    
    recommended_jobs_hybrid.append(job_id_reco)
    recommendation_scores_hybrid.append(q_value_reco)
    
    remaining_job_indices_reco.pop(action_idx_reco)

# Display recommendations from agent_hybrid
print(f"\n=== Top Job Recommendations from agent_hybrid ({reward_strategy} strategy) ===\n")
for i, (job_id, score) in enumerate(zip(recommended_jobs_hybrid, recommendation_scores_hybrid)):
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
    except KeyError:
        job_id_str = str(job_id)
        print(f"Error: Metadata not found in TensorCache for job {job_id_str}")
    except Exception as e:
        print(f"Error retrieving job details from cache: {e}")
    print("-" * 50)

# %% [markdown]
# ## 10 Save Final Model Weights (agent_hybrid)
# 
# This block explicitly saves the state dictionary of the trained Q-network 
# from `agent_hybrid` to the results directory.

# %%
# Explicitly save the final Q-network weights of agent_hybrid
import os

# Define the filename for the saved model
try:
    model_filename_hybrid = f"q_network_final_{reward_strategy}_{str(target_candidate_id)}.pt"
except NameError:
    model_filename_hybrid = f"q_network_final_{reward_strategy}_unknown_candidate.pt"
    print("Warning: target_candidate_id not found, using default filename for hybrid agent.")

# Get the results directory from PATH_CONFIG (should be results_dir_hybrid or similar)
# Using results_dir_hybrid which was defined in Block 10
if 'results_dir_hybrid' not in locals():
    print("Warning: results_dir_hybrid not defined. Attempting to use general results_dir.")
    results_dir_final_save = PATH_CONFIG.get("results_dir", "../results")
else:
    results_dir_final_save = results_dir_hybrid

os.makedirs(results_dir_final_save, exist_ok=True)

# Construct the full save path
save_path_hybrid = os.path.join(results_dir_final_save, model_filename_hybrid)

# Save the agent_hybrid's Q-network state dictionary
try:
    if agent_hybrid and hasattr(agent_hybrid, 'q_network'):
        torch.save(agent_hybrid.q_network.state_dict(), save_path_hybrid)
        print(f"Successfully saved agent_hybrid Q-network weights to: {save_path_hybrid}")
    else:
        print("Error: agent_hybrid or its Q-network not found. Cannot save weights.")
except Exception as e:
    print(f"Error saving agent_hybrid Q-network weights: {e}")

# %% [markdown]
# ## 11. Conclusion and Next Steps
# 
# In this notebook, we've implemented and demonstrated a Neural Dyna-Q job recommendation system. This approach combines the strengths of deep reinforcement learning with model-based planning to provide personalized job recommendations.
# 
# ### 11.1 Key Accomplishments
# 
# 1. **Data Integration**: Connected to the MongoDB database to retrieve real candidate and job data
# 2. **Neural Networks**: Implemented deep Q-network and world model for value function and dynamics prediction
# 3. **Dyna-Q Algorithm**: Combined direct RL with model-based planning for efficient learning
# 4. **Personalized Recommendations**: Generated job recommendations tailored to a specific candidate
# 
# ### 11.2 Potential Improvements
# 
# 1. **Extended Training**: Train for more episodes to improve recommendation quality
# 2. **Hyperparameter Tuning**: Optimize learning rates, network architectures, and other parameters
# 3. **Advanced Reward Functions**: Implement more sophisticated reward strategies using LLMs
# 4. **User Feedback**: Incorporate real user feedback to improve recommendations
# 
# ### 11.3 Applications
# 
# This system could be deployed as:
# - A personalized job recommendation service for job seekers
# - A candidate-job matching tool for recruiters
# - A component in a larger career guidance system

# %%
# Clean up resources
db_connector.close()
print("Database connection closed.")

# Free up GPU memory
if tensor_cache is not None and hasattr(tensor_cache, 'clear'):
    tensor_cache.clear()
    print("Tensor cache cleared.")
    
# Free PyTorch memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("CUDA cache emptied.")
    
print("Notebook execution complete.")