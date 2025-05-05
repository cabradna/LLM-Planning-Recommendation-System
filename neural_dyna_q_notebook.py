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

# Ensure log directory exists
log_dir = PATH_CONFIG.get("log_dir", "logs") # Default to ./logs if not in config
if not os.path.isabs(log_dir):
    # If relative path, assume relative to project root (modify if needed)
    try:
        project_root_for_log = Path(__file__).resolve().parent.parent
    except NameError:
        project_root_for_log = Path('.').absolute()
    log_dir = os.path.join(project_root_for_log, log_dir)
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'notebook_run.log')

# Get root logger and set level
# Set level to INFO to see INFO, WARNING, ERROR, CRITICAL
# Set level to DEBUG to see everything
log_level = logging.INFO # Or logging.DEBUG for more verbosity
logger = logging.getLogger() 
logger.setLevel(log_level)

# Remove existing handlers if any (to avoid duplicates, optional)
# for handler in logger.handlers[:]:
#    logger.removeHandler(handler)

# Create formatter
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create File Handler
file_handler = logging.FileHandler(log_file_path, mode='w') # 'w' to overwrite each run
file_handler.setFormatter(log_formatter)
file_handler.setLevel(log_level)
logger.addHandler(file_handler)

# Create Stream Handler (for notebook output)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(log_level)
logger.addHandler(stream_handler)

logger.info(f"Logging configured. Level: {logging.getLevelName(log_level)}. File: {log_file_path}")
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
    ENV_CONFIG
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
# ## 4. Database Connection
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
# ## 5. Data Loading, Cache Init, Agent/Env Init, and First Reset
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
agent = None

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
    if reward_strategy == "cosine":
        env = JobRecommendationEnv(
            tensor_cache=tensor_cache,
            reward_scheme=STRATEGY_CONFIG["llm"]["response_mapping"], # Keep scheme for potential hybrid switch
            reward_strategy="cosine",
            random_seed=ENV_CONFIG["random_seed"]
        )
    elif reward_strategy == "llm":
        env = LLMSimulatorEnv(
            tensor_cache=tensor_cache,
            reward_scheme=STRATEGY_CONFIG["llm"]["response_mapping"],
            random_seed=ENV_CONFIG["random_seed"]
        )
    elif reward_strategy == "hybrid":
        env = HybridEnv(
            tensor_cache=tensor_cache,
            reward_scheme=STRATEGY_CONFIG["llm"]["response_mapping"],
            cosine_weight=STRATEGY_CONFIG["hybrid"]["initial_cosine_weight"],
            random_seed=ENV_CONFIG["random_seed"]
        )
    else:
        raise ValueError(f"Unknown reward strategy: {reward_strategy}")
    print(f"Created {reward_strategy} environment instance.")

else:
    raise RuntimeError("TensorCache is disabled in CACHE_CONFIG, but the environment now requires it.")

# Initialize Models
q_network = QNetwork(
    state_dim=MODEL_CONFIG["q_network"]["state_dim"],
    action_dim=MODEL_CONFIG["q_network"]["action_dim"],
    hidden_dims=MODEL_CONFIG["q_network"]["hidden_dims"],
    dropout_rate=MODEL_CONFIG["q_network"]["dropout_rate"],
    activation=MODEL_CONFIG["q_network"]["activation"]
).to(device)

world_model = WorldModel(
    input_dim=MODEL_CONFIG["world_model"]["input_dim"],
    hidden_dims=MODEL_CONFIG["world_model"]["hidden_dims"],
    dropout_rate=MODEL_CONFIG["world_model"]["dropout_rate"],
    activation=MODEL_CONFIG["world_model"]["activation"]
).to(device)
print("Initialized QNetwork and WorldModel.")

# Initialize DynaQAgent
agent = DynaQAgent(
    state_dim=MODEL_CONFIG["q_network"]["state_dim"],
    action_dim=MODEL_CONFIG["q_network"]["action_dim"],
    q_network=q_network,
    target_network=None, # Allow init to create target network
    world_model=world_model,
    training_strategy=reward_strategy,
    device=device,
    target_applicant_id=target_candidate_id,
    tensor_cache=tensor_cache # Pass the populated cache
)
print(f"Initialized DynaQAgent for applicant {target_candidate_id}.")

# Initialize Visualizer & Evaluator
visualizer = Visualizer()
evaluator = Evaluator()
print("Initialized Visualizer and Evaluator.")

# Reset environment and verify
try:
    print(f"Resetting environment for applicant: {target_candidate_id}")
    # Check cache content just before reset for debugging
    if tensor_cache and target_candidate_id:
         print(f"Cache check before reset: Applicant '{str(target_candidate_id)}' in applicant_states: {str(target_candidate_id) in tensor_cache.applicant_states}")
         
    initial_state = env.reset(applicant_id=target_candidate_id)
    
    if not hasattr(env, 'current_state') or env.current_state is None:
         raise ValueError("Environment reset failed - current_state is None or missing after reset.")
    if initial_state is None:
         raise ValueError("Environment reset failed - returned None state.")
         
    print(f"Environment reset successful. Initial state shape: {initial_state.shape}")
    print(f"Environment setup complete with {reward_strategy} reward strategy.")
    
except Exception as e:
    print(f"ERROR during env.reset or verification: {e}")
    logger.error("Failed during initial environment reset", exc_info=True)
    raise

# %% [markdown]
# ## 7. LLM Integration (If Using LLM-Based Reward Strategy)
# 
# If we're using an LLM-based reward strategy (either "llm" or "hybrid"), we need to set up a language model to simulate candidate responses to job recommendations. This helps in generating more realistic rewards based on semantic understanding of both candidate profiles and job descriptions.
# 
# Note: This section will only execute if we're using an LLM-based strategy and running in a Colab environment with sufficient resources.

# %%
# Set up LLM in the environment for reward calculation
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
    
    # Add LLM to the environment using the updated method
    # env.setup_llm should handle creating HybridEnv if needed
    # The method now returns the potentially new env instance
    env = env.setup_llm(llm_model, tokenizer, device="auto") 
    print(f"Ensured LLM is set up in {env.__class__.__name__} for strategy '{reward_strategy}'")

# %% [markdown]
# ## 8. Training Setup
# 
# We'll now set up the training process using the DynaQAgent's built-in training interface.
# The agent handles:
# 1. Experience replay buffer
# 2. Q-Network and World Model updates
# 3. Loss tracking and metrics
# 4. Model checkpointing

# %%
# Verify the agent is properly set up
print(f"Agent device: {agent.device}")
print(f"Training strategy: {agent.training_strategy}")
print(f"Q-Network parameters: {sum(p.numel() for p in agent.q_network.parameters() if p.requires_grad):,}")
print(f"World Model parameters: {sum(p.numel() for p in agent.world_model.parameters() if p.requires_grad):,}")
print(f"Planning steps: {TRAINING_CONFIG['planning_steps']}")
print(f"Batch size: {TRAINING_CONFIG['batch_size']}")

# Configure any additional training parameters
agent.planning_steps = TRAINING_CONFIG['planning_steps']
agent.batch_size = TRAINING_CONFIG['batch_size']
agent.gamma = TRAINING_CONFIG['gamma']

print("Training setup completed successfully.")

# %% [markdown]
# ## 9. Pretraining Phase
# 
# In the pretraining phase, we'll use the agent's built-in pretraining method to:
# 1. Generate initial experiences
# 2. Train the Q-network and world model
# 3. Track training metrics
# 4. Save model checkpoints

# %%
# Define pretraining parameters from config
num_pretraining_samples = TRAINING_CONFIG["pretraining"]["num_samples"]
num_pretraining_epochs = TRAINING_CONFIG["pretraining"]["num_epochs"]
batch_size = TRAINING_CONFIG["batch_size"]

# Force cosine strategy for pretraining, regardless of main strategy
pretraining_strategy = "cosine" 

print(f"Starting pretraining with {pretraining_strategy} strategy...")

# Generate pretraining data
print("Generating pretraining data...")
pretraining_data = []

# Get applicant state from cache
try:
    initial_state_tensor = tensor_cache.get_applicant_state(target_candidate_id)
except KeyError as e:
    print(f"Error: {e}. Applicant not found in cache. Cannot proceed with pretraining.")
    raise

# Get valid job indices from tensor cache
valid_job_indices = tensor_cache.get_valid_job_indices()
print(f"Found {len(valid_job_indices)} valid jobs in tensor cache")

if not valid_job_indices:
    raise ValueError("Tensor cache contains no valid jobs. Cannot generate pretraining data.")

# Determine number of samples to generate
num_samples_to_generate = min(num_pretraining_samples, len(valid_job_indices))
indices_to_use = valid_job_indices[:num_samples_to_generate] # Use the first N valid indices

if pretraining_strategy == "cosine":
    print("Using tensor cache only for cosine strategy.")
    # Pre-calculate all cosine similarity rewards
    print("Pre-calculating all cosine similarity rewards...")
    all_rewards = tensor_cache.calculate_cosine_similarities(initial_state_tensor)
    if STRATEGY_CONFIG["cosine"]["scale_reward"]:
        all_rewards = (all_rewards + 1) / 2 # Scale from [-1, 1] to [0, 1]
    print("Cosine rewards calculated.")
    
    # Collect pretraining experiences using cache
    print(f"Generating {num_samples_to_generate} samples using cache...")
    for cache_job_idx in tqdm(indices_to_use, desc="Generating pretraining data (cosine)"):
        action_tensor = tensor_cache.get_job_vector_by_index(cache_job_idx)
        reward = all_rewards[cache_job_idx].item()
        # Assuming static state for pretraining
        next_state_tensor = initial_state_tensor 
        pretraining_data.append((initial_state_tensor, action_tensor, reward, next_state_tensor))

# This part should now NOT be reached if pretraining_strategy is forced to 'cosine'
# elif pretraining_strategy in ["llm", "hybrid"]:
#     print(f"Using environment interaction for {pretraining_strategy} strategy.")
#     # ... (code for env interaction - removed for clarity as it's not used now)
else:
    # This handles the case where pretraining_strategy is somehow not 'cosine'
    raise ValueError(f"Unsupported pretraining strategy: {pretraining_strategy}")

print(f"Collected {len(pretraining_data)} pretraining experiences")

# Check if any data was collected before stacking
if not pretraining_data:
    raise RuntimeError("No pretraining data was collected. Cannot proceed.")

# Create tensor batches
# Ensure all tensors in the list have the same shape before stacking
try:
    states = torch.stack([d[0] for d in pretraining_data]).to(device)
    actions = torch.stack([d[1] for d in pretraining_data]).to(device)
    rewards = torch.tensor([d[2] for d in pretraining_data], dtype=torch.float32, device=device)
    next_states = torch.stack([d[3] for d in pretraining_data]).to(device)
except RuntimeError as e:
    print(f"Error creating tensor batches: {e}")
    # Add more detailed logging if needed
    # for i, d in enumerate(pretraining_data):
    #     print(f"Data point {i}: state shape {d[0].shape}, action shape {d[1].shape}, next_state shape {d[3].shape}")
    raise

# Run pretraining using agent's built-in pretrain method
print("Starting pretraining of neural networks")
pretraining_metrics = agent.pretrain(
    states=states,
    actions=actions,
    rewards=rewards,
    next_states=next_states,
    num_epochs=num_pretraining_epochs,
    batch_size=batch_size
)

# Plot pretraining metrics
visualizer.plot_training_metrics(
    metrics={
        'q_losses': pretraining_metrics['q_losses'],
        'world_losses': pretraining_metrics['world_losses']
    },
    title="Pretraining Losses"
)

# Configure agent for main training
agent.planning_steps = TRAINING_CONFIG['planning_steps']
agent.batch_size = TRAINING_CONFIG['batch_size']
agent.gamma = TRAINING_CONFIG['gamma']

print("Pretraining completed successfully.")

# %% [markdown]
# ## 10. Main Training Loop
# 
# Now we'll run the main training loop using the agent's training interface.
# The agent will:
# 1. Interact with the environment
# 2. Store experiences in its replay buffer
# 3. Update networks using the stored experiences
# 4. Track and report training metrics

# %%
# Training parameters from config
num_episodes = TRAINING_CONFIG["num_episodes"]
max_steps_per_episode = TRAINING_CONFIG["max_steps_per_episode"]

print("Starting main training loop...")

# Set up for multiple experiments
num_experiments = 5  # Number of experiments to run
experiment_results = {
    'q_network_loss': [],
    'world_model_loss': [],
    'episode_reward': [],
    'eval_reward': []
}

# Create directory for saving results
results_dir = os.path.join(PATH_CONFIG["results_dir"], "multi_experiment")
os.makedirs(results_dir, exist_ok=True)

# Run multiple experiments
for exp_idx in range(num_experiments):
    print(f"\nStarting experiment {exp_idx+1}/{num_experiments}")
    
    # Reinitialize agent for each experiment to ensure independence
    if exp_idx > 0:
        q_network = QNetwork(
            state_dim=MODEL_CONFIG["q_network"]["state_dim"],
            action_dim=MODEL_CONFIG["q_network"]["action_dim"],
            hidden_dims=MODEL_CONFIG["q_network"]["hidden_dims"],
            dropout_rate=MODEL_CONFIG["q_network"]["dropout_rate"],
            activation=MODEL_CONFIG["q_network"]["activation"]
        ).to(device)

        world_model = WorldModel(
            input_dim=MODEL_CONFIG["world_model"]["input_dim"],
            hidden_dims=MODEL_CONFIG["world_model"]["hidden_dims"],
            dropout_rate=MODEL_CONFIG["world_model"]["dropout_rate"],
            activation=MODEL_CONFIG["world_model"]["activation"]
        ).to(device)
        
        agent = DynaQAgent(
            state_dim=MODEL_CONFIG["q_network"]["state_dim"],
            action_dim=MODEL_CONFIG["q_network"]["action_dim"],
            q_network=q_network,
            world_model=world_model,
            training_strategy=reward_strategy,
            device=device,
            target_applicant_id=target_candidate_id,
            tensor_cache=tensor_cache
        )
    
    # Run training for this experiment
    training_metrics = agent.train(
        env=env,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        applicant_ids=[target_candidate_id]
    )
    
    # Store results for this experiment
    for key in experiment_results:
        if key in training_metrics:
            experiment_results[key].append(training_metrics[key])
    
    # Save individual experiment results
    exp_file = os.path.join(results_dir, f"experiment_{exp_idx+1}.pt")
    torch.save(training_metrics, exp_file)
    print(f"Saved experiment {exp_idx+1} results to {exp_file}")

# Save aggregated results
aggregated_file = os.path.join(results_dir, "aggregated_results.pt")
torch.save(experiment_results, aggregated_file)
print(f"Saved aggregated results to {aggregated_file}")

# Visualize results using the Visualizer
visualizer.plot_experiment_results(
    experiment_results=experiment_results,
    title_prefix="Dyna-Q Performance",
    filename_prefix="multi_experiment"
)

# Plot training metrics for the last experiment
visualizer.plot_training_metrics(
    metrics={
        'q_losses': training_metrics['q_network_loss'],
        'world_losses': training_metrics['world_model_loss'],
        'episode_rewards': training_metrics['episode_reward']
    },
    title="Training Metrics (Last Experiment)"
)

print("Multiple experiments completed successfully.")

# %% [markdown]
# ## 10.1 Loading and Visualizing Saved Experiment Results
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
# ## 11. Evaluation
# 
# Finally, we'll evaluate the trained agent using the Evaluator class.
# This will:
# 1. Test the agent on a set of evaluation episodes
# 2. Compare performance against baseline
# 3. Generate evaluation metrics and visualizations

# %%
# Evaluation parameters from config
num_eval_episodes = EVAL_CONFIG["num_episodes"]
baseline_strategy = EVAL_CONFIG["baseline_strategy"]

print("Starting evaluation...")

# Create a baseline agent for comparison (NO db_connector)
baseline_agent = DynaQAgent(
    state_dim=MODEL_CONFIG["q_network"]["state_dim"],
    action_dim=MODEL_CONFIG["q_network"]["action_dim"],
    training_strategy=baseline_strategy,
    device=device,
    tensor_cache=tensor_cache  # Include tensor_cache
)

# Use evaluator's compare_agents method which internally calls evaluate_agent
comparison_results = evaluator.compare_agents(
    baseline_agent=baseline_agent,
    pretrained_agent=agent,
    env=env,
    applicant_ids=[target_candidate_id],
    num_episodes=num_eval_episodes
)

# Extract the results for visualization
evaluation_results = {
    'agent_rewards': comparison_results['pretrained']['episode_rewards'],
    'baseline_rewards': comparison_results['baseline']['episode_rewards']
}

# Plot evaluation results
visualizer.plot_evaluation_results(
    baseline_rewards=evaluation_results['baseline_rewards'],
    pretrained_rewards=evaluation_results['agent_rewards'],
    title="Evaluation Results"
)

print("Evaluation completed successfully.")

# %% [markdown]
# ## 12. Generate Job Recommendations
# 
# Now let's use our trained agent to generate personalized job recommendations for our target candidate. We'll:
# 
# 1. Use the trained Q-network to evaluate jobs
# 2. Select the top-K jobs with highest Q-values
# 3. Display the recommendations along with job details
# 
# These recommendations represent the jobs the agent believes are most suitable for the candidate.

# %%
# Define testing parameters
num_recommendations = EVAL_CONFIG["top_k_recommendations"]
test_epsilon = 0.0  # No exploration during testing (pure exploitation)

# Initialize lists to store recommendations
recommended_jobs = []
recommendation_scores = []

# Get all valid job indices
valid_job_indices = list(range(len(tensor_cache.job_ids)))
print(f"Found {len(valid_job_indices)} valid jobs in tensor cache for recommendations")

# Reset the environment to get the initial state
state = env.reset(applicant_id=target_candidate_id)

# Create a copy of job indices to work with
remaining_job_indices = valid_job_indices.copy()

# Generate top-K recommendations
for _ in range(min(num_recommendations, len(valid_job_indices))):
    # Get job tensors for remaining indices from CACHE
    action_tensors = [tensor_cache.get_job_vector_by_index(idx) for idx in remaining_job_indices]
    
    # Handle case where action_tensors might be empty if remaining_job_indices is empty
    if not action_tensors:
        print("No more actions available to recommend.")
        break
        
    # Select best job according to Q-network (using cache vectors)
    # Ensure state is valid tensor
    if not isinstance(state, torch.Tensor):
         state = torch.tensor(state, device=device, dtype=torch.float32) # Ensure state is tensor
         
    # Pass the list of tensors directly
    action_idx, _ = agent.select_action(state, action_tensors, eval_mode=True) 
    
    # Get the corresponding CACHE index
    selected_cache_idx = remaining_job_indices[action_idx]
    job_id = tensor_cache.get_job_id(selected_cache_idx) # Get original job ID using cache index
    
    # Calculate Q-value for logging (using cache vector)
    with torch.no_grad():
        # Ensure state_tensor is correctly shaped [1, state_dim]
        state_tensor = state.unsqueeze(0) if state.dim() == 1 else state 
        # Get action tensor from cache and ensure shape [1, action_dim]
        job_tensor = tensor_cache.get_job_vector_by_index(selected_cache_idx).unsqueeze(0)
        q_value = agent.q_network(state_tensor, job_tensor).item()
    
    recommended_jobs.append(job_id)
    recommendation_scores.append(q_value)
    
    # Remove selected job (using its index within remaining_job_indices) from consideration for next selection
    remaining_job_indices.pop(action_idx)

# Display recommendations with details from CACHE METADATA
print("\n=== Top Job Recommendations ===\n")
for i, (job_id, score) in enumerate(zip(recommended_jobs, recommendation_scores)):
    print(f"Recommendation #{i+1}: [Q-Value: {score:.4f}]")
    # Ensure job_id is the correct type (likely ObjectId initially)
    job_id_str = str(job_id) 
    print(f"Job ID: {job_id_str}")
    
    # Retrieve and display job details from TensorCache metadata
    try:
        # Use string representation for metadata lookup
        job_details = tensor_cache.get_job_metadata(job_id_str) 
        print(f"Title: {job_details.get('job_title', 'N/A')}")
        
        # Display truncated description
        description = job_details.get('description', 'N/A')
        print(f"Description: {description[:100]}..." if len(description) > 100 else f"Description: {description}")
        
        # Display technical skills if available in metadata
        if 'technical_skills' in job_details:
            # Ensure skills are joinable (list of strings)
            skills = job_details['technical_skills']
            if isinstance(skills, list):
                 print(f"Technical Skills: {', '.join(map(str, skills))}")
            else:
                 print(f"Technical Skills: {skills}") # Print as is if not a list

    except KeyError:
        print(f"Error: Metadata not found in TensorCache for job {job_id_str}")
    except Exception as e:
        print(f"Error retrieving job details from cache: {e}")
    print("-" * 50)

# %% [markdown]
# ## 13. Conclusion and Next Steps
# 
# In this notebook, we've implemented and demonstrated a Neural Dyna-Q job recommendation system. This approach combines the strengths of deep reinforcement learning with model-based planning to provide personalized job recommendations.
# 
# ### 13.1 Key Accomplishments
# 
# 1. **Data Integration**: Connected to the MongoDB database to retrieve real candidate and job data
# 2. **Neural Networks**: Implemented deep Q-network and world model for value function and dynamics prediction
# 3. **Dyna-Q Algorithm**: Combined direct RL with model-based planning for efficient learning
# 4. **Personalized Recommendations**: Generated job recommendations tailored to a specific candidate
# 
# ### 13.2 Potential Improvements
# 
# 1. **Extended Training**: Train for more episodes to improve recommendation quality
# 2. **Hyperparameter Tuning**: Optimize learning rates, network architectures, and other parameters
# 3. **Advanced Reward Functions**: Implement more sophisticated reward strategies using LLMs
# 4. **User Feedback**: Incorporate real user feedback to improve recommendations
# 
# ### 13.3 Applications
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