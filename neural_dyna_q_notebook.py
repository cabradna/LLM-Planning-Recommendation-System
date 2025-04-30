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
import sys
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    try:
        # Get repository URL from configuration
        repo_url = PATH_CONFIG["repo_url"]
        
        # Clone the repository using subprocess
        import subprocess
        subprocess.run(["git", "clone", repo_url], check=True)
        os.chdir("LLM-Planning-Recommendation-System")
        
        # Install required packages
        subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)
        subprocess.run(["pip", "install", "-e", "."], check=True)
        
        # If we need to use LLM for simulation, install required packages
        if STRATEGY_CONFIG["llm"]["enabled"] or STRATEGY_CONFIG["hybrid"]["enabled"]:
            subprocess.run(["pip", "install", "-q", "transformers", "accelerate", "bitsandbytes"], check=True)
            print("LLM-related packages installed successfully.")
            
        print("Repository cloned and dependencies installed successfully.")
    except Exception as e:
        print(f"Setup error: {e}")
        print("Continuing with local environment...")

# %%
# Standard imports for data manipulation, visualization, and deep learning
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import torch
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

# Import configuration
from config.config import (
    DB_CONFIG,
    MODEL_CONFIG,
    TRAINING_CONFIG,
    STRATEGY_CONFIG,
    PATH_CONFIG,
    EVAL_CONFIG
)

# Set random seeds for reproducibility across runs
random.seed(TRAINING_CONFIG["random_seed"])
np.random.seed(TRAINING_CONFIG["random_seed"])
torch.manual_seed(TRAINING_CONFIG["random_seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(TRAINING_CONFIG["random_seed"])

# Set device for PyTorch - use GPU if available for faster training
device = torch.device(TRAINING_CONFIG["device"])
print(f"Using device: {device}")

# Import project modules
from src.data.database import DatabaseConnector
from src.data.data_loader import JobRecommendationDataset, ReplayBuffer
from src.environments.job_env import JobRecommendationEnv
from src.models.q_network import QNetwork
from src.models.world_model import WorldModel
from src.training.agent import DynaQAgent
from src.utils.visualizer import Visualizer
from src.utils.evaluator import Evaluator

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
    db = DatabaseConnector()  # Uses default connection string and db name from config
    print("Database connection established successfully.")
except Exception as e:
    print(f"Database connection error: {e}")
    raise

# %% [markdown]
# ## 5. Data Loading
# 
# We'll define utility functions to:
# 1. Retrieve candidate (applicant) embedding data
# 2. Sample a subset of jobs from the database
# 3. Retrieve job embedding vectors
# 
# These functions will help us efficiently access the data needed for training and evaluation.

# %%
# Function to get candidate embeddings from the database
def get_applicant_state(applicant_id):
    """
    Retrieve the embedding vector representing an applicant's state.
    
    Args:
        applicant_id: Unique identifier for the candidate
        
    Returns:
        torch.Tensor: State vector representing the candidate's skills and attributes
    """
    return db.get_applicant_state(applicant_id)

# Function to sample candidate jobs from the database
def sample_candidate_jobs(n=TRAINING_CONFIG["num_jobs"], filter_criteria=None):
    """
    Sample a subset of jobs to be considered for recommendation.
    
    Args:
        n: Number of jobs to sample
        filter_criteria: Optional dictionary for filtering jobs
        
    Returns:
        List[Dict]: List of job documents
    """
    return db.sample_candidate_jobs(n, filter_criteria)

# Function to get job embedding vectors
def get_job_vectors(job_ids):
    """
    Retrieve embedding vectors for a list of job IDs.
    
    Args:
        job_ids: List of job identifiers
        
    Returns:
        List[torch.Tensor]: List of job embedding vectors
    """
    return db.get_job_vectors(job_ids)

# %% [markdown]
# ## 6. Select a Target Candidate and Fetch Jobs
# 
# Now we'll:
# 1. Query the database to get a list of candidate IDs
# 2. Select a target candidate for personalized training
# 3. Retrieve the candidate's embedding vector
# 4. Sample jobs from the database
# 5. Retrieve embedding vectors for the sampled jobs

# %%
# Query the database to get a list of candidate IDs
try:
    # Get the candidates_text collection
    collection = db.db[db.collections["candidates_text"]]
    
    # Query for candidate IDs (limit to 10 for demonstration)
    candidate_ids = [doc["_id"] for doc in collection.find({}, {"_id": 1}).limit(TRAINING_CONFIG["num_candidates"])]
    
    if not candidate_ids:
        raise ValueError("No candidates found in the database")
    
    # Select the first candidate for demonstration
    target_candidate_id = candidate_ids[0]
    print(f"Selected target candidate ID: {target_candidate_id}")
    
    # Get the candidate embedding
    candidate_embedding = get_applicant_state(target_candidate_id)
    print(f"Candidate embedding shape: {candidate_embedding.shape}")
    
    # Sample jobs from the database
    sampled_jobs = sample_candidate_jobs(n=TRAINING_CONFIG["num_jobs"])
    job_ids = [job["_id"] for job in sampled_jobs]
    print(f"Sampled {len(job_ids)} jobs from the database")
    
    # Get job vectors for the sampled jobs
    job_vectors = get_job_vectors(job_ids)
    print(f"Fetched {len(job_vectors)} job vectors")
    
    # Convert job vectors to NumPy array for easier handling
    job_vectors_np = np.array([tensor.cpu().numpy() for tensor in job_vectors])
    print(f"Job vectors shape: {job_vectors_np.shape}")
    
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# %% [markdown]
# ## 7. Model Initialization
# 
# We'll initialize the neural networks used in the Dyna-Q algorithm:
# 
# 1. **Q-Network**: Approximates the value function mapping state-action pairs to expected returns
# 2. **Target Q-Network**: A copy of the Q-Network used for stable learning
# 3. **World Model**: Predicts next states and rewards based on current states and actions
# 
# Each network is configured according to the parameters specified in the configuration file.

# %%
# Initialize Q-Network
q_network = QNetwork(
    state_dim=MODEL_CONFIG["q_network"]["state_dim"],
    action_dim=MODEL_CONFIG["q_network"]["action_dim"],
    hidden_dims=MODEL_CONFIG["q_network"]["hidden_dims"],
    dropout_rate=MODEL_CONFIG["q_network"]["dropout_rate"],
    activation=MODEL_CONFIG["q_network"]["activation"]
).to(device)

# Initialize World Model
world_model = WorldModel(
    input_dim=MODEL_CONFIG["world_model"]["input_dim"],
    hidden_dims=MODEL_CONFIG["world_model"]["hidden_dims"],
    dropout_rate=MODEL_CONFIG["world_model"]["dropout_rate"],
    activation=MODEL_CONFIG["world_model"]["activation"]
).to(device)

# Initialize DynaQAgent
agent = DynaQAgent(
    state_dim=MODEL_CONFIG["q_network"]["state_dim"],
    action_dim=MODEL_CONFIG["q_network"]["action_dim"],
    q_network=q_network,
    world_model=world_model,
    training_strategy=reward_strategy,
    device=device,
    target_applicant_id=target_candidate_id
)

# Initialize Visualizer
visualizer = Visualizer()

# Initialize Evaluator
evaluator = Evaluator()

# Initialize environment with required parameters
env = JobRecommendationEnv(
    db_connector=db,
    reward_strategy=reward_strategy,
    random_seed=TRAINING_CONFIG["random_seed"]
)

# Set the target candidate ID for the environment
env.reset(applicant_id=target_candidate_id)

# Verify environment initialization
if not hasattr(env, 'current_state'):
    raise ValueError("Environment initialization failed - missing current_state")

print(f"Environment initialized with {reward_strategy} reward strategy.")

# %% [markdown]
# ## 8. LLM Integration (If Using LLM-Based Reward Strategy)
# 
# If we're using an LLM-based reward strategy (either "llm" or "hybrid"), we need to set up a language model to simulate candidate responses to job recommendations. This helps in generating more realistic rewards based on semantic understanding of both candidate profiles and job descriptions.
# 
# Note: This section will only execute if we're using an LLM-based strategy and running in a Colab environment with sufficient resources.

# %%
if STRATEGY_CONFIG["llm"]["enabled"] and IN_COLAB:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        # LLM configuration from config
        model_id = STRATEGY_CONFIG["llm"]["model_id"]
        
        # Configure quantization settings from config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=STRATEGY_CONFIG["llm"]["quantization"]["load_in_4bit"],
            bnb_4bit_quant_type=STRATEGY_CONFIG["llm"]["quantization"]["quant_type"],
            bnb_4bit_use_nested_quant=STRATEGY_CONFIG["llm"]["quantization"]["use_nested_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16 
        )
        
        print(f"Loading tokenizer for model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        print(f"Loading model {model_id} with 4-bit quantization...")
        
        # Load the model with quantization for efficiency
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        print(f"LLM model loaded successfully.")
        
        # Set up LLM in the environment for reward calculation and capture returned environment
        # setup_llm may return a new environment instance if current one doesn't support LLM
        env = env.setup_llm(llm_model, tokenizer)
    except Exception as e:
        print(f"Error loading LLM: {e}")
        print("Switching to cosine similarity reward strategy.")
        env.reward_strategy = "cosine"

# %% [markdown]
# ## 10. Training Setup
# 
# We'll now set up the training process using the DynaQAgent's built-in training interface.
# The agent handles:
# 1. Experience replay buffer
# 2. Q-Network and World Model updates
# 3. Loss tracking and metrics
# 4. Model checkpointing

# %%
# Initialize the DynaQAgent with all required components
agent = DynaQAgent(
    state_dim=MODEL_CONFIG["q_network"]["state_dim"],
    action_dim=MODEL_CONFIG["q_network"]["action_dim"],
    q_network=q_network,
    world_model=world_model,
    training_strategy=reward_strategy,
    device=device,
    target_applicant_id=target_candidate_id
)

print("Dyna-Q agent initialized successfully.")

# %% [markdown]
# ## 11. Pretraining Phase
# 
# In the pretraining phase, we'll use the agent's built-in pretraining method to:
# 1. Generate initial experiences
# 2. Train the Q-network and world model
# 3. Track training metrics
# 4. Save model checkpoints

# %%
# Define pretraining parameters from config
num_pretraining_samples = min(TRAINING_CONFIG["pretraining"]["num_samples"], len(job_ids))
num_pretraining_epochs = TRAINING_CONFIG["pretraining"]["num_epochs"]
batch_size = TRAINING_CONFIG["batch_size"]

print(f"Starting pretraining with {reward_strategy} strategy...")

# Generate pretraining data
pretraining_data = []
state = env.reset()

for i in tqdm(range(num_pretraining_samples), desc="Generating pretraining data"):
    job_idx = i % len(job_ids)
    job_id = job_ids[job_idx]
    job_vector = job_vectors_np[job_idx]
    
    next_state, reward, done, _ = env.step(job_id)
    pretraining_data.append((state, job_vector, reward, next_state))
    state = next_state

# Convert to numpy arrays
states = np.array([d[0] for d in pretraining_data])
actions = np.array([d[1] for d in pretraining_data])
rewards = np.array([d[2] for d in pretraining_data])
next_states = np.array([d[3] for d in pretraining_data])

# Run pretraining using agent's interface
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
    q_losses=pretraining_metrics['q_losses'],
    world_losses=pretraining_metrics['world_losses'],
    title="Pretraining Losses"
)

print("Pretraining completed successfully.")

# %% [markdown]
# ## 12. Main Training Loop
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
update_frequency = TRAINING_CONFIG["update_frequency"]

print("Starting main training loop...")

# Run training using agent's interface
training_metrics = agent.train(
    env=env,
    num_episodes=num_episodes,
    max_steps_per_episode=max_steps_per_episode,
    update_frequency=update_frequency
)

# Plot training metrics
visualizer.plot_training_metrics(
    q_losses=training_metrics['q_losses'],
    world_losses=training_metrics['world_losses'],
    rewards=training_metrics['episode_rewards'],
    title="Training Metrics"
)

print("Training completed successfully.")

# %% [markdown]
# ## 13. Evaluation
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

# Run evaluation using Evaluator
evaluation_results = evaluator.evaluate_agent(
    agent=agent,
    env=env,
    num_episodes=num_eval_episodes,
    baseline_strategy=baseline_strategy
)

# Plot evaluation results
visualizer.plot_evaluation_results(
    agent_rewards=evaluation_results['agent_rewards'],
    baseline_rewards=evaluation_results['baseline_rewards'],
    title="Evaluation Results"
)

print("Evaluation completed successfully.")

# %% [markdown]
# ## 14. Generate Job Recommendations
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

# Make a copy of job data for testing
test_job_ids = job_ids.copy()
test_job_vectors = job_vectors_np.copy()

# Reset the environment to get the initial state
state = env.reset()

# Generate top-K recommendations
for _ in range(num_recommendations):
    # Select best job according to Q-network
    job_id, action, q_value = agent.select_action(
        state, 
        test_job_ids, 
        test_job_vectors, 
        epsilon=test_epsilon
    )
    recommended_jobs.append(job_id)
    recommendation_scores.append(q_value)
    
    # Remove selected job from consideration for next selection
    job_idx = test_job_ids.index(job_id)
    test_job_ids.pop(job_idx)
    test_job_vectors = np.delete(test_job_vectors, job_idx, axis=0)

# Display recommendations with details
print("\n=== Top Job Recommendations ===\n")
for i, (job_id, score) in enumerate(zip(recommended_jobs, recommendation_scores)):
    print(f"Recommendation #{i+1}: [Q-Value: {score:.4f}]")
    print(f"Job ID: {job_id}")
    
    # Retrieve and display job details
    try:
        job_details = db.get_job_details(job_id)
        print(f"Title: {job_details.get('job_title', 'N/A')}")
        
        # Display truncated description
        description = job_details.get('description', 'N/A')
        print(f"Description: {description[:100]}..." if len(description) > 100 else f"Description: {description}")
        
        # Display technical skills if available
        if 'technical_skills' in job_details:
            print(f"Technical Skills: {', '.join(job_details['technical_skills'])}")
    except Exception as e:
        print(f"Error retrieving job details: {e}")
    print("-" * 50)

# %% [markdown]
# ## 15. Conclusion and Next Steps
# 
# In this notebook, we've implemented and demonstrated a Neural Dyna-Q job recommendation system. This approach combines the strengths of deep reinforcement learning with model-based planning to provide personalized job recommendations.
# 
# ### 15.1 Key Accomplishments
# 
# 1. **Data Integration**: Connected to the MongoDB database to retrieve real candidate and job data
# 2. **Neural Networks**: Implemented deep Q-network and world model for value function and dynamics prediction
# 3. **Dyna-Q Algorithm**: Combined direct RL with model-based planning for efficient learning
# 4. **Personalized Recommendations**: Generated job recommendations tailored to a specific candidate
# 
# ### 15.2 Potential Improvements
# 
# 1. **Extended Training**: Train for more episodes to improve recommendation quality
# 2. **Hyperparameter Tuning**: Optimize learning rates, network architectures, and other parameters
# 3. **Advanced Reward Functions**: Implement more sophisticated reward strategies using LLMs
# 4. **User Feedback**: Incorporate real user feedback to improve recommendations
# 
# ### 15.3 Applications
# 
# This system could be deployed as:
# - A personalized job recommendation service for job seekers
# - A candidate-job matching tool for recruiters
# - A component in a larger career guidance system

# %%
# Clean up resources
db.close()
print("Database connection closed.")
print("Notebook execution complete.")