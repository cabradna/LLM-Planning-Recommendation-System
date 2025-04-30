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
# Create a database connector to access MongoDB Atlas
try:
    # Initialize database connector using configuration
    db_connector = DatabaseConnector()  # Uses default connection string and db name from config
    print("Database connection established successfully.")
except Exception as e:
    print(f"Database connection error: {e}")
    print("Please ensure MongoDB Atlas credentials are properly set in environment variables.")
    print("Continuing with local environment...")

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
    return db_connector.get_applicant_state(applicant_id)

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
    return db_connector.sample_candidate_jobs(n, filter_criteria)

# Function to get job embedding vectors
def get_job_vectors(job_ids):
    """
    Retrieve embedding vectors for a list of job IDs.
    
    Args:
        job_ids: List of job identifiers
        
    Returns:
        List[torch.Tensor]: List of job embedding vectors
    """
    return db_connector.get_job_vectors(job_ids)

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
    collection = db_connector.db[db_connector.collections["candidates_text"]]
    
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
# Extract dimensions and hyperparameters from config
state_dim = MODEL_CONFIG["q_network"]["state_dim"]
action_dim = MODEL_CONFIG["q_network"]["action_dim"]
hidden_dims_q = MODEL_CONFIG["q_network"]["hidden_dims"]
hidden_dims_world = MODEL_CONFIG["world_model"]["hidden_dims"]
dropout_rate = MODEL_CONFIG["q_network"]["dropout_rate"]

# Initialize Q-Network for value function approximation
q_network = QNetwork(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dims=hidden_dims_q,
    dropout=dropout_rate
).to(device)

# Initialize target Q-Network (copy of Q-Network)
# This is used for stable learning by providing a fixed target
target_q_network = QNetwork(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dims=hidden_dims_q,
    dropout=dropout_rate
).to(device)

# Copy parameters from q_network to target_q_network
target_q_network.load_state_dict(q_network.state_dict())
target_q_network.eval()  # Set to evaluation mode (no gradient updates)

# Initialize World Model for environment dynamics prediction
world_model = WorldModel(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dims=hidden_dims_world,
    dropout=dropout_rate
).to(device)

# Initialize optimizers for neural networks
learning_rate_q = TRAINING_CONFIG["lr"]
learning_rate_world = TRAINING_CONFIG["lr"]
q_optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate_q)
world_optimizer = torch.optim.Adam(world_model.parameters(), lr=learning_rate_world)

print("Models initialized successfully.")

# %% [markdown]
# ## 8. Environment Setup
# 
# Now we'll set up the job recommendation environment, which simulates interactions between candidates and job recommendations. The environment:
# 
# 1. Maintains the state of the candidate
# 2. Processes job recommendation actions
# 3. Computes rewards based on the selected strategy
# 4. Returns the next state after each action
# 
# We can use different reward strategies:
# - **Cosine similarity**: Direct semantic matching between candidate and job embeddings
# - **LLM feedback**: Using a language model to simulate candidate responses
# - **Hybrid**: A combination of cosine similarity and LLM feedback

# %%
# Get reward strategy settings from config
reward_strategy = STRATEGY_CONFIG["reward_strategy"]

# If using hybrid strategy, set the weight for cosine similarity
cosine_weight = STRATEGY_CONFIG["hybrid"]["initial_cosine_weight"] if reward_strategy == "hybrid" else None

# Determine if we need to use an LLM for reward calculation
use_llm = reward_strategy in ["llm", "hybrid"]

# Initialize the job recommendation environment
env = JobRecommendationEnv(
    candidate_embedding=candidate_embedding.cpu().numpy(),
    job_vectors=job_vectors_np,
    job_ids=job_ids,
    reward_strategy=reward_strategy,
    cosine_weight=cosine_weight,
    use_llm=use_llm,
    device=device
)

print(f"Environment initialized with {reward_strategy} reward strategy.")

# %% [markdown]
# ## 9. LLM Integration (If Using LLM-Based Reward Strategy)
# 
# If we're using an LLM-based reward strategy (either "llm" or "hybrid"), we need to set up a language model to simulate candidate responses to job recommendations. This helps in generating more realistic rewards based on semantic understanding of both candidate profiles and job descriptions.
# 
# Note: This section will only execute if we're using an LLM-based strategy and running in a Colab environment with sufficient resources.

# %%
if use_llm and IN_COLAB:
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
        
        # Set up LLM in the environment for reward calculation
        env.setup_llm(llm_model, tokenizer)
    except Exception as e:
        print(f"Error loading LLM: {e}")
        print("Switching to cosine similarity reward strategy.")
        env.reward_strategy = "cosine"
        env.use_llm = False

# %% [markdown]
# ## 10. Replay Buffer Setup
# 
# The replay buffer stores experiences (state, action, reward, next_state) that the agent encounters during training. This allows:
# 
# 1. More efficient use of experiences through batch learning
# 2. Breaking correlation between sequential experiences
# 3. Stabilizing the learning process
# 
# We'll also initialize the Dyna-Q agent, which combines the Q-Network, World Model, and replay buffer to implement the Dyna-Q algorithm.

# %%
# Initialize replay buffer with capacity from config
buffer_capacity = TRAINING_CONFIG["replay_buffer_size"]
replay_buffer = ReplayBuffer(capacity=buffer_capacity)
print(f"Replay buffer initialized with capacity {buffer_capacity}.")

# Initialize the Dyna-Q agent
agent = DynaQAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    q_network=q_network,
    target_q_network=target_q_network,
    world_model=world_model,
    q_optimizer=q_optimizer,
    world_optimizer=world_optimizer,
    replay_buffer=replay_buffer,
    device=device
)
print("Dyna-Q agent initialized successfully.")

# %% [markdown]
# ## 11. Pretraining Phase
# 
# In the pretraining phase, we generate a dataset of experiences and train both the Q-network and world model using supervised learning. This helps to:
# 
# 1. Initialize the networks with reasonable parameters
# 2. Address the cold-start problem
# 3. Provide a baseline for further learning
# 
# We'll generate experiences using the environment's reward function and then train the networks on batches of these experiences.

# %%
# Define pretraining hyperparameters from config
num_pretraining_samples = min(TRAINING_CONFIG["pretraining"]["num_samples"], len(job_ids))
batch_size = TRAINING_CONFIG["batch_size"]
num_pretraining_epochs = TRAINING_CONFIG["pretraining"]["num_epochs"]

print(f"Starting pretraining with {reward_strategy} strategy...")

# Initialize datasets to store experiences
pretraining_states = []
pretraining_actions = []
pretraining_rewards = []
pretraining_next_states = []

# Reset the environment to get the initial state
state = env.reset()

# Generate experiences by sampling jobs and observing rewards
for i in tqdm(range(num_pretraining_samples), desc="Generating pretraining data"):
    # Sample a job (cycle through jobs if needed)
    job_idx = i % len(job_ids)
    job_id = job_ids[job_idx]
    job_vector = job_vectors_np[job_idx]
    
    # Get reward using the environment's reward function
    next_state, reward, done, _ = env.step(job_id)
    
    # Store the experience
    pretraining_states.append(state)
    pretraining_actions.append(job_vector)
    pretraining_rewards.append(reward)
    pretraining_next_states.append(next_state)
    
    # Set next state as current state for next iteration
    state = next_state

# Convert to numpy arrays for efficient processing
pretraining_states = np.array(pretraining_states)
pretraining_actions = np.array(pretraining_actions)
pretraining_rewards = np.array(pretraining_rewards)
pretraining_next_states = np.array(pretraining_next_states)

# Create PyTorch dataset for batch training
pretrain_dataset = JobRecommendationDataset(
    states=pretraining_states,
    actions=pretraining_actions,
    rewards=pretraining_rewards,
    next_states=pretraining_next_states
)

# Create data loader for batch processing
pretrain_loader = torch.utils.data.DataLoader(
    pretrain_dataset,
    batch_size=batch_size,
    shuffle=True
)

print(f"Pretraining dataset created with {len(pretrain_dataset)} samples.")

# %% [markdown]
# ### 11.1 Pretrain Q-Network and World Model
# 
# Now we'll train our models on the pretraining dataset. For each epoch, we:
# 
# 1. Process batches of experiences
# 2. Update the Q-Network to predict expected returns
# 3. Update the World Model to predict next states and rewards
# 4. Track and report loss metrics
# 
# This pretraining gives the models a head start before online learning.

# %%
# Initialize lists to track loss metrics
q_losses = []
world_losses = []
gamma = TRAINING_CONFIG["gamma"]  # Discount factor for future rewards

# Training loop for pretraining
for epoch in range(num_pretraining_epochs):
    epoch_q_loss = 0
    epoch_world_loss = 0
    
    # Process batches of experiences
    for batch in tqdm(pretrain_loader, desc=f"Pretraining Epoch {epoch+1}/{num_pretraining_epochs}"):
        # Get batch data and transfer to device
        states, actions, rewards, next_states = [b.to(device) for b in batch]
        
        # Update Q-Network to predict expected returns
        q_loss = agent.update_q_network(states, actions, rewards, next_states, gamma)
        
        # Update World Model to predict next states and rewards
        world_loss = agent.update_world_model(states, actions, rewards, next_states)
        
        # Accumulate losses for this epoch
        epoch_q_loss += q_loss.item()
        epoch_world_loss += world_loss.item()
    
    # Calculate average losses for this epoch
    avg_q_loss = epoch_q_loss / len(pretrain_loader)
    avg_world_loss = epoch_world_loss / len(pretrain_loader)
    
    # Store losses for visualization
    q_losses.append(avg_q_loss)
    world_losses.append(avg_world_loss)
    
    # Report progress
    print(f"Epoch {epoch+1}/{num_pretraining_epochs} - Q-Loss: {avg_q_loss:.4f}, World-Loss: {avg_world_loss:.4f}")

# Visualize training progress
plt.figure(figsize=(12, 5))

# Plot Q-Network loss
plt.subplot(1, 2, 1)
plt.plot(q_losses)
plt.title("Q-Network Loss During Pretraining")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Plot World Model loss
plt.subplot(1, 2, 2)
plt.plot(world_losses)
plt.title("World Model Loss During Pretraining")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 12. Online Learning Phase (Dyna-Q)
# 
# Now we'll implement the full Dyna-Q algorithm, which combines:
# 
# 1. **Direct RL**: Learning from actual experiences in the environment
# 2. **Model-Based Planning**: Using the world model to simulate experiences
# 3. **Exploration**: Using an epsilon-greedy policy to balance exploration and exploitation
# 
# This phase allows the agent to refine its policy based on both real and simulated experiences.

# %%
# Define Dyna-Q hyperparameters from config
num_episodes = TRAINING_CONFIG["num_episodes"]
max_steps_per_episode = TRAINING_CONFIG["max_steps_per_episode"]
num_planning_steps = TRAINING_CONFIG["planning_steps"]
target_update_freq = TRAINING_CONFIG["target_update_freq"]
gamma = TRAINING_CONFIG["gamma"]
epsilon_start = TRAINING_CONFIG["epsilon_start"]
epsilon_end = TRAINING_CONFIG["epsilon_end"]
epsilon_decay = TRAINING_CONFIG["epsilon_decay"]

# Lists to store metrics for evaluation
episode_rewards = []
avg_q_values = []

# Training loop over episodes
for episode in range(num_episodes):
    # Reset environment and calculate epsilon for this episode
    state = env.reset()
    epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
    
    # Initialize metrics for this episode
    total_reward = 0
    total_q_value = 0
    
    # Episode loop over steps
    for step in range(max_steps_per_episode):
        # Select action using epsilon-greedy policy
        if np.random.random() < epsilon:
            # Exploration: select random job
            job_idx = np.random.randint(0, len(job_ids))
            job_id = job_ids[job_idx]
            action = job_vectors_np[job_idx]
            q_value = 0  # No Q-value for random action
        else:
            # Exploitation: select best job according to Q-network
            job_id, action, q_value = agent.select_action(
                state, 
                job_ids, 
                job_vectors_np, 
                epsilon=0  # No exploration within select_action
            )
        
        # Track Q-value for this step
        total_q_value += q_value
        
        # Take action in environment and observe results
        next_state, reward, done, _ = env.step(job_id)
        total_reward += reward
        
        # Store experience in replay buffer
        agent.replay_buffer.add(state, action, reward, next_state)
        
        # Direct RL update (if enough experiences are available)
        if len(agent.replay_buffer) >= batch_size:
            agent.update_from_replay_buffer(batch_size, gamma)
        
        # Planning phase (Dyna-Q) - using world model to simulate experiences
        agent.planning(num_planning_steps, gamma)
        
        # Update target network periodically for stable learning
        if step % target_update_freq == 0:
            agent.update_target_network()
        
        # Move to next state
        state = next_state
        
        # End episode if done signal received
        if done:
            break
    
    # Calculate average Q-value for this episode
    avg_q_value = total_q_value / max_steps_per_episode
    
    # Store episode metrics for visualization
    episode_rewards.append(total_reward)
    avg_q_values.append(avg_q_value)
    
    # Report episode stats
    print(f"Episode {episode+1}/{num_episodes} - Reward: {total_reward:.4f}, Avg Q-Value: {avg_q_value:.4f}, Epsilon: {epsilon:.4f}")

# %% [markdown]
# ## 13. Evaluation and Visualization
# 
# Let's evaluate our trained model and visualize the learning progress. We'll look at:
# 
# 1. Episode rewards over time - indicates how well the agent is performing
# 2. Average Q-values over time - indicates the agent's confidence in its actions
# 
# These metrics help us understand the learning dynamics and performance of our agent.

# %%
# Create visualization of training metrics
plt.figure(figsize=(12, 5))

# Plot episode rewards
plt.subplot(1, 2, 1)
plt.plot(episode_rewards)
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True, alpha=0.3)

# Plot average Q-values
plt.subplot(1, 2, 2)
plt.plot(avg_q_values)
plt.title("Average Q-Value per Episode")
plt.xlabel("Episode")
plt.ylabel("Avg Q-Value")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

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
        job_details = db_connector.get_job_details(job_id)
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
db_connector.close()
print("Database connection closed.")
print("Notebook execution complete.")