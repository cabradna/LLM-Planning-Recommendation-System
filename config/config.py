"""
Configuration settings for the Dyna-Q Job Recommender model.
"""

import torch

# Hugging Face Configuration
HF_CONFIG = {
    "token_path": "hf_primary_token.txt"
}

# Database Configuration
DB_CONFIG = {
    "host": "cluster0.zqzq6hs.mongodb.net",
    "port": 27017,
    "database": "rl_jobsdb",
    "collections": {
        "jobs_text": "jobs_text",
        "job_embeddings": "job_embeddings",
        "candidates_text": "candidates_text",
        "candidates_embeddings": "candidates_embeddings"
    },
    "username": "default_user",  # Replace with actual username
    "password": "default_password",  # Replace with actual password
    "auth_file": "db_auth.txt",  # Path to the file containing database credentials
    "connection_string": "mongodb+srv://"  # Base connection string for MongoDB Atlas
}

# Model Architecture Configuration
MODEL_CONFIG = {
    # Neural network architecture
    "q_network": {
        "state_dim": 768,  # Dimension of applicant embeddings (was 384, updating to match actual env output)
        "action_dim": 384*4,  # Dimension of job embeddings (job title, tech skills, experience reqs, soft skills)
        "hidden_dims": [512, 256, 128],  # Hidden layer dimensions
        "dropout_rate": 0.2,  # Dropout rate
        "activation": "relu"  # Activation function
    },
    
    # World model network architecture
    "world_model": {
        "input_dim": 768+384*4,  # state_dim + action_dim (updating to match new state_dim)
        "hidden_dims": [512, 256],  # Hidden layer dimensions
        "dropout_rate": 0.2,  # Dropout rate
        "activation": "relu"  # Activation function
    }
}

# Training Configuration
TRAINING_CONFIG = {
    # Core reinforcement learning parameters
    "gamma": 0.9,  # Discount factor
    "lr": 0.001,  # Learning rate
    "batch_size": 64,  # Batch size for training
    "target_update_freq": 1000,  # Target network update frequency
    "planning_steps": 5,  # Number of planning steps in Dyna-Q
    
    # Exploration parameters
    "epsilon_start": 1.0,  # Initial exploration rate
    "epsilon_end": 0.01,  # Final exploration rate
    "epsilon_decay": 0.995,  # Decay rate for exploration
    
    # Buffer parameters
    "replay_buffer_size": 100000,  # Maximum replay buffer size
    "min_replay_buffer_size": 1000,  # Minimum buffer size before training starts

    # Training process
    "num_episodes": 100,  # Number of episodes
    "max_steps_per_episode": 100,  # Maximum steps per episode
    "eval_frequency": 10,  # Evaluation frequency (episodes)
    "save_frequency": 1000,  # Model saving frequency (episodes)

    # Data parameters
    "num_jobs": 100,  # Number of jobs to sample
    "num_candidates": 10,  # Number of candidates to consider

    # Pretraining parameters
    "pretraining": {
        "num_samples": 10000,  # Number of samples for pretraining
        "num_epochs": 20,  # Number of pretraining epochs
    },

    "random_seed": 42,  # Random seed for reproducibility
    
    # Hardware settings
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# Pretraining Configuration
PRETRAINING_CONFIG = {
    "dataset_size": 10000,  # Number of samples in pretraining dataset
    "num_epochs": 20,  # Number of pretraining epochs
    "batch_size": 128,  # Batch size for pretraining
    "lr": 0.001,  # Learning rate for pretraining
    "weight_decay": 1e-5,  # Weight decay for regularization
    "validation_split": 0.1  # Portion of data used for validation
}

# Evaluation Configuration
EVAL_CONFIG = {
    "num_eval_episodes": 100,  # Number of evaluation episodes
    "top_k_recommendations": 10,  # Number of top recommendations to consider
    "eval_metrics": ["cumulative_reward", "average_reward", "apply_rate"],  # Metrics to track
    "detailed_analysis": True,  # Whether to do detailed performance analysis
    "baseline_comparison": True  # Compare against baseline models
}

# Training Strategies Configuration
STRATEGY_CONFIG = {
    # Cosine Similarity Strategy
    "cosine": {
        "enabled": True,
        "description": "Direct semantic matching with cosine similarity between applicant and job embeddings",
        "scale_reward": True,  # Scale cosine similarity from [-1,1] to [0,1]
        "similarity_threshold": 0.5  # Minimum similarity to consider a match (if scaling is disabled)
    },
    
    # LLM Feedback Strategy
    "llm": {
        "enabled": True,
        "description": "Uses LLM to simulate user responses based on applicant-job match",
        "model_name": "mistralai/Mistral-7B-Instruct-v0.1",  # LLM model to use
        "model_id": "mistralai/Mistral-7B-Instruct-v0.1",    # LLM model identifier for Hugging Face
        "response_mapping": {  # Mapping from LLM response to reward values
            "APPLY": 1.0,
            "SAVE": 0.5,
            "CLICK": 0.0,
            "IGNORE": -0.1
        },
        "temperature": 0.2,  # LLM sampling temperature
        "max_tokens": 50,  # Maximum tokens in LLM response
        "cache_responses": True,  # Whether to cache LLM responses for similar inputs
        "quantization": {
            "load_in_4bit": True,  # Use 4-bit quantization to reduce memory usage
            "quant_type": "nf4",  # Quantization type - normalized float 4
            "use_nested_quant": False  # Whether to use nested quantization
        }
    },
    
    # Hybrid Strategy
    "hybrid": {
        "enabled": True,
        "description": "Combines cosine similarity with LLM feedback, allowing for transition between strategies",
        "initial_cosine_weight": 1.0,  # Initial weight for cosine similarity (vs LLM feedback)
        "final_cosine_weight": 0.0,  # Final weight for cosine similarity after annealing
        "annealing_episodes": 1000,  # Number of episodes to anneal from initial to final weight
        "switch_episode": 1500  # Episode to completely switch from cosine to LLM (overrides annealing)
    }
}

# Paths Configuration
PATH_CONFIG = {
    "model_dir": "../models",  # Directory to save models
    "log_dir": "../logs",  # Directory to save logs
    "data_dir": "../data",  # Directory to save/load data
    "results_dir": "../results",  # Directory to save evaluation results
    "repo_url": "https://github.com/cabradna/LLM-Planning-Recommendation-System.git"  # Repository URL
} 