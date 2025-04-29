# Dyna-Q Job Recommender Neural Model

## Overview

This project implements a neural network-based job recommendation system using the Dyna-Q reinforcement learning algorithm. The system combines model-free RL learning with model-based planning to provide personalized job recommendations for individual applicants. It addresses the cold-start problem in recommendation systems by pre-training the Q-network and World Model with preferences distilled from multiple sources, including Large Language Model (LLM) simulations.

The core philosophy of this system is to build specialized recommendation agents for individual applicants rather than a one-size-fits-all approach. Each trained agent focuses exclusively on optimizing recommendations for a single target applicant throughout the entire training process.

## Project Structure

```
neural_model_repo/
│
├── config/                 # Configuration files for the system
├── docs/                   # Documentation with system specification
│   ├── training_procedure.md  # Detailed explanation of the single-applicant approach
│   ├── rl_formulation.md     # Reinforcement learning problem formulation
│   ├── database_structure.md # MongoDB database schema specification
│   └── model_description.md  # Neural network architecture descriptions
├── scripts/                # Utility scripts
├── src/                    # Source code
│   ├── data/               # Data handling and database interfaces
│   │   ├── data_loader.py  # Data loading utilities and replay buffer
│   │   └── database.py     # MongoDB integration for job and applicant data
│   ├── environments/       # Simulation environments
│   │   └── job_env.py      # Job recommendation environment with multiple reward strategies
│   ├── models/             # Neural network models
│   │   ├── q_network.py    # Q-Network for value function approximation
│   │   └── world_model.py  # World Model for environment dynamics prediction
│   ├── training/           # Training utilities
│   │   └── agent.py        # Dyna-Q agent implementation for single-applicant focus
│   ├── utils/              # Utility functions for visualization and evaluation
│   ├── evaluate.py         # Performance evaluation system
│   ├── main.py             # Unified command-line interface
│   ├── pretraining.py      # Pre-training system with multiple strategies
│   └── train.py            # Model training implementation
├── tests/                  # Test cases
├── .gitignore              # Git ignore configuration
├── jupyter_example.txt     # Example Jupyter notebook usage
├── neural_dyna_q           # Executable wrapper
├── neural_dyna_q_notebook.py # Notebook helper
├── requirements.txt        # Project dependencies
├── setup.py                # Package configuration
└── README.md               # This file
```

## Features

- **Single-Applicant Specialization**: Each agent is trained specifically for one applicant, focusing the entire training process on optimizing recommendations for that individual
- **Neural Q-Network**: Deep neural network for value function approximation in recommendation decision making
- **Neural World Model**: Predictive model of environment dynamics for Dyna-Q planning, enhancing sample efficiency
- **Multiple Reward Generation Strategies**: 
  - **Cosine Similarity**: Uses semantic similarity between applicant and job embeddings
  - **LLM Feedback**: Leverages LLM to simulate realistic applicant responses to job recommendations
  - **Hybrid Approach**: Combines both methods with customizable weighting or phased training
- **MongoDB Integration**: Structured data retrieval for job profiles and applicant information
- **Dyna-Q Algorithm Implementation**: Combines direct reinforcement learning with model-based planning
- **Comprehensive Pre-Training**: Addresses the cold-start problem through multiple pre-training approaches
- **Evaluation Framework**: Tools for comparing different strategies and measuring cold-start performance

## MongoDB Database Structure

The system connects to a MongoDB database with the following collections:
- `jobs_text`: Contains job posting text data
- `job_embeddings`: Stores vector embeddings for jobs
- `candidates_text`: Contains candidate profile information
- `candidates_embeddings`: Stores vector embeddings for applicants' skills and experience

### Authentication Files

The system requires two authentication files in the project root directory:

1. `db_auth.txt`: Contains MongoDB Atlas credentials
   - First line: MongoDB username
   - Second line: MongoDB password
   - Example:
     ```
     your_mongodb_username
     your_mongodb_password
     ```

2. `hf_primary_token.txt`: Contains the Hugging Face API token for LLM-based strategies
   - Single line containing the API token
   - Example:
     ```
     hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
     ```

These files are used by the system to authenticate with MongoDB Atlas and Hugging Face services. They should be placed in the project root directory and are referenced in the configuration file (`config/config.py`).

## Installation

1. Clone the repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Install the package in development mode:
   ```
   pip install -e .
   ```

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- NumPy 1.20+
- MongoDB 4.0+
- Matplotlib 3.4+
- tqdm 4.60+
- transformers (for LLM-based strategies)
- pytest 6.2.5+ (for testing)

## Pre-Training Process

The pre-training process involves creating datasets using one of three strategies:

1. **Cosine Similarity**: Creates reward signals based on semantic similarity between applicant and job embeddings
2. **LLM Feedback**: Uses a large language model to simulate applicant responses to jobs, mapping decisions like "APPLY", "SAVE", or "IGNORE" to reward values
3. **Hybrid Approach**: Combines both methods, weighting them based on configuration

Pre-training teaches the Q-Network and World Model to predict these reward values before deploying the system for real interactions.

## Training Philosophy

The system follows a **single-applicant specialization approach**:
- Each agent is trained specifically for one target applicant
- The applicant's profile information remains constant throughout the training run
- Training focuses entirely on optimizing job recommendations for that specific individual
- This approach allows for deeply personalized recommendations rather than generic matches

## Quick Start Guide

### 1. Set Up Environment

```bash
# Install required packages
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 2. Explore Available Commands

```bash
# Show information about the system and available commands
python src/main.py info
```

### 3. Pretrain Models

```bash
# Pretrain models using cosine similarity (baseline approach)
python src/main.py pretraining --method cosine --save_path models/cosine_pretrained

# Pretrain models using LLM-based preference distillation
python src/main.py pretraining --method llm --save_path models/llm_pretrained

# Pretrain models using hybrid approach
python src/main.py pretraining --method hybrid --cosine_weight 0.3 --save_path models/hybrid_pretrained
```

### 4. Train Agents for Specific Applicants

```bash
# Train a baseline agent for a specific applicant using cosine similarity feedback
python src/main.py train --train_baseline --strategy cosine --num_episodes 100 --save_path models/applicant1_baseline

# Fine-tune a pretrained agent for a specific applicant using LLM feedback
python src/main.py train --train_pretrained --pretrained_model_path models/llm_pretrained --strategy llm --num_episodes 100 --save_path models/applicant1_llm
```

### 5. Evaluate Models

```bash
# Compare baseline and pretrained agents for a specific applicant
python src/main.py evaluate --compare --baseline_path models/applicant1_baseline --pretrained_path models/applicant1_llm

# Evaluate cold-start performance
python src/main.py evaluate --cold_start --pretrained_path models/applicant1_llm
```

## Command-Line Interface

The system provides a unified command-line interface through `src/main.py` with the following main commands:

- `pretraining`: Pretrain models using one of the three reward strategies
- `train`: Train models for specific applicants using interaction data
- `evaluate`: Evaluate trained models and compare performance
- `simulate`: Run simulations with trained models
- `info`: Display information about the system

Each command supports various options. Run `python src/main.py <command> --help` for detailed information.

## Development and Testing

Run the test suite to verify the installation and functionality:

```bash
# Run all tests
pytest tests/
```

## Documentation

For more detailed information about the system design and implementation:

- `docs/training_procedure.md`: Explains the single-applicant training approach in detail
- `docs/rl_formulation.md`: Provides the reinforcement learning problem formulation
- `docs/database_structure.md`: Details the MongoDB database schema
- `docs/model_description.md`: Describes the neural network architectures

## License

MIT License 