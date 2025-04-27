# Dyna-Q Job Recommender Neural Model

## Overview

This project implements a neural network-based model for job recommendations using the Dyna-Q reinforcement learning algorithm. It aims to solve the cold-start problem in recommendation systems by pre-training the Q-network with preferences distilled from a Large Language Model (LLM).

## Project Structure

```
neural_model/
│
├── config/            # Configuration files
├── data/              # Data storage (not tracked in git)
├── docs/              # Documentation
├── src/               # Source code
│   ├── data/          # Data handling modules
│   ├── models/        # Neural network models
│   ├── utils/         # Utility functions
│   ├── environments/  # Simulation environments
│   └── training/      # Training scripts
├── tests/             # Test cases
├── requirements.txt   # Project dependencies
├── setup.py           # Package configuration
└── README.md          # This file
```

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

## Features

- Q-Network for value function approximation
- World Model for environment dynamics prediction
- MongoDB integration for data retrieval
- Dyna-Q algorithm implementation
- LLM-based pre-training capability
- Evaluation utilities

## Quick Start Guide

This guide will help you get up and running with the neural model quickly:

### 1. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 2. Set Up Test Database

```bash
# Set up a test MongoDB database with dummy data
python scripts/init_test_db.py
```

### 3. Run Tests

```bash
# Run all tests to verify the installation
python tests/run_tests.py
```

### 4. Explore the Command-Line Interface

```bash
# Show available commands
python src/main.py info
```

### 5. Try a Basic Workflow

```bash
# Generate synthetic data and pretrain models
python src/main.py pretraining --generate_data --save_path ../models/pretrained

# Train a baseline agent
python src/main.py train --train_baseline --num_episodes 100

# Run an evaluation
python src/main.py evaluate --evaluate_baseline --baseline_path ../models/baseline
```

## Usage

The main entry points are:

- `src/train.py`: For training the model
- `src/evaluate.py`: For evaluating model performance
- `src/pretraining.py`: For LLM-based pre-training

## Requirements

- Python 3.8+
- PyTorch 2.0+
- MongoDB 4.0+
- Other dependencies in requirements.txt 