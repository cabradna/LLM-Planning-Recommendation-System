import pytest
from src.config.config import (
    ENV_CONFIG,
    MODEL_CONFIG,
    TRAINING_CONFIG,
    DB_CONFIG,
    PRETRAINING_CONFIG,
    EVAL_CONFIG,
    STRATEGY_CONFIG
)

def test_env_config():
    """Test environment configuration parameters"""
    assert isinstance(ENV_CONFIG["random_seed"], int)
    assert isinstance(ENV_CONFIG["max_jobs_per_episode"], int)
    assert isinstance(ENV_CONFIG["reward_scaling"], float)
    assert isinstance(ENV_CONFIG["normalize_rewards"], bool)
    assert isinstance(ENV_CONFIG["use_tensor_cache"], bool)
    assert ENV_CONFIG["cache_device"] in ["cuda", "cpu"]

def test_model_config():
    """Test model configuration parameters"""
    q_network = MODEL_CONFIG["q_network"]
    assert isinstance(q_network["state_dim"], int)
    assert isinstance(q_network["action_dim"], int)
    assert isinstance(q_network["hidden_dims"], list)
    assert isinstance(q_network["dropout_rate"], float)
    assert q_network["activation"] in ["relu", "tanh", "sigmoid"]
    
    world_model = MODEL_CONFIG["world_model"]
    assert isinstance(world_model["input_dim"], int)
    assert isinstance(world_model["hidden_dims"], list)
    assert isinstance(world_model["dropout_rate"], float)
    assert world_model["activation"] in ["relu", "tanh", "sigmoid"]

def test_training_config():
    """Test training configuration parameters"""
    assert isinstance(TRAINING_CONFIG["gamma"], float)
    assert isinstance(TRAINING_CONFIG["lr"], float)
    assert isinstance(TRAINING_CONFIG["batch_size"], int)
    assert isinstance(TRAINING_CONFIG["target_update_freq"], int)
    assert isinstance(TRAINING_CONFIG["planning_steps"], int)
    assert isinstance(TRAINING_CONFIG["epsilon_start"], float)
    assert isinstance(TRAINING_CONFIG["epsilon_end"], float)
    assert isinstance(TRAINING_CONFIG["epsilon_decay"], float)
    assert isinstance(TRAINING_CONFIG["replay_buffer_size"], int)
    assert isinstance(TRAINING_CONFIG["min_replay_buffer_size"], int)
    assert isinstance(TRAINING_CONFIG["num_episodes"], int)
    assert isinstance(TRAINING_CONFIG["max_steps_per_episode"], int)
    assert isinstance(TRAINING_CONFIG["eval_frequency"], int)
    assert isinstance(TRAINING_CONFIG["save_frequency"], int)

def test_db_config():
    """Test database configuration parameters"""
    assert isinstance(DB_CONFIG["host"], str)
    assert isinstance(DB_CONFIG["port"], int)
    assert isinstance(DB_CONFIG["database"], str)
    assert isinstance(DB_CONFIG["collections"], dict)
    assert isinstance(DB_CONFIG["username"], str)
    assert isinstance(DB_CONFIG["password"], str)
    assert isinstance(DB_CONFIG["auth_file"], str)
    assert isinstance(DB_CONFIG["connection_string"], str)

def test_strategy_config():
    """Test strategy configuration parameters"""
    cosine = STRATEGY_CONFIG["cosine"]
    assert isinstance(cosine["enabled"], bool)
    assert isinstance(cosine["scale_reward"], bool)
    assert isinstance(cosine["similarity_threshold"], float)
    
    llm = STRATEGY_CONFIG["llm"]
    assert isinstance(llm["enabled"], bool)
    assert isinstance(llm["model_name"], str)
    assert isinstance(llm["model_id"], str)
    assert isinstance(llm["response_mapping"], dict)
    assert isinstance(llm["temperature"], float)
    assert isinstance(llm["max_tokens"], int)
    assert isinstance(llm["cache_responses"], bool)
    
    hybrid = STRATEGY_CONFIG["hybrid"]
    assert isinstance(hybrid["enabled"], bool)
    assert isinstance(hybrid["initial_cosine_weight"], float)
    assert isinstance(hybrid["final_cosine_weight"], float)
    assert isinstance(hybrid["annealing_episodes"], int)
    assert isinstance(hybrid["switch_episode"], int) 