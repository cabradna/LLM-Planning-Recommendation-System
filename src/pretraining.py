"""
Pretraining script for the Dyna-Q job recommender model.

This script handles pretraining of the Q-network and world model using
different strategies to mitigate the cold-start problem:
1. Cosine similarity (baseline)
2. LLM simulation
3. Hybrid (LLM + cosine)
"""

import os
import sys
import logging
import argparse
import torch
import numpy as np
import random
import json
import time
from typing import List, Dict, Tuple, Optional, Any
from torch.utils.data import Dataset, DataLoader

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import MODEL_CONFIG, PRETRAINING_CONFIG, PATH_CONFIG, TRAINING_CONFIG, HF_CONFIG
from config.config import STRATEGY_CONFIG, ENV_CONFIG

# Import modules
from data.database import DatabaseConnector
from data.tensor_cache import TensorCache
from data.data_loader import JobRecommendationDataset, create_pretraining_data_loader
from models.q_network import QNetwork
from models.world_model import WorldModel
from utils.visualizer import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pretraining.log"),
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

def load_cosine_data_using_tensor_cache(tensor_cache: TensorCache, 
                                       target_applicant_id: str,
                                       limit: int = PRETRAINING_CONFIG["dataset_size"],
                                       device: str = TRAINING_CONFIG["device"]) -> JobRecommendationDataset:
    """
    Load data using TensorCache and create a dataset for pretraining using cosine similarity.
    
    Args:
        tensor_cache: TensorCache with preloaded tensors.
        target_applicant_id: The applicant ID to generate data for.
        limit: Maximum number of examples to load.
        device: Device to load tensors to.
        
    Returns:
        JobRecommendationDataset: Dataset with examples from tensor cache.
    """
    logger.info(f"Loading data from tensor cache using cosine similarity, limit: {limit}")
    
    # Prepare lists for dataset
    states = []
    actions = []
    rewards = []
    next_states = []
    
    try:
        # Get state vector for target applicant
        initial_state_tensor = tensor_cache.get_applicant_state(target_applicant_id)
        if initial_state_tensor is None:
            logger.error(f"No state tensor found for applicant {target_applicant_id}")
            raise ValueError(f"No state tensor found for applicant {target_applicant_id}")
            
        logger.info(f"Retrieved state tensor for applicant {target_applicant_id} with shape {initial_state_tensor.shape}")
        
        # Get valid job indices from tensor cache
        valid_job_indices = tensor_cache.get_valid_job_indices()
        
        if not valid_job_indices:
            logger.error("No valid job indices found in tensor cache")
            raise ValueError("No valid job indices found in tensor cache")
            
        logger.info(f"Found {len(valid_job_indices)} valid jobs in tensor cache")
        
        # Calculate all cosine similarities at once for efficiency
        all_rewards = tensor_cache.calculate_cosine_similarities(initial_state_tensor)
        if STRATEGY_CONFIG["cosine"]["scale_reward"]:
            all_rewards = (all_rewards + 1) / 2  # Scale from [-1, 1] to [0, 1]
        
        # Limit the number of samples if needed
        num_samples = min(limit, len(valid_job_indices))
        indices_to_use = valid_job_indices[:num_samples]
        
        # Generate samples
        for idx in indices_to_use:
            # Get job vector
            action_tensor = tensor_cache.get_job_vector_by_index(idx)
            
            # Use precomputed reward
            reward = all_rewards[idx].item()
            
            # For simplicity in pretraining, next_state = state (static state)
            next_state = initial_state_tensor
            
            # Add to lists
            states.append(initial_state_tensor)
            actions.append(action_tensor)
            rewards.append(reward)
            next_states.append(next_state)
                
        logger.info(f"Created dataset with {len(states)} valid examples using cosine similarity")
        
        # Convert lists to tensors and create dataset
        states_tensor = torch.stack(states).to(device)
        actions_tensor = torch.stack(actions).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states_tensor = torch.stack(next_states).to(device)
        
        dataset = JobRecommendationDataset(states_tensor, actions_tensor, rewards_tensor, next_states_tensor)
        return dataset
        
    except Exception as e:
        logger.error(f"Error loading data from tensor cache: {e}")
        raise ValueError(f"Failed to load data from tensor cache: {e}")

# Legacy function - keep for backward compatibility
def load_cosine_data_from_database(db_connector: DatabaseConnector, limit: int = PRETRAINING_CONFIG["dataset_size"]) -> JobRecommendationDataset:
    """
    Load data from the database and create a dataset for pretraining using cosine similarity.
    
    Args:
        db_connector: Database connector for fetching data.
        limit: Maximum number of examples to load.
        
    Returns:
        JobRecommendationDataset: Dataset with examples from the database.
    """
    logger.info(f"Loading data from database using cosine similarity, limit: {limit}")
    
    # Prepare lists for dataset
    states = []
    actions = []
    rewards = []
    next_states = []
    
    try:
        # Get a list of candidate IDs from the database
        candidates_collection = db_connector.db[db_connector.collections["candidates_embeddings"]]
        candidate_docs = list(candidates_collection.find().limit(limit))
        
        if not candidate_docs:
            logger.error("No candidate data found in the database")
            raise ValueError("No candidate data found in the database")
            
        logger.info(f"Found {len(candidate_docs)} candidates in the database")
        
        # Get job samples to use as actions
        job_samples = db_connector.sample_candidate_jobs(n=min(100, limit))
        job_ids = [job.get("original_job_id") for job in job_samples if job.get("original_job_id")]
        
        if not job_ids:
            logger.error("No job IDs found in the database")
            raise ValueError("No job IDs found in the database")
            
        logger.info(f"Found {len(job_ids)} job IDs for actions")
        
        # Process each candidate
        for i, candidate_doc in enumerate(candidate_docs):
            try:
                # Get candidate ID
                candidate_id = candidate_doc.get("original_candidate_id")
                if not candidate_id:
                    logger.warning(f"Candidate {i} missing original_candidate_id, skipping")
                    continue
                
                # Get state vector for candidate
                state = db_connector.get_applicant_state(candidate_id)
                
                # Sample a job for this candidate
                job_id = random.choice(job_ids)
                
                # Get job vector
                job_vectors = db_connector.get_job_vectors([job_id])
                if not job_vectors:
                    logger.warning(f"No job vector found for job_id {job_id}, skipping")
                    continue
                    
                action = job_vectors[0]
                
                # For simplicity in pretraining, next_state = state (static state)
                next_state = state
                
                # CONCEPTUAL NOTE: Computing the "ground truth" reward for this state-action pair
                # This reward value will be used as the target for model training/validation
                # In cosine strategy, we're assuming semantic similarity = reward value
                reward = float(torch.cosine_similarity(
                    state.unsqueeze(0), 
                    action.unsqueeze(0), 
                    dim=1
                ).item())
                
                # Scale reward to [0, 1] range
                reward = (reward + 1) / 2
                
                # Add to lists - these precomputed rewards will be used as the "correct answers"
                # that the model should learn to predict for these state-action pairs
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                
                # Log progress
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(candidate_docs)} candidates")
                    
            except Exception as e:
                logger.error(f"Error processing candidate {i}: {e}")
                continue
                
        logger.info(f"Created dataset with {len(states)} valid examples using cosine similarity")
        
        # Create dataset with precomputed reward values
        dataset = JobRecommendationDataset(states, actions, rewards, next_states)
        return dataset
        
    except Exception as e:
        logger.error(f"Error loading data from database: {e}")
        raise ValueError(f"Failed to load data from database: {e}")

def load_llm_data_from_database(db_connector: DatabaseConnector, llm_model=None, tokenizer=None, 
                              device="cpu", limit: int = PRETRAINING_CONFIG["dataset_size"]) -> JobRecommendationDataset:
    """
    Load data from the database and create a dataset for pretraining using LLM simulation.
    
    Args:
        db_connector: Database connector for fetching data.
        llm_model: LLM model for simulation.
        tokenizer: Tokenizer for the LLM model.
        device: Device to run the LLM on.
        limit: Maximum number of examples to load.
        
    Returns:
        JobRecommendationDataset: Dataset with examples from the database.
    """
    logger.info(f"Loading data from database using LLM simulation, limit: {limit}")
    
    if llm_model is None or tokenizer is None:
        logger.error("LLM model and tokenizer are required for LLM simulation")
        raise ValueError("LLM model and tokenizer are required for LLM simulation")
    
    # Prepare lists for dataset
    states = []
    actions = []
    rewards = []
    next_states = []
    
    try:
        # Get a list of candidate IDs from the database
        candidates_collection = db_connector.db[db_connector.collections["candidates_embeddings"]]
        candidate_docs = list(candidates_collection.find().limit(limit))
        
        if not candidate_docs:
            logger.error("No candidate data found in the database")
            raise ValueError("No candidate data found in the database")
            
        logger.info(f"Found {len(candidate_docs)} candidates in the database")
        
        # Get candidate text data
        candidates_text_collection = db_connector.db["candidates_text"]
        
        # Get job samples
        job_samples = db_connector.sample_candidate_jobs(n=min(100, limit))
        job_ids = [job.get("original_job_id") for job in job_samples if job.get("original_job_id")]
        
        if not job_ids:
            logger.error("No job IDs found in the database")
            raise ValueError("No job IDs found in the database")
            
        logger.info(f"Found {len(job_ids)} job IDs for actions")
        
        # Process each candidate
        for i, candidate_doc in enumerate(candidate_docs):
            try:
                # Get candidate ID
                candidate_id = candidate_doc.get("original_candidate_id")
                if not candidate_id:
                    logger.warning(f"Candidate {i} missing original_candidate_id, skipping")
                    continue
                
                # Get candidate text data
                candidate_text = candidates_text_collection.find_one({"original_candidate_id": candidate_id})
                if not candidate_text:
                    logger.warning(f"No text data found for candidate {candidate_id}, skipping")
                    continue
                
                # Format candidate profile for LLM
                applicant_profile = {
                    "bio": candidate_text.get("bio", "Experienced professional seeking new opportunities."),
                    "resume": f"Skills:\n- {', '.join(candidate_text.get('skills', []))}\n\nExperience:\n"
                    + '\n'.join([f"- {exp.get('title')} at {exp.get('company')}" for exp in candidate_text.get('experience', [])])
                }
                
                # Get state vector for candidate
                state = db_connector.get_applicant_state(candidate_id)
                
                # Sample 3 jobs for each candidate to avoid making too many LLM calls
                sampled_job_ids = random.sample(job_ids, min(3, len(job_ids)))
                
                for job_id in sampled_job_ids:
                    # Get job details
                    job_details = db_connector.get_job_details(job_id)
                    if not job_details:
                        logger.warning(f"No details found for job {job_id}, skipping")
                        continue
                    
                    job_title = job_details.get("job_title", "Unknown Position")
                    job_description = job_details.get("description", "No description available.")
                    
                    # Get job vector
                    job_vectors = db_connector.get_job_vectors([job_id])
                    if not job_vectors:
                        logger.warning(f"No job vector found for job_id {job_id}, skipping")
                        continue
                    
                    action = job_vectors[0]
                    
                    # For simplicity in pretraining, next_state = state (static state)
                    next_state = state
                    
                    # First, get title-based decision
                    title_prompt = build_title_prompt(applicant_profile, job_title)
                    title_decision, title_reason, _ = get_llm_decision(title_prompt, llm_model, tokenizer, device)
                    
                    # If title decision is CLICK, get description-based decision
                    if title_decision == "CLICK":
                        # Make description decision
                        desc_prompt = build_description_prompt(applicant_profile, job_title, job_description)
                        desc_decision, desc_reason, _ = get_llm_decision(desc_prompt, llm_model, tokenizer, device, max_new_tokens=100)
                        
                        # Map decision to reward
                        reward_map = {
                            "APPLY": 1.0,
                            "SAVE": 0.5,
                            "IGNORE": -0.1,
                            "PARSE_ERROR": 0.0  # Fallback
                        }
                        reward = reward_map.get(desc_decision, 0.0)
                    else:
                        # If not CLICK, assign a small negative reward
                        reward = -0.1
                    
                    # Add to lists
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    next_states.append(next_state)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(candidate_docs)} candidates with LLM simulation")
                    
            except Exception as e:
                logger.error(f"Error processing candidate {i} with LLM: {e}")
                continue
                
        logger.info(f"Created dataset with {len(states)} valid examples using LLM simulation")
        
        # Create dataset
        dataset = JobRecommendationDataset(states, actions, rewards, next_states)
        return dataset
        
    except Exception as e:
        logger.error(f"Error loading data from database with LLM: {e}")
        raise ValueError(f"Failed to load data from database with LLM: {e}")

def load_hybrid_data_from_database(db_connector: DatabaseConnector, llm_model=None, tokenizer=None, 
                                 device="cpu", limit: int = PRETRAINING_CONFIG["dataset_size"],
                                 cosine_weight: float = 0.3) -> JobRecommendationDataset:
    """
    Load data from the database and create a dataset for pretraining using a hybrid of 
    LLM simulation and cosine similarity.
    
    Args:
        db_connector: Database connector for fetching data.
        llm_model: LLM model for simulation.
        tokenizer: Tokenizer for the LLM model.
        device: Device to run the LLM on.
        limit: Maximum number of examples to load.
        cosine_weight: Weight for cosine similarity in the hybrid reward (0-1).
        
    Returns:
        JobRecommendationDataset: Dataset with examples from the database.
    """
    logger.info(f"Loading data from database using hybrid approach, limit: {limit}")
    
    if llm_model is None or tokenizer is None:
        logger.error("LLM model and tokenizer are required for hybrid approach")
        raise ValueError("LLM model and tokenizer are required for hybrid approach")
    
    # Prepare lists for dataset
    states = []
    actions = []
    rewards = []
    next_states = []
    
    try:
        # Get a list of candidate IDs from the database
        candidates_collection = db_connector.db[db_connector.collections["candidates_embeddings"]]
        candidate_docs = list(candidates_collection.find().limit(limit))
        
        if not candidate_docs:
            logger.error("No candidate data found in the database")
            raise ValueError("No candidate data found in the database")
            
        logger.info(f"Found {len(candidate_docs)} candidates in the database")
        
        # Get candidate text data
        candidates_text_collection = db_connector.db["candidates_text"]
        
        # Get job samples
        job_samples = db_connector.sample_candidate_jobs(n=min(100, limit))
        job_ids = [job.get("original_job_id") for job in job_samples if job.get("original_job_id")]
        
        if not job_ids:
            logger.error("No job IDs found in the database")
            raise ValueError("No job IDs found in the database")
            
        logger.info(f"Found {len(job_ids)} job IDs for actions")
        
        # Process each candidate
        for i, candidate_doc in enumerate(candidate_docs):
            try:
                # Get candidate ID
                candidate_id = candidate_doc.get("original_candidate_id")
                if not candidate_id:
                    logger.warning(f"Candidate {i} missing original_candidate_id, skipping")
                    continue
                
                # Get candidate text data
                candidate_text = candidates_text_collection.find_one({"original_candidate_id": candidate_id})
                if not candidate_text:
                    logger.warning(f"No text data found for candidate {candidate_id}, skipping")
                    continue
                
                # Format candidate profile for LLM
                applicant_profile = {
                    "bio": candidate_text.get("bio", "Experienced professional seeking new opportunities."),
                    "resume": f"Skills:\n- {', '.join(candidate_text.get('skills', []))}\n\nExperience:\n"
                    + '\n'.join([f"- {exp.get('title')} at {exp.get('company')}" for exp in candidate_text.get('experience', [])])
                }
                
                # Get state vector for candidate
                state = db_connector.get_applicant_state(candidate_id)
                
                # Sample 3 jobs for each candidate to avoid making too many LLM calls
                sampled_job_ids = random.sample(job_ids, min(3, len(job_ids)))
                
                for job_id in sampled_job_ids:
                    # Get job details
                    job_details = db_connector.get_job_details(job_id)
                    if not job_details:
                        logger.warning(f"No details found for job {job_id}, skipping")
                        continue
                    
                    job_title = job_details.get("job_title", "Unknown Position")
                    job_description = job_details.get("description", "No description available.")
                    
                    # Get job vector
                    job_vectors = db_connector.get_job_vectors([job_id])
                    if not job_vectors:
                        logger.warning(f"No job vector found for job_id {job_id}, skipping")
                        continue
                    
                    action = job_vectors[0]
                    
                    # For simplicity in pretraining, next_state = state (static state)
                    next_state = state
                    
                    # Calculate cosine similarity reward
                    cosine_reward = float(torch.cosine_similarity(
                        state.unsqueeze(0), 
                        action.unsqueeze(0), 
                        dim=1
                    ).item())
                    
                    # Scale cosine reward to [0, 1] range
                    cosine_reward = (cosine_reward + 1) / 2
                    
                    # Get LLM-based reward
                    # First, get title-based decision
                    title_prompt = build_title_prompt(applicant_profile, job_title)
                    title_decision, title_reason, _ = get_llm_decision(title_prompt, llm_model, tokenizer, device)
                    
                    # If title decision is CLICK, get description-based decision
                    if title_decision == "CLICK":
                        # Make description decision
                        desc_prompt = build_description_prompt(applicant_profile, job_title, job_description)
                        desc_decision, desc_reason, _ = get_llm_decision(desc_prompt, llm_model, tokenizer, device, max_new_tokens=100)
                        
                        # Map decision to reward
                        reward_map = {
                            "APPLY": 1.0,
                            "SAVE": 0.5,
                            "IGNORE": -0.1,
                            "PARSE_ERROR": 0.0  # Fallback
                        }
                        llm_reward = reward_map.get(desc_decision, 0.0)
                    else:
                        # If not CLICK, assign a small negative reward
                        llm_reward = -0.1
                    
                    # Combine rewards using the specified weight
                    hybrid_reward = (cosine_weight * cosine_reward) + ((1 - cosine_weight) * llm_reward)
                    
                    # Add to lists
                    states.append(state)
                    actions.append(action)
                    rewards.append(hybrid_reward)
                    next_states.append(next_state)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(candidate_docs)} candidates with hybrid approach")
                    
            except Exception as e:
                logger.error(f"Error processing candidate {i} with hybrid approach: {e}")
                continue
                
        logger.info(f"Created dataset with {len(states)} valid examples using hybrid approach")
        
        # Create dataset
        dataset = JobRecommendationDataset(states, actions, rewards, next_states)
        return dataset
        
    except Exception as e:
        logger.error(f"Error loading data from database with hybrid approach: {e}")
        raise ValueError(f"Failed to load data from database with hybrid approach: {e}")

def build_title_prompt(applicant_profile, job_title):
    """
    Build prompt for title-based decision using LLM.
    
    Args:
        applicant_profile: Dictionary containing candidate profile information.
        job_title: Job title string.
        
    Returns:
        String: Formatted prompt for the LLM.
    """
    prompt = f"""### ROLE ###
    You are role-playing as a job applicant. Your persona is defined by the profile below.

    ### APPLICANT PROFILE ###
    {applicant_profile['bio']}

    ---
    {applicant_profile['resume']}

    ### SIMULATION INSTRUCTION ###
    You are evaluating job postings. You will be shown a Job Title and must decide whether to 'CLICK' to view the full job description or 'IGNORE' the job based *only* on how well the title matches your profile and interests.

    Your response MUST follow the format below. Provide a brief reason for your decision.

    ### JOB TITLE ###
    {job_title}

    ### YOUR DECISION ###
    DECISION: [CLICK or IGNORE]
    REASON: [Brief explanation based on the title and your profile]
    """
    return prompt

def build_description_prompt(applicant_profile, job_title, job_description):
    """
    Build prompt for description-based decision using LLM.
    
    Args:
        applicant_profile: Dictionary containing candidate profile information.
        job_title: Job title string.
        job_description: Job description string.
        
    Returns:
        String: Formatted prompt for the LLM.
    """
    prompt = f"""### ROLE ###
    You are role-playing as a job applicant. Your persona is defined by the profile below.

    ### APPLICANT PROFILE ###
    {applicant_profile['bio']}

    ---
    {applicant_profile['resume']}

    ### SIMULATION INSTRUCTION ###
    You previously clicked on a job posting based on its title. Now you have the full job description. Based on your profile, the job title, and the full description, you must decide whether to 'APPLY' for this job, 'SAVE' it for later consideration, or 'IGNORE' it because it's not a good fit after all.

    Your response MUST follow the format below. Provide a brief reason for your decision.

    ### JOB TITLE ###
    {job_title}

    ### JOB DESCRIPTION ###
    {job_description}

    ### YOUR DECISION ###
    DECISION: [APPLY or SAVE or IGNORE]
    REASON: [Brief explanation based on the job description and your profile]
    """
    return prompt

def get_llm_decision(prompt, model, tokenizer, device, max_new_tokens=50):
    """
    Get decision from LLM based on the prompt.
    
    Args:
        prompt: Formatted prompt for the LLM.
        model: LLM model.
        tokenizer: Tokenizer for the LLM.
        device: Device to run inference on.
        max_new_tokens: Maximum number of tokens to generate.
        
    Returns:
        Tuple: (decision, reason, raw_response)
    """
    messages = [{"role": "user", "content": prompt}]
    encoded_input = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)

    # Generate a short response as we only need the decision and reason
    generated_ids = model.generate(
        encoded_input,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Use greedy decoding for more predictable format
        pad_token_id=tokenizer.eos_token_id  # Avoid potential issues with padding in short generations
    )
    decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Find the section after your prompt where the model's response starts
    try:
        response_start_index = decoded_output.find("[/INST]") + len("[/INST]")
        llm_response_text = decoded_output[response_start_index:].strip()
    except:
        llm_response_text = decoded_output  # Fallback

    # Parse the output
    decision = "PARSE_ERROR"
    reason = "N/A"
    decision_line = None

    # Split response into lines to find the DECISION line
    lines = llm_response_text.split('\n')
    for line in lines:
        if line.startswith("DECISION:"):
            decision_line = line.replace("DECISION:", "").strip()
            # Simple parsing assuming it's the first word after DECISION:
            decision = decision_line.split()[0].upper() if decision_line else "PARSE_ERROR"
            break  # Assume the first DECISION line is the one we want

    # Find the REASON line
    reason_line_start = llm_response_text.find("REASON:")
    if reason_line_start != -1:
        # Extract everything after REASON: and potentially trim
        reason = llm_response_text[reason_line_start + len("REASON:"):].strip()

    # Validate decision
    valid_decisions_step1 = ["CLICK", "IGNORE"]
    valid_decisions_step2 = ["APPLY", "SAVE", "IGNORE"]

    # Determine which step's decisions to validate against
    if "JOB DESCRIPTION" in prompt:
        valid_decisions = valid_decisions_step2
    else:
        valid_decisions = valid_decisions_step1

    if decision not in valid_decisions:
        logger.warning(f"LLM returned unexpected decision format: {decision}. Raw output:\n{llm_response_text}\n")
        decision = "PARSE_ERROR"  # Mark as error if not in expected set

    return decision, reason, llm_response_text

def pretrain_q_network(q_network: QNetwork, 
                        train_loader: DataLoader, 
                        val_loader: Optional[DataLoader] = None,
                        device: torch.device = TRAINING_CONFIG["device"],
                        num_epochs: int = PRETRAINING_CONFIG["num_epochs"],
                        lr: float = PRETRAINING_CONFIG["lr"],
                        weight_decay: float = PRETRAINING_CONFIG["weight_decay"]) -> Tuple[List[float], List[float]]:
    """
    Pretrain the Q-network using supervised learning.
    
    Args:
        q_network: Q-network model.
        train_loader: DataLoader for training data.
        val_loader: Optional DataLoader for validation data.
        device: Device to train on.
        num_epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: Weight decay for regularization.
        
    Returns:
        Tuple of (training_losses, validation_losses).
    """
    logger.info(f"Pretraining Q-network for {num_epochs} epochs")
    
    # Move model to device
    q_network = q_network.to(device)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(q_network.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Loss function (MSE for Q-value prediction)
    criterion = torch.nn.MSELoss()
    
    # Track losses
    training_losses = []
    validation_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        q_network.train()
        epoch_loss = 0.0
        
        for batch_idx, (states, actions, rewards, _) in enumerate(train_loader):
            # Move batch to device
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            
            # Forward pass - model predicts the reward value for this state-action pair
            q_values = q_network(states, actions)
            
            # Compute loss against precomputed "ground truth" rewards
            # The model is learning to predict the expected reward for recommending
            # a particular job to a particular applicant
            loss = criterion(q_values, rewards)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update epoch loss
            epoch_loss += loss.item()
            
            # Log batch progress
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        # Compute average training loss for epoch
        avg_train_loss = epoch_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        
        # Validation - testing the model's ability to predict rewards for unseen applicant-job pairs
        # This measures generalization and prevents overfitting to the training data
        if val_loader:
            # CONCEPTUAL NOTE: During validation, we use a separate subset of data
            # to evaluate how well the model predicts rewards for state-action pairs
            # it hasn't seen during training. This helps us measure generalization
            # and prevent overfitting.
            q_network.eval()  # Set model to evaluation mode (disables dropout, etc.)
            val_loss = 0.0
            
            with torch.no_grad():  # No need to track gradients during validation
                for states, actions, rewards, _ in val_loader:
                    # Move batch to device
                    states = states.to(device)
                    actions = actions.to(device)
                    rewards = rewards.to(device)
                    
                    # Forward pass - predict rewards for state-action pairs in validation set
                    q_values = q_network(states, actions)
                    
                    # Compute loss against precomputed rewards
                    # Testing if model can predict the expected reward for job recommendations
                    # it hasn't seen during training
                    loss = criterion(q_values, rewards)
                    val_loss += loss.item()
            
            # Compute average validation loss
            avg_val_loss = val_loss / len(val_loader)
            validation_losses.append(avg_val_loss)
            
            # Log epoch results
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        else:
            # Log epoch results (training only)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}")
    
    logger.info("Q-network pretraining complete")
    return training_losses, validation_losses


def pretrain_world_model(world_model: WorldModel, 
                           train_loader: DataLoader, 
                           val_loader: Optional[DataLoader] = None,
                           device: torch.device = TRAINING_CONFIG["device"],
                           num_epochs: int = PRETRAINING_CONFIG["num_epochs"],
                           lr: float = PRETRAINING_CONFIG["lr"],
                           weight_decay: float = PRETRAINING_CONFIG["weight_decay"]) -> Tuple[List[float], List[float]]:
    """
    Pretrain the world model using supervised learning.
    
    Args:
        world_model: World model.
        train_loader: DataLoader for training data.
        val_loader: Optional DataLoader for validation data.
        device: Device to train on.
        num_epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: Weight decay for regularization.
        
    Returns:
        Tuple of (training_losses, validation_losses).
    """
    logger.info(f"Pretraining world model for {num_epochs} epochs")
    
    # Move model to device
    world_model = world_model.to(device)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(world_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Loss function (MSE for reward prediction and next state prediction)
    reward_criterion = torch.nn.MSELoss()
    next_state_criterion = torch.nn.MSELoss()
    
    # Track losses
    training_losses = []
    validation_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        world_model.train()
        epoch_loss = 0.0
        
        for batch_idx, (states, actions, rewards, next_states) in enumerate(train_loader):
            # Move batch to device
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            
            # Combine state and action for input
            inputs = torch.cat([states, actions], dim=1)
            
            # Forward pass
            pred_rewards, pred_next_states = world_model(inputs)
            
            # Compute loss components
            reward_loss = reward_criterion(pred_rewards, rewards)
            next_state_loss = next_state_criterion(pred_next_states, next_states)
            
            # Combined loss
            # Note: we weight the components based on their importance
            loss = reward_loss + 0.1 * next_state_loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update epoch loss
            epoch_loss += loss.item()
            
            # Log batch progress
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                            f"Loss: {loss.item():.6f} (R: {reward_loss.item():.6f}, S: {next_state_loss.item():.6f})")
        
        # Compute average training loss for epoch
        avg_train_loss = epoch_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        
        # Validation
        if val_loader:
            world_model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for states, actions, rewards, next_states in val_loader:
                    # Move batch to device
                    states = states.to(device)
                    actions = actions.to(device)
                    rewards = rewards.to(device)
                    next_states = next_states.to(device)
                    
                    # Combine state and action for input
                    inputs = torch.cat([states, actions], dim=1)
                    
                    # Forward pass
                    pred_rewards, pred_next_states = world_model(inputs)
                    
                    # Compute loss components
                    reward_loss = reward_criterion(pred_rewards, rewards)
                    next_state_loss = next_state_criterion(pred_next_states, next_states)
                    
                    # Combined loss
                    loss = reward_loss + 0.1 * next_state_loss
                    val_loss += loss.item()
            
            # Compute average validation loss
            avg_val_loss = val_loss / len(val_loader)
            validation_losses.append(avg_val_loss)
            
            # Log epoch results
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        else:
            # Log epoch results (training only)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}")
    
    logger.info("World model pretraining complete")
    return training_losses, validation_losses


def save_pretrained_models(q_network: QNetwork, world_model: WorldModel, 
                           output_dir: str, model_metrics: Dict[str, List[float]]) -> None:
    """
    Save pretrained models and training metrics.
    
    Args:
        q_network: Pretrained Q-network model.
        world_model: Pretrained world model.
        output_dir: Directory to save models and metrics.
        model_metrics: Dictionary of training and validation metrics.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Q-network
    q_network_path = os.path.join(output_dir, "q_network_pretrained.pt")
    torch.save(q_network.state_dict(), q_network_path)
    logger.info(f"Saved pretrained Q-network to {q_network_path}")
    
    # Save world model
    world_model_path = os.path.join(output_dir, "world_model_pretrained.pt")
    torch.save(world_model.state_dict(), world_model_path)
    logger.info(f"Saved pretrained world model to {world_model_path}")
    
    # Save model architecture configurations
    q_network_config = {
        "state_dim": q_network.state_dim,
        "action_dim": q_network.action_dim,
        "hidden_dims": q_network.hidden_dims,
        "dropout_rate": q_network.dropout_rate
    }
    
    world_model_config = {
        "input_dim": world_model.input_dim,
        "state_dim": world_model.state_dim,
        "reward_dim": world_model.reward_dim,
        "hidden_dims": world_model.hidden_dims,
        "dropout_rate": world_model.dropout_rate
    }
    
    # Save combined config
    config = {
        "q_network": q_network_config,
        "world_model": world_model_config,
        "metrics": model_metrics
    }
    
    config_path = os.path.join(output_dir, "pretrained_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Saved model configurations to {config_path}")
    
    # Visualize training metrics
    visualizer = Visualizer()
    
    # Plot Q-network losses
    if "q_train_losses" in model_metrics and len(model_metrics["q_train_losses"]) > 0:
        q_loss_plot_path = os.path.join(output_dir, "q_network_training.png")
        visualizer.plot_training_curves(
            train_values=model_metrics["q_train_losses"],
            val_values=model_metrics.get("q_val_losses"),
            title="Q-Network Training Loss",
            xlabel="Epoch",
            ylabel="Loss",
            save_path=q_loss_plot_path
        )
        logger.info(f"Saved Q-network training plot to {q_loss_plot_path}")
    
    # Plot world model losses
    if "world_train_losses" in model_metrics and len(model_metrics["world_train_losses"]) > 0:
        world_loss_plot_path = os.path.join(output_dir, "world_model_training.png")
        visualizer.plot_training_curves(
            train_values=model_metrics["world_train_losses"],
            val_values=model_metrics.get("world_val_losses"),
            title="World Model Training Loss",
            xlabel="Epoch",
            ylabel="Loss",
            save_path=world_loss_plot_path
        )
        logger.info(f"Saved world model training plot to {world_loss_plot_path}")


def main(args: argparse.Namespace) -> None:
    """
    Main function for pretraining Dyna-Q models.
    
    Args:
        args: Command-line arguments.
    """
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if args.device else TRAINING_CONFIG["device"])
    logger.info(f"Using device: {device}")
    
    # Initialize database connector
    db_connector = DatabaseConnector()
    
    # Initialize TensorCache
    tensor_cache = TensorCache(device=device)
    logger.info(f"Initializing TensorCache on {device}")
    
    # Get applicant IDs
    try:
        collection = db_connector.db[db_connector.collections["candidates_text"]]
        applicant_ids = [doc["_id"] for doc in collection.find({}, {"_id": 1}).limit(args.num_applicants)]
        if not applicant_ids:
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
    
    # Create output directory
    output_dir = args.save_path or os.path.join(PATH_CONFIG["model_dir"], "pretrained", args.strategy)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data based on strategy
    try:
        if args.strategy == "cosine":
            logger.info("Using cosine similarity for pretraining (baseline)")
            dataset = load_cosine_data_using_tensor_cache(
                tensor_cache=tensor_cache,
                target_applicant_id=target_applicant_id,
                limit=args.dataset_limit,
                device=device
            )
        elif args.strategy == "llm":
            logger.info("Using LLM simulation for pretraining")
            # Import and load LLM model if needed for LLM or hybrid methods
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                from huggingface_hub import login
                
                logger.info("Loading LLM model for simulation...")
                model_id = STRATEGY_CONFIG["llm"]["model_id"]
                
                # Login to Hugging Face
                try:
                    token_path = HF_CONFIG["token_path"]
                    
                    # Ensure an absolute path by resolving relative to project root
                    if not os.path.isabs(token_path):
                        token_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), token_path))
                    
                    with open(token_path, "r") as f:
                        token = f.read().strip()
                    login(token=token)
                    logger.info("Successfully logged in to Hugging Face for model loading")
                except Exception as e:
                    logger.error(f"Failed to login to Hugging Face: {e}")
                    raise
                
                # Configure 4-bit quantization
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=STRATEGY_CONFIG["llm"]["quantization"]["load_in_4bit"],
                    bnb_4bit_quant_type=STRATEGY_CONFIG["llm"]["quantization"]["quant_type"],
                    bnb_4bit_use_nested_quant=STRATEGY_CONFIG["llm"]["quantization"]["use_nested_quant"],
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                
                # Load model with quantization settings
                llm_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto"
                )
                
                logger.info("LLM model loaded successfully")
                
                # For now, we'll use a simplified version with TensorCache:
                # First, create the dataset with cosine rewards as a fallback
                dataset = load_cosine_data_using_tensor_cache(
                    tensor_cache=tensor_cache,
                    target_applicant_id=target_applicant_id,
                    limit=args.dataset_limit,
                    device=device
                )
                
                # TODO: Implement proper LLM-based reward generation with TensorCache
                # This is a temporary approach until we update the full LLM integration
                logger.warning("Using cosine rewards as fallback for LLM strategy - proper LLM integration with TensorCache pending")
                
            except ImportError as e:
                logger.error(f"Failed to load transformers library: {e}")
                logger.error("Please install with: pip install transformers accelerate bitsandbytes")
                raise
            except Exception as e:
                logger.error(f"Failed to load LLM model: {e}")
                raise
        elif args.strategy == "hybrid":
            logger.info(f"Using hybrid approach for pretraining (cosine weight: {args.cosine_weight})")
            # Import and load LLM model
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                
                logger.info("Loading LLM model for simulation...")
                model_id = STRATEGY_CONFIG["llm"]["model_id"]
                
                # Configure 4-bit quantization
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=STRATEGY_CONFIG["llm"]["quantization"]["load_in_4bit"],
                    bnb_4bit_quant_type=STRATEGY_CONFIG["llm"]["quantization"]["quant_type"],
                    bnb_4bit_use_nested_quant=STRATEGY_CONFIG["llm"]["quantization"]["use_nested_quant"],
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                
                # Load model with quantization settings
                llm_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto"
                )
                
                logger.info("LLM model loaded successfully")
                
                # For now, we'll use a simplified version with TensorCache:
                # Use the cosine data as a starting point
                dataset = load_cosine_data_using_tensor_cache(
                    tensor_cache=tensor_cache,
                    target_applicant_id=target_applicant_id,
                    limit=args.dataset_limit,
                    device=device
                )
                
                # TODO: Implement proper hybrid reward generation with TensorCache
                # This is a temporary approach until we update the full hybrid integration
                logger.warning("Using cosine rewards as fallback for hybrid strategy - proper hybrid integration with TensorCache pending")
                
            except ImportError as e:
                logger.error(f"Failed to load transformers library: {e}")
                logger.error("Please install with: pip install transformers accelerate bitsandbytes")
                raise
            except Exception as e:
                logger.error(f"Failed to load LLM model: {e}")
                raise
        else:
            logger.error(f"Invalid pretraining strategy: {args.strategy}")
            raise ValueError(f"Invalid pretraining strategy: {args.strategy}")
        
        logger.info(f"Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    
    # Create data loaders
    train_loader, val_loader = create_pretraining_data_loader(
        dataset,
        batch_size=args.batch_size or PRETRAINING_CONFIG["batch_size"],
        validation_split=PRETRAINING_CONFIG["validation_split"]
    )
    
    # Initialize metrics dictionary
    model_metrics = {
        "pretraining_strategy": args.strategy
    }
    
    # Initialize and pretrain Q-network if requested
    if args.pretrain_q_network:
        # Initialize Q-network
        q_network = QNetwork(
            state_dim=MODEL_CONFIG["q_network"]["state_dim"],
            action_dim=MODEL_CONFIG["q_network"]["action_dim"],
            hidden_dims=MODEL_CONFIG["q_network"]["hidden_dims"],
            dropout_rate=MODEL_CONFIG["q_network"]["dropout_rate"],
            activation=MODEL_CONFIG["q_network"]["activation"]
        ).to(device)
        
        # Pretrain Q-network
        q_train_losses, q_val_losses = pretrain_q_network(
            q_network=q_network,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.num_epochs or PRETRAINING_CONFIG["num_epochs"],
            lr=args.learning_rate or PRETRAINING_CONFIG["lr"]
        )
        
        # Add metrics to dictionary
        model_metrics["q_train_losses"] = q_train_losses
        model_metrics["q_val_losses"] = q_val_losses
    else:
        # Initialize an untrained Q-network for saving
        q_network = QNetwork(
            state_dim=MODEL_CONFIG["q_network"]["state_dim"],
            action_dim=MODEL_CONFIG["q_network"]["action_dim"],
            hidden_dims=MODEL_CONFIG["q_network"]["hidden_dims"],
            dropout_rate=MODEL_CONFIG["q_network"]["dropout_rate"],
            activation=MODEL_CONFIG["q_network"]["activation"]
        ).to(device)
    
    # Initialize and pretrain world model if requested
    if args.pretrain_world_model:
        # Initialize world model
        world_model = WorldModel(
            input_dim=MODEL_CONFIG["world_model"]["input_dim"],
            state_dim=MODEL_CONFIG["q_network"]["state_dim"],
            hidden_dims=MODEL_CONFIG["world_model"]["hidden_dims"],
            dropout_rate=MODEL_CONFIG["world_model"]["dropout_rate"],
            activation=MODEL_CONFIG["world_model"]["activation"]
        ).to(device)
        
        # Pretrain world model
        world_train_losses, world_val_losses = pretrain_world_model(
            world_model=world_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.num_epochs or PRETRAINING_CONFIG["num_epochs"],
            lr=args.learning_rate or PRETRAINING_CONFIG["lr"]
        )
        
        # Add metrics to dictionary
        model_metrics["world_train_losses"] = world_train_losses
        model_metrics["world_val_losses"] = world_val_losses
    else:
        # Initialize an untrained world model for saving
        world_model = WorldModel(
            input_dim=MODEL_CONFIG["world_model"]["input_dim"],
            state_dim=MODEL_CONFIG["q_network"]["state_dim"],
            hidden_dims=MODEL_CONFIG["world_model"]["hidden_dims"],
            dropout_rate=MODEL_CONFIG["world_model"]["dropout_rate"],
            activation=MODEL_CONFIG["world_model"]["activation"]
        ).to(device)
    
    # Save pretrained models
    save_pretrained_models(q_network, world_model, output_dir, model_metrics)
    
    # Clean up resources
    if tensor_cache is not None and hasattr(tensor_cache, 'clear'):
        tensor_cache.clear()
        logger.info("Tensor cache cleared")
        
    # Free PyTorch memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache emptied")
    
    # Close database connection
    db_connector.close()
    
    logger.info("Pretraining complete")
    

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Pretrain Dyna-Q job recommender models")
    
    # Data generation options
    parser.add_argument("--strategy", type=str, choices=["cosine", "llm", "hybrid"], default="cosine",
                      help="Strategy for pretraining: cosine (baseline), llm, or hybrid")
    parser.add_argument("--cosine_weight", type=float, default=STRATEGY_CONFIG["hybrid"]["initial_cosine_weight"],
                      help="Weight for cosine similarity in hybrid method (0-1)")
    parser.add_argument("--dataset_limit", type=int, default=PRETRAINING_CONFIG["dataset_size"],
                      help="Maximum number of examples to load from database")
    
    # Pretraining options
    parser.add_argument("--pretrain_q_network", action="store_true", help="Pretrain Q-network")
    parser.add_argument("--pretrain_world_model", action="store_true", help="Pretrain world model")
    
    # Model parameters
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate for pretraining")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for pretraining")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs for pretraining")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=ENV_CONFIG["random_seed"], help="Random seed")
    parser.add_argument("--device", type=str, default=TRAINING_CONFIG["device"], help="Device to use for training")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save pretrained models")
    parser.add_argument("--num_applicants", type=int, default=TRAINING_CONFIG["num_candidates"],
                      help="Number of applicants to use for pretraining")
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no pretraining option specified, enable all
    if not (args.pretrain_q_network or args.pretrain_world_model):
        args.pretrain_q_network = True
        args.pretrain_world_model = True
    
    # Run main function
    main(args) 