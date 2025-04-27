"""
Pretraining script for the Dyna-Q job recommender model.

This script handles pretraining of the Q-network and world model using
different methods to mitigate the cold-start problem:
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
from config.config import MODEL_CONFIG, PRETRAINING_CONFIG, PATH_CONFIG, TRAINING_CONFIG

# Import modules
from data.database import DatabaseConnector
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
                
                # Assign a reward based on cosine similarity
                reward = float(torch.cosine_similarity(
                    state.unsqueeze(0), 
                    action.unsqueeze(0), 
                    dim=1
                ).item())
                
                # Scale reward to [0, 1] range
                reward = (reward + 1) / 2
                
                # Add to lists
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
        
        # Create dataset
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
            
            # Forward pass
            q_values = q_network(states, actions)
            
            # Compute loss
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
        
        # Validation
        if val_loader:
            q_network.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for states, actions, rewards, _ in val_loader:
                    # Move batch to device
                    states = states.to(device)
                    actions = actions.to(device)
                    rewards = rewards.to(device)
                    
                    # Forward pass
                    q_values = q_network(states, actions)
                    
                    # Compute loss
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize database connector
    db_connector = DatabaseConnector()
    
    # Create output directory
    output_dir = args.save_path or os.path.join(PATH_CONFIG["model_dir"], "pretrained", args.method)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data from database based on method
    try:
        if args.method == "cosine":
            logger.info("Using cosine similarity for pretraining (baseline)")
            dataset = load_cosine_data_from_database(db_connector, limit=args.dataset_limit)
        elif args.method == "llm":
            logger.info("Using LLM simulation for pretraining")
            # Import and load LLM model if needed for LLM or hybrid methods
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                
                logger.info("Loading LLM model for simulation...")
                model_id = "mistralai/Mistral-7B-Instruct-v0.2"
                
                # Configure 4-bit quantization
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_nested_quant=True,
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
                
                # Load data using LLM simulation
                dataset = load_llm_data_from_database(
                    db_connector, 
                    llm_model=llm_model, 
                    tokenizer=tokenizer, 
                    device=device,
                    limit=args.dataset_limit
                )
                
            except ImportError as e:
                logger.error(f"Failed to load transformers library: {e}")
                logger.error("Please install with: pip install transformers accelerate bitsandbytes")
                raise
            except Exception as e:
                logger.error(f"Failed to load LLM model: {e}")
                raise
        elif args.method == "hybrid":
            logger.info(f"Using hybrid approach for pretraining (cosine weight: {args.cosine_weight})")
            # Import and load LLM model
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                
                logger.info("Loading LLM model for simulation...")
                model_id = "mistralai/Mistral-7B-Instruct-v0.2"
                
                # Configure 4-bit quantization
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_nested_quant=True,
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
                
                # Load data using hybrid approach
                dataset = load_hybrid_data_from_database(
                    db_connector, 
                    llm_model=llm_model, 
                    tokenizer=tokenizer, 
                    device=device,
                    limit=args.dataset_limit,
                    cosine_weight=args.cosine_weight
                )
                
            except ImportError as e:
                logger.error(f"Failed to load transformers library: {e}")
                logger.error("Please install with: pip install transformers accelerate bitsandbytes")
                raise
            except Exception as e:
                logger.error(f"Failed to load LLM model: {e}")
                raise
        else:
            logger.error(f"Invalid pretraining method: {args.method}")
            raise ValueError(f"Invalid pretraining method: {args.method}")
        
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
        "pretraining_method": args.method
    }
    
    # Initialize and pretrain Q-network if requested
    if args.pretrain_q_network:
        # Initialize Q-network
        q_network = QNetwork(
            state_dim=MODEL_CONFIG["q_network"]["state_dim"],
            action_dim=MODEL_CONFIG["q_network"]["action_dim"],
            hidden_dims=MODEL_CONFIG["q_network"]["hidden_dims"],
            dropout_rate=MODEL_CONFIG["q_network"]["dropout_rate"]
        )
        
        # Pretrain Q-network
        q_train_losses, q_val_losses = pretrain_q_network(
            q_network=q_network,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.epochs or PRETRAINING_CONFIG["num_epochs"],
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
            dropout_rate=MODEL_CONFIG["q_network"]["dropout_rate"]
        )
    
    # Initialize and pretrain world model if requested
    if args.pretrain_world_model:
        # Initialize world model
        world_model = WorldModel(
            input_dim=MODEL_CONFIG["world_model"]["input_dim"],
            state_dim=MODEL_CONFIG["q_network"]["state_dim"],
            hidden_dims=MODEL_CONFIG["world_model"]["hidden_dims"],
            dropout_rate=MODEL_CONFIG["world_model"]["dropout_rate"]
        )
        
        # Pretrain world model
        world_train_losses, world_val_losses = pretrain_world_model(
            world_model=world_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.epochs or PRETRAINING_CONFIG["num_epochs"],
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
            dropout_rate=MODEL_CONFIG["world_model"]["dropout_rate"]
        )
    
    # Save pretrained models
    save_pretrained_models(q_network, world_model, output_dir, model_metrics)
    

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Pretrain Dyna-Q job recommender models")
    
    # Data generation options
    parser.add_argument("--method", type=str, choices=["cosine", "llm", "hybrid"], default="cosine",
                      help="Method for pretraining: cosine (baseline), llm, or hybrid")
    parser.add_argument("--cosine_weight", type=float, default=0.3,
                      help="Weight for cosine similarity in hybrid method (0-1)")
    parser.add_argument("--dataset_limit", type=int, default=PRETRAINING_CONFIG["dataset_size"],
                      help="Maximum number of examples to load from database")
    
    # Pretraining options
    parser.add_argument("--pretrain_q_network", action="store_true", help="Pretrain Q-network")
    parser.add_argument("--pretrain_world_model", action="store_true", help="Pretrain world model")
    
    # Model parameters
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate for pretraining")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for pretraining")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs for pretraining")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save pretrained models")
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no pretraining option specified, enable all
    if not (args.pretrain_q_network or args.pretrain_world_model):
        args.pretrain_q_network = True
        args.pretrain_world_model = True
    
    # Run main function
    main(args) 