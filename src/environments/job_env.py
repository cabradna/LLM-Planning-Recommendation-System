"""
Job recommendation environment for Dyna-Q training and evaluation.

This module provides the simulation environment for job recommendation,
including reward calculation and state transitions.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any

# Import configuration and other modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config.config import TRAINING_CONFIG, STRATEGY_CONFIG

# Database connection
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from data.database import DatabaseConnector

logger = logging.getLogger(__name__)

class JobRecommendationEnv:
    """
    Environment for job recommendation.
    
    This environment represents the MDP for job recommendation:
    - States: Applicant profiles (embeddings)
    - Actions: Job postings (embeddings)
    - Rewards: Depends on the reward strategy (cosine similarity, LLM feedback, or hybrid)
    - Transitions: Currently static (state doesn't change)
    """
    
    def __init__(self, db_connector: Optional[DatabaseConnector] = None, 
                 tensor_cache = None,
                 reward_scheme: Optional[Dict[str, float]] = None,
                 reward_strategy: str = "cosine",
                 random_seed: int = 42):
        """
        Initialize the environment.
        
        Args:
            db_connector: Database connector for retrieving data.
            tensor_cache: Optional tensor cache for faster data access.
            reward_scheme: Dictionary mapping response types to reward values.
            reward_strategy: Strategy for generating rewards ("cosine", "llm", or "hybrid").
            random_seed: Random seed for reproducibility.
        """
        # Database connection
        self.db = db_connector or DatabaseConnector()
        
        # Tensor cache for fast data access
        self.tensor_cache = tensor_cache
        self.using_cache = tensor_cache is not None and hasattr(tensor_cache, 'initialized') and tensor_cache.initialized
        
        # Set reward scheme from config if not provided
        self.reward_scheme = reward_scheme or STRATEGY_CONFIG["llm"]["response_mapping"]
        
        # Set reward strategy
        self.reward_strategy = reward_strategy
        
        # Random number generator
        self.rng = np.random.RandomState(random_seed)
        
        # Current state
        self.current_applicant_id = None
        self.current_state = None
        self.candidate_jobs = []
        self.job_vectors = []
        self.job_id_to_idx = {}
        
        logger.info(f"Initialized JobRecommendationEnv with {reward_strategy} reward strategy")
        if self.using_cache:
            logger.info(f"Using tensor cache with {len(self.tensor_cache)} cached jobs")
    
    def use_tensor_cache(self, tensor_cache):
        """
        Configure environment to use tensor cache instead of database queries.
        
        Args:
            tensor_cache: Initialized TensorCache instance
        """
        self.tensor_cache = tensor_cache
        self.using_cache = tensor_cache is not None and hasattr(tensor_cache, 'initialized') and tensor_cache.initialized
        logger.info("Environment configured to use tensor cache")
    
    def reset(self, applicant_id: Optional[str] = None) -> torch.Tensor:
        """
        Reset the environment with a new applicant.
        
        Args:
            applicant_id: Specific applicant ID to use, or None for random selection.
            
        Returns:
            torch.Tensor: Initial state vector.
        """
        # If no applicant_id provided, could randomly select one (implementation depends on your DB)
        # For now, assume applicant_id is always provided
        if applicant_id is None:
            raise ValueError("applicant_id must be provided")
        
        # Set current applicant ID
        self.current_applicant_id = applicant_id
        
        if self.using_cache:
            # Use tensor cache for faster data access
            try:
                # Get state vector from cache
                self.current_state = self.tensor_cache.get_applicant_state(applicant_id)
                
                # Sample candidate jobs from cache
                self.candidate_jobs, self.job_vectors, job_ids = self.tensor_cache.sample_jobs(n=100)
                
                # Create mapping from job ID to index
                self.job_id_to_idx = {job_id: idx for idx, job_id in enumerate(job_ids)}
                
                logger.info(f"Environment reset with cached data for applicant {applicant_id}, {len(self.candidate_jobs)} candidate jobs")
            except KeyError as e:
                # Fallback to database if applicant not in cache
                logger.warning(f"Cache miss: {e}. Falling back to database.")
                self.using_cache = False
                return self._reset_with_db(applicant_id)
        else:
            # Use database query
            return self._reset_with_db(applicant_id)
        
        return self.current_state
    
    def _reset_with_db(self, applicant_id: str) -> torch.Tensor:
        """
        Reset using database queries (original implementation).
        
        Args:
            applicant_id: ID of the applicant
            
        Returns:
            torch.Tensor: Initial state vector
        """
        # Get state vector for this applicant
        self.current_state = self.db.get_applicant_state(applicant_id)
        
        # Sample candidate jobs for this episode
        self.candidate_jobs = self.db.sample_candidate_jobs(n=100, validate_embeddings=True)  # Adjust sample size as needed
        
        # Extract job IDs
        job_ids = [job["_id"] for job in self.candidate_jobs]
        
        # Create mapping from job ID to index
        self.job_id_to_idx = {job_id: idx for idx, job_id in enumerate(job_ids)}
        
        # Get vector representations for these jobs
        self.job_vectors = self.db.get_job_vectors(job_ids)
        
        logger.info(f"Environment reset with DB data for applicant {applicant_id}, {len(self.candidate_jobs)} candidate jobs")
        
        return self.current_state
    
    def step(self, action_idx: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Take a step in the environment by recommending a job.
        
        Args:
            action_idx: Index of the job to recommend from the candidate_jobs list.
            
        Returns:
            Tuple containing:
                - torch.Tensor: Next state (same as current_state in this phase)
                - float: Reward value
                - bool: Done flag (always False in continuous recommendation)
                - Dict: Additional info
                
        Raises:
            RuntimeError: If using a reward strategy other than "cosine" in the base environment
        """
        if action_idx < 0 or action_idx >= len(self.candidate_jobs):
            raise ValueError(f"Invalid action index: {action_idx}. Must be between 0 and {len(self.candidate_jobs)-1}")
        
        # Get the recommended job
        job = self.candidate_jobs[action_idx]
        job_id = job.get("original_job_id", job.get("_id"))
        
        # Get reward based on the selected reward strategy
        if self.reward_strategy == "cosine":
            reward = self.calculate_cosine_reward(action_idx)
        else:
            # Base class only supports cosine similarity rewards
            raise RuntimeError(
                f"The base JobRecommendationEnv only supports 'cosine' reward strategy, "
                f"but '{self.reward_strategy}' was requested. Use appropriate subclass "
                f"(LLMSimulatorEnv or HybridEnv) for non-cosine reward strategies."
            )
        
        # In the current phase, state doesn't change after action
        next_state = self.current_state
        
        # Episode doesn't end
        done = False
        
        # Additional info
        info = {
            "job_id": job_id,
            "reward_strategy": self.reward_strategy,
            "job_title": job.get("job_title", ""),
            "applicant_id": self.current_applicant_id
        }
        
        return next_state, reward, done, info
    
    def calculate_cosine_reward(self, action_idx: int) -> float:
        """
        Calculate reward using cosine similarity between applicant and job embeddings.
        
        Args:
            action_idx: Index of the job action.
            
        Returns:
            float: Cosine similarity reward, scaled to [0,1] if configured.
        """
        if action_idx < 0 or action_idx >= len(self.job_vectors):
            raise ValueError(f"Invalid action index: {action_idx}")
        
        if self.using_cache and hasattr(self.tensor_cache, 'calculate_cosine_similarities'):
            # Fast tensor-based calculation for a single job
            job_vector = self.job_vectors[action_idx]
            
            # Handle dimension mismatch between applicant and job vectors
            if job_vector.shape[0] != self.current_state.shape[0]:
                logger.info(f"Dimension mismatch: job vector {job_vector.shape[0]}, applicant state {self.current_state.shape[0]}")
                
                # When comparing applicant with job, we need to ensure we're comparing analogous components
                # - Applicant vectors: [hard_skills (384), soft_skills (384)] = 768 dimensions
                # - Job vectors: [tech_skills (384), soft_skills (384), job_title (384), experience (384)] = 1536 dimensions
                # 
                # We take the first 768 dimensions of the job vector because those contain:
                # 1. tech_skills - which should be compared with applicant's hard_skills
                # 2. soft_skills - which should be compared with applicant's soft_skills
                #
                # This creates a meaningful semantic comparison between matching components:
                # - Job tech skills ⟷ Applicant hard skills
                # - Job soft skills ⟷ Applicant soft skills
                if job_vector.shape[0] > self.current_state.shape[0]:
                    # Take a slice of the job vector to match applicant dimension
                    job_vector = job_vector[:self.current_state.shape[0]]
                # If job vector has fewer dimensions (unlikely but handle it)
                else:
                    # Take a slice of the applicant state to match job vector dimension
                    applicant_state = self.current_state[:job_vector.shape[0]]
                    # Calculate cosine similarity using tensor operations
                    cos_sim = torch.cosine_similarity(
                        applicant_state.unsqueeze(0),
                        job_vector.unsqueeze(0),
                        dim=1
                    ).item()
                    
                    # Scale reward based on config
                    if STRATEGY_CONFIG["cosine"]["scale_reward"]:
                        return (cos_sim + 1) / 2  # Scale from [-1,1] to [0,1]
                    else:
                        return cos_sim  # Keep in [-1,1] range
            
            # Calculate cosine similarity using tensor operations with correctly sized vectors
            cos_sim = torch.cosine_similarity(
                self.current_state.unsqueeze(0),
                job_vector.unsqueeze(0),
                dim=1
            ).item()
        else:
            # Original implementation
            job_vector = self.job_vectors[action_idx]
            
            # Handle dimension mismatch between applicant and job vectors
            if job_vector.shape[0] != self.current_state.shape[0]:
                logger.info(f"Dimension mismatch: job vector {job_vector.shape[0]}, applicant state {self.current_state.shape[0]}")
                
                # When comparing applicant with job, we need to ensure we're comparing analogous components
                # - Applicant vectors: [hard_skills (384), soft_skills (384)] = 768 dimensions
                # - Job vectors: [tech_skills (384), soft_skills (384), job_title (384), experience (384)] = 1536 dimensions
                # 
                # We take the first 768 dimensions of the job vector because those contain:
                # 1. tech_skills - which should be compared with applicant's hard_skills
                # 2. soft_skills - which should be compared with applicant's soft_skills
                #
                # This creates a meaningful semantic comparison between matching components:
                # - Job tech skills ⟷ Applicant hard skills
                # - Job soft skills ⟷ Applicant soft skills
                if job_vector.shape[0] > self.current_state.shape[0]:
                    # Take a slice of the job vector to match applicant dimension
                    job_vector = job_vector[:self.current_state.shape[0]]
                # If job vector has fewer dimensions (unlikely but handle it)
                else:
                    # Take a slice of the applicant state to match job vector dimension
                    applicant_state = self.current_state[:job_vector.shape[0]]
                    # Calculate cosine similarity using tensor operations
                    cos_sim = torch.cosine_similarity(
                        applicant_state.unsqueeze(0),
                        job_vector.unsqueeze(0),
                        dim=1
                    ).item()
                    
                    # Scale reward based on config
                    if STRATEGY_CONFIG["cosine"]["scale_reward"]:
                        return (cos_sim + 1) / 2  # Scale from [-1,1] to [0,1]
                    else:
                        return cos_sim  # Keep in [-1,1] range
            
            # Calculate cosine similarity with correctly sized vectors
            cos_sim = torch.cosine_similarity(
                self.current_state.unsqueeze(0),
                job_vector.unsqueeze(0),
                dim=1
            ).item()
        
        # Scale reward based on config
        if STRATEGY_CONFIG["cosine"]["scale_reward"]:
            return (cos_sim + 1) / 2  # Scale from [-1,1] to [0,1]
        else:
            return cos_sim  # Keep in [-1,1] range
    
    def calculate_all_cosine_rewards(self) -> torch.Tensor:
        """
        Calculate cosine similarity rewards for all candidate jobs at once.
        
        This is a faster vectorized implementation that can be used for
        exploration and planning.
        
        Returns:
            torch.Tensor: Cosine similarities for all candidate jobs
        """
        if self.using_cache:
            # Extract vectors for current candidate jobs
            job_indices = [self.tensor_cache.job_ids.index(self.job_id_to_idx[idx]) 
                          for idx in range(len(self.candidate_jobs))
                          if self.job_id_to_idx[idx] in self.tensor_cache.job_ids]
            
            # Get job vectors for these indices
            job_vectors_tensor = self.tensor_cache.job_vectors[job_indices]
            
            # Handle dimension mismatch between applicant and job vectors
            if job_vectors_tensor.shape[1] != self.current_state.shape[0]:
                logger.info(f"Dimension mismatch in calculate_all_cosine_rewards: job vector {job_vectors_tensor.shape[1]}, applicant state {self.current_state.shape[0]}")
                
                # Option 1: Project job vectors to match applicant state dimension
                if job_vectors_tensor.shape[1] > self.current_state.shape[0]:
                    job_vectors_tensor = job_vectors_tensor[:, :self.current_state.shape[0]]
                    # Continue with normalized vectors calculation below
                # Option 2: Project applicant vector to match job vectors dimension
                else:
                    applicant_state = self.current_state[:job_vectors_tensor.shape[1]]
                    # Normalize vectors
                    normalized_state = applicant_state / applicant_state.norm()
                    normalized_jobs = job_vectors_tensor / job_vectors_tensor.norm(dim=1, keepdim=True)
                    
                    # Calculate similarities
                    similarities = torch.matmul(normalized_jobs, normalized_state)
                    
                    # Scale if configured
                    if STRATEGY_CONFIG["cosine"]["scale_reward"]:
                        similarities = (similarities + 1) / 2
                        
                    return similarities

                # When comparing applicant vectors with job vectors, we need to ensure semantic alignment:
                # - Applicant vectors: [hard_skills (384), soft_skills (384)] = 768 dimensions
                # - Job vectors: [tech_skills (384), soft_skills (384), job_title (384), experience (384)] = 1536 dimensions
                #
                # We take the first 768 dimensions of job vectors because they contain the matching components:
                # 1. tech_skills - which should be compared with applicant's hard_skills
                # 2. soft_skills - which should be compared with applicant's soft_skills
                #
                # This ensures we're making semantically meaningful comparisons between:
                # - Job tech skills ⟷ Applicant hard skills  
                # - Job soft skills ⟷ Applicant soft skills
                if job_vectors_tensor.shape[1] > self.current_state.shape[0]:
                    # Take a slice of job vectors to match applicant dimension
                    job_vectors_tensor = job_vectors_tensor[:, :self.current_state.shape[0]]
                    # Continue with normalized vectors calculation below
                # If job vectors have fewer dimensions (unlikely but handle it)
                else:
                    applicant_state = self.current_state[:job_vectors_tensor.shape[1]]
                    # Normalize vectors
                    normalized_state = applicant_state / applicant_state.norm()
                    normalized_jobs = job_vectors_tensor / job_vectors_tensor.norm(dim=1, keepdim=True)
                    
                    # Calculate similarities
                    similarities = torch.matmul(normalized_jobs, normalized_state)
                    
                    # Scale if configured
                    if STRATEGY_CONFIG["cosine"]["scale_reward"]:
                        similarities = (similarities + 1) / 2
                        
                    return similarities
            
            # Normalize vectors
            normalized_state = self.current_state / self.current_state.norm()
            normalized_jobs = job_vectors_tensor / job_vectors_tensor.norm(dim=1, keepdim=True)
            
            # Calculate similarities
            similarities = torch.matmul(normalized_jobs, normalized_state)
            
            # Scale if configured
            if STRATEGY_CONFIG["cosine"]["scale_reward"]:
                similarities = (similarities + 1) / 2
                
            return similarities
        else:
            # Calculate for each job individually using the original implementation
            rewards = []
            for idx in range(len(self.job_vectors)):
                reward = self.calculate_cosine_reward(idx)
                rewards.append(reward)
            return torch.tensor(rewards)
    
    def get_valid_actions(self) -> List[int]:
        """
        Get the list of valid action indices.
        
        Returns:
            List[int]: List of valid action indices.
        """
        return list(range(len(self.candidate_jobs)))
    
    def get_action_vector(self, action_idx: int) -> torch.Tensor:
        """
        Get the vector representation of an action.
        
        Args:
            action_idx: Index of the action.
            
        Returns:
            torch.Tensor: Vector representation of the action.
        """
        if action_idx < 0 or action_idx >= len(self.job_vectors):
            raise ValueError(f"Invalid action index: {action_idx}")
        
        return self.job_vectors[action_idx]
    
    def close(self) -> None:
        """
        Clean up resources.
        """
        if hasattr(self, 'db') and self.db is not None:
            self.db.close()
            
    def setup_llm(self, llm_model, tokenizer, device="auto"):
        """
        Set up an LLM for reward calculation.
        
        This method allows setting up or replacing the LLM model during runtime.
        If the environment is not using an LLM-based reward strategy, this
        will automatically switch to a hybrid strategy.
        
        Args:
            llm_model: The LLM model to use for simulation.
            tokenizer: Tokenizer for the LLM model.
            device: Device to run the LLM on.
        """
        # Determine the appropriate environment class based on the reward strategy
        if self.reward_strategy == "cosine":
            # Create a new hybrid environment with the provided LLM
            hybrid_env = HybridEnv(
                db_connector=self.db,
                tensor_cache=self.tensor_cache,
                reward_scheme=self.reward_scheme,
                llm_model=llm_model,
                tokenizer=tokenizer,
                device=device,
                cosine_weight=1.0,  # Start with cosine only
                random_seed=self.rng.randint(0, 2**32-1)
            )
            
            # Transfer state
            hybrid_env.current_applicant_id = self.current_applicant_id
            hybrid_env.current_state = self.current_state
            hybrid_env.candidate_jobs = self.candidate_jobs
            hybrid_env.job_vectors = self.job_vectors
            hybrid_env.job_id_to_idx = self.job_id_to_idx
            
            # Return the new environment
            logger.info("Switching to hybrid environment with LLM")
            return hybrid_env
        
        elif self.reward_strategy in ["llm", "hybrid"]:
            # Update the LLM components directly
            self.llm_model = llm_model
            self.tokenizer = tokenizer
            self.device = device
            logger.info(f"Updated LLM components in {self.__class__.__name__}")
        
        # Return self if we didn't create a new environment
        return self

    def get_job_vectors_tensor(self) -> torch.Tensor:
        """
        Get all current job vectors as a single tensor.
        
        This is useful for vectorized operations in the agent.
        
        Returns:
            torch.Tensor: Tensor of all job vectors [num_jobs, embedding_dim]
        """
        if self.using_cache:
            # Return job vectors directly from tensor cache if available
            return self.job_vectors
        else:
            # Stack individual vectors into a single tensor
            return torch.stack(self.job_vectors)


class LLMSimulatorEnv(JobRecommendationEnv):
    """
    Environment that uses an LLM to simulate user responses.
    
    This extends the basic environment by replacing the simple response
    simulation with LLM-based response generation, implementing the
    "LLM Feedback Only" training strategy from the documentation.
    """
    
    def __init__(self, db_connector: Optional[DatabaseConnector] = None,
                 tensor_cache = None,
                 reward_scheme: Optional[Dict[str, float]] = None,
                 llm_model=None, tokenizer=None, device="cpu",
                 random_seed: int = 42):
        """
        Initialize the LLM-based simulation environment.
        
        Args:
            db_connector: Database connector for retrieving data.
            tensor_cache: Optional tensor cache for faster data access.
            reward_scheme: Dictionary mapping response types to reward values.
            llm_model: The LLM model to use for simulation.
            tokenizer: Tokenizer for the LLM model.
            device: Device to run the LLM on.
            random_seed: Random seed for reproducibility.
        """
        super().__init__(
            db_connector=db_connector,
            tensor_cache=tensor_cache,
            reward_scheme=reward_scheme,
            reward_strategy="llm",
            random_seed=random_seed
        )
        
        # LLM components for reward calculation
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.device = device
        
        # Cache for applicant profiles
        self.applicant_profiles = {}
        
        logger.info("Initialized LLMSimulatorEnv with LLM-based reward strategy")
    
    def reset(self, applicant_id: Optional[str] = None) -> torch.Tensor:
        """
        Reset the environment with a new applicant.
        
        Args:
            applicant_id: Specific applicant ID to use, or None for random selection.
            
        Returns:
            torch.Tensor: Initial state vector.
        """
        state = super().reset(applicant_id)
        
        # Get and cache applicant profile text for LLM prompts if not already cached
        if applicant_id not in self.applicant_profiles:
            try:
                candidates_text_collection = self.db.db.get_collection("candidates_text")
                candidate_text = candidates_text_collection.find_one({"original_candidate_id": applicant_id})
                
                if candidate_text:
                    # Format candidate profile for LLM
                    self.applicant_profiles[applicant_id] = {
                        "bio": candidate_text.get("bio", "Experienced professional seeking new opportunities."),
                        "resume": f"Skills:\n- {', '.join(candidate_text.get('skills', []))}\n\nExperience:\n"
                        + '\n'.join([f"- {exp.get('title')} at {exp.get('company')}" 
                                     for exp in candidate_text.get('experience', [])])
                    }
                else:
                    logger.warning(f"No text data found for candidate {applicant_id}")
                    self.applicant_profiles[applicant_id] = {
                        "bio": "Experienced professional seeking new opportunities.",
                        "resume": "Skills:\n- Programming\n\nExperience:\n- Software Developer"
                    }
            except Exception as e:
                logger.error(f"Error fetching candidate text: {e}")
                self.applicant_profiles[applicant_id] = {
                    "bio": "Experienced professional seeking new opportunities.",
                    "resume": "Skills:\n- Programming\n\nExperience:\n- Software Developer"
                }
        
        return state
    
    def simulate_user_response(self, job: Dict) -> str:
        """
        Use an LLM to simulate user response to a job recommendation.
        
        This method requires a properly initialized LLM model and tokenizer.
        It will throw an error if any required component is missing.
        
        Args:
            job: Job document from the database.
            
        Returns:
            str: Response type ('APPLY', 'SAVE', 'CLICK', 'IGNORE')
            
        Raises:
            RuntimeError: If LLM is not properly initialized or data is missing
        """
        if not self.llm_model or not self.tokenizer:
            raise RuntimeError(
                "LLM model or tokenizer not provided. For an academic project, random simulation "
                "is not acceptable. Either properly initialize the LLM or use cosine similarity instead."
            )
        
        # Get job details
        job_id = job.get("original_job_id", job.get("_id"))
        job_details = self.db.get_job_details(job_id)
        
        if not job_details:
            raise RuntimeError(
                f"No details found for job {job_id}. Cannot generate LLM-based response "
                "without proper job details."
            )
        
        job_title = job_details.get("job_title", "Unknown Position")
        job_description = job_details.get("description", "No description available.")
        
        # Get applicant profile
        applicant_profile = self.applicant_profiles.get(self.current_applicant_id)
        
        if not applicant_profile:
            raise RuntimeError(
                f"No profile found for applicant {self.current_applicant_id}. Cannot generate "
                "LLM-based response without proper applicant profile."
            )
        
        # Build prompt for LLM
        prompt = self._build_llm_prompt(applicant_profile, job_title, job_description)
        
        # Get LLM response
        response = self._get_llm_decision(prompt)
        
        # Map LLM response to predefined response types
        mapped_response = self._map_llm_response(response)
        
        logger.debug(f"LLM simulation: {mapped_response} for job {job_title}")
        
        return mapped_response
    
    def _build_llm_prompt(self, applicant_profile: Dict, job_title: str, job_description: str) -> str:
        """
        Build a prompt for the LLM to simulate user response.
        
        Args:
            applicant_profile: Applicant profile information.
            job_title: Job title.
            job_description: Job description.
            
        Returns:
            str: Prompt for the LLM.
        """
        return f"""
You are simulating a job seeker's response to a job recommendation.

JOB SEEKER PROFILE:
{applicant_profile['bio']}

RESUME:
{applicant_profile['resume']}

JOB RECOMMENDATION:
Title: {job_title}

Description:
{job_description}

Based on the match between the job seeker's profile and the job recommendation,
how would the job seeker respond? Choose ONE of the following options:
- APPLY (The job seeker would submit an application for this position)
- SAVE (The job seeker would save this job for later consideration)
- CLICK (The job seeker would click to view more details)
- IGNORE (The job seeker would ignore this recommendation)

Provide your answer as a single word: APPLY, SAVE, CLICK, or IGNORE.
"""
    
    def _get_llm_decision(self, prompt: str) -> str:
        """
        Get a decision from the LLM.
        
        Args:
            prompt: Prompt for the LLM.
            
        Returns:
            str: LLM response text.
            
        Raises:
            Exception: Any errors during LLM processing are propagated up
        """
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.2,
                do_sample=True,
                top_p=0.95
            )
        
        # Decode the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the part after the prompt
        response_only = response[len(prompt):].strip()
        
        return response_only
    
    def _map_llm_response(self, response: str) -> str:
        """
        Map the LLM response to a predefined response type.
        
        Args:
            response: LLM response text.
            
        Returns:
            str: Mapped response type ('APPLY', 'SAVE', 'CLICK', 'IGNORE').
            
        Raises:
            ValueError: If the LLM response cannot be mapped to a valid response type
        """
        response_lower = response.lower()
        
        if "apply" in response_lower:
            return "APPLY"
        elif "save" in response_lower:
            return "SAVE"
        elif "click" in response_lower:
            return "CLICK"
        elif "ignore" in response_lower:
            return "IGNORE"
        else:
            # Raise error on unclear response
            raise ValueError(f"Unclear LLM response: '{response}'. Cannot map to a valid response type.")


class HybridEnv(LLMSimulatorEnv):
    """
    Environment that implements the hybrid approach described in the documentation.
    
    This environment can switch between cosine similarity and LLM feedback
    reward strategies, supporting the hybrid training approach where the agent
    is first pretrained with cosine similarity rewards and then fine-tuned
    with LLM feedback rewards.
    """
    
    def __init__(self, db_connector: Optional[DatabaseConnector] = None,
                 tensor_cache = None,
                 reward_scheme: Optional[Dict[str, float]] = None,
                 llm_model=None, tokenizer=None, device="cpu",
                 cosine_weight: float = 0.5,
                 random_seed: int = 42):
        """
        Initialize the hybrid environment.
        
        Args:
            db_connector: Database connector for retrieving data.
            tensor_cache: Optional tensor cache for faster data access.
            reward_scheme: Dictionary mapping response types to reward values.
            llm_model: The LLM model to use for simulation.
            tokenizer: Tokenizer for the LLM model.
            device: Device to run the LLM on.
            cosine_weight: Weight for cosine similarity in hybrid approach [0.0, 1.0].
                          0.0 means use only LLM feedback, 1.0 means use only cosine similarity.
            random_seed: Random seed for reproducibility.
        """
        super().__init__(
            db_connector=db_connector,
            tensor_cache=tensor_cache,
            reward_scheme=reward_scheme,
            llm_model=llm_model,
            tokenizer=tokenizer,
            device=device,
            random_seed=random_seed
        )
        
        # Set hybrid parameters
        self.cosine_weight = cosine_weight
        self.reward_strategy = "hybrid"
        
        logger.info(f"Initialized HybridEnv with cosine_weight={cosine_weight}")
    
    def step(self, action_idx: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Take a step in the environment by recommending a job.
        
        In hybrid mode, computes both cosine similarity and LLM feedback rewards
        and combines them according to the cosine_weight parameter.
        
        Args:
            action_idx: Index of the job to recommend from the candidate_jobs list.
            
        Returns:
            Tuple containing:
                - torch.Tensor: Next state (same as current_state in this phase)
                - float: Combined reward value
                - bool: Done flag (always False in continuous recommendation)
                - Dict: Additional info
        """
        if action_idx < 0 or action_idx >= len(self.candidate_jobs):
            raise ValueError(f"Invalid action index: {action_idx}. Must be between 0 and {len(self.candidate_jobs)-1}")
        
        # Get the recommended job
        job = self.candidate_jobs[action_idx]
        job_id = job.get("original_job_id", job.get("_id"))
        
        # Calculate cosine similarity reward
        cosine_reward = self.calculate_cosine_reward(action_idx)
        
        # Get LLM feedback reward
        response = self.simulate_user_response(job)
        llm_reward = self.reward_scheme.get(response, 0.0)
        
        # Combine rewards based on the weight parameter
        combined_reward = (self.cosine_weight * cosine_reward) + ((1 - self.cosine_weight) * llm_reward)
        
        # In the current phase, state doesn't change after action
        next_state = self.current_state
        
        # Episode doesn't end
        done = False
        
        # Additional info
        info = {
            "job_id": job_id,
            "reward_strategy": "hybrid",
            "cosine_reward": cosine_reward,
            "llm_reward": llm_reward,
            "cosine_weight": self.cosine_weight,
            "response": response,
            "job_title": job.get("job_title", ""),
            "applicant_id": self.current_applicant_id
        }
        
        return next_state, combined_reward, done, info
    
    def set_cosine_weight(self, weight: float) -> None:
        """
        Set the weight for cosine similarity in the hybrid approach.
        
        Args:
            weight: Weight for cosine similarity [0.0, 1.0].
                  0.0 means use only LLM feedback, 1.0 means use only cosine similarity.
        """
        if weight < 0.0 or weight > 1.0:
            raise ValueError(f"Cosine weight must be between 0.0 and 1.0, got {weight}")
        
        self.cosine_weight = weight
        logger.info(f"Set cosine_weight to {weight}") 