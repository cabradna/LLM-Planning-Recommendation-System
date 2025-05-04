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
        """
        if action_idx < 0 or action_idx >= len(self.candidate_jobs):
            raise ValueError(f"Invalid action index: {action_idx}. Must be between 0 and {len(self.candidate_jobs)-1}")
        
        # Get the recommended job
        job = self.candidate_jobs[action_idx]
        job_id = job.get("_id")
        
        # Get reward based on the selected reward strategy
        if self.reward_strategy == "cosine":
            reward = self.calculate_cosine_reward(action_idx)
        else:
            # For LLM and hybrid strategies, simulate user response
            response = self.simulate_user_response(job)
            # Calculate reward based on response
            reward = self.reward_scheme.get(response, 0.0)
        
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
            
            # Calculate cosine similarity using tensor operations
            cos_sim = torch.cosine_similarity(
                self.current_state.unsqueeze(0),
                job_vector.unsqueeze(0),
                dim=1
            ).item()
        else:
            # Original implementation
            job_vector = self.job_vectors[action_idx]
            
            # Calculate cosine similarity
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
    
    def simulate_user_response(self, job: Dict) -> str:
        """
        Simulate a user's response to a job recommendation.
        
        In a real system, this would be replaced by actual user feedback.
        This simple simulation is a placeholder for testing/development.
        In production, this could be replaced by an LLM-based simulator
        or connect to a user interface for real feedback.
        
        Args:
            job: Job document from the database.
            
        Returns:
            str: Response type ('APPLY', 'SAVE', 'CLICK', 'IGNORE')
        """
        # Simple random simulation as a placeholder
        # In practice, should implement a more sophisticated simulation
        # based on job-applicant match quality
        response_probs = {
            'APPLY': 0.1,   # 10% chance the user applies
            'SAVE': 0.2,    # 20% chance the user saves
            'CLICK': 0.3,   # 30% chance the user clicks/views
            'IGNORE': 0.4   # 40% chance the user ignores
        }
        
        # Get a random response based on the probabilities
        responses = list(response_probs.keys())
        probs = list(response_probs.values())
        response = self.rng.choice(responses, p=probs)
        
        return response
    
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
        method has no effect.
        
        Args:
            llm_model: LLM model for generating responses
            tokenizer: Tokenizer for the LLM model
            device: Device to run the LLM on
        """
        # This would be implemented when needed for LLM-based reward strategies
        pass

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
    Environment for job recommendation with LLM-based reward calculation.
    
    This extends the basic environment with an LLM-based simulator that:
    - Generates realistic user responses (APPLY, SAVE, CLICK, IGNORE)
    - Provides more nuanced feedback based on applicant-job matching
    - Can be fine-tuned for specific types of applicant behavior
    """
    
    def __init__(self, db_connector: Optional[DatabaseConnector] = None,
                 tensor_cache = None,
                 reward_scheme: Optional[Dict[str, float]] = None,
                 llm_model=None, tokenizer=None, device="cpu",
                 random_seed: int = 42):
        """
        Initialize the environment.
        
        Args:
            db_connector: Database connector for retrieving data.
            tensor_cache: Optional tensor cache for faster data access.
            reward_scheme: Dictionary mapping response types to reward values.
            llm_model: LLM model for simulating user responses.
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
        
        # LLM response mapping
        # This is used to convert the raw LLM output to a standardized response
        self.llm_response_mapping = {
            "apply": "APPLY",
            "save": "SAVE",
            "click": "CLICK",
            "ignore": "IGNORE"
        }
        
        logger.info("Initialized LLMSimulatorEnv")
    
    def reset(self, applicant_id: Optional[str] = None) -> torch.Tensor:
        """
        Reset the environment with a new applicant.
        
        This extends the parent class reset to load applicant profile information
        needed for LLM-based simulation.
        
        Args:
            applicant_id: Specific applicant ID to use, or None for random selection.
            
        Returns:
            torch.Tensor: Initial state vector.
        """
        # Use the parent class reset which now handles tensor cache if available
        state = super().reset(applicant_id)
        
        # Additional setup specific to LLM simulation (if needed)
        # This would load additional profile information for the LLM prompt
        
        return state
    
    def simulate_user_response(self, job: Dict) -> str:
        """
        Use an LLM to simulate user response to a job recommendation.
        
        This override replaces the random simulation with an LLM-based simulation
        that analyzes the applicant profile and job details to generate a realistic
        response.
        
        Args:
            job: Job document from the database.
            
        Returns:
            str: Response type ('APPLY', 'SAVE', 'CLICK', 'IGNORE')
        """
        if not self.llm_model or not self.tokenizer:
            logger.warning("LLM model or tokenizer not provided, falling back to random simulation")
            return super().simulate_user_response(job)
        
        try:
            # Get job details
            job_id = job.get("original_job_id")
            job_details = self.db.get_job_details(job_id)
            
            if not job_details:
                logger.warning(f"No details found for job {job_id}, falling back to random simulation")
                return super().simulate_user_response(job)
            
            job_title = job_details.get("job_title", "Unknown Position")
            job_description = job_details.get("description", "No description available.")
            
            # Get applicant profile
            applicant_profile = self.applicant_profiles.get(self.current_applicant_id)
            
            if not applicant_profile:
                logger.warning(f"No profile found for applicant {self.current_applicant_id}, falling back to random simulation")
                return super().simulate_user_response(job)
            
            # Build prompt for LLM
            prompt = self._build_llm_prompt(applicant_profile, job_title, job_description)
            
            # Get LLM response
            response = self._get_llm_decision(prompt)
            
            # Map LLM response to predefined response types
            mapped_response = self._map_llm_response(response)
            
            logger.debug(f"LLM simulation: {mapped_response} for job {job_title}")
            
            return mapped_response
            
        except Exception as e:
            logger.error(f"Error in LLM simulation: {e}")
            return super().simulate_user_response(job)
    
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
        """
        try:
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
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return "IGNORE"  # Default to IGNORE on error
    
    def _map_llm_response(self, response: str) -> str:
        """
        Map the LLM response to a predefined response type.
        
        Args:
            response: LLM response text.
            
        Returns:
            str: Mapped response type ('APPLY', 'SAVE', 'CLICK', 'IGNORE').
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
            # Default to IGNORE if response is unclear
            logger.warning(f"Unclear LLM response: {response}, defaulting to IGNORE")
            return "IGNORE"


class HybridEnv(LLMSimulatorEnv):
    """
    Environment for job recommendation with hybrid reward strategy.
    
    This environment combines cosine similarity and LLM-based rewards,
    with a configurable weighting between the two.
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
            llm_model: LLM model for simulating user responses.
            tokenizer: Tokenizer for the LLM model.
            device: Device to run the LLM on.
            cosine_weight: Weight for cosine similarity (0.0 to 1.0).
                           1.0 = pure cosine, 0.0 = pure LLM.
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
        
        # Override reward strategy
        self.reward_strategy = "hybrid"
        
        # Set cosine weight (0.0 to 1.0)
        self.cosine_weight = cosine_weight
        self.llm_weight = 1.0 - cosine_weight
        
        logger.info(f"Initialized HybridEnv with cosine_weight={cosine_weight}, llm_weight={self.llm_weight}")
    
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
        job_id = job.get("original_job_id")
        
        # Calculate cosine similarity reward
        cosine_reward = self.calculate_cosine_reward(action_idx)
        
        # Get LLM feedback reward
        response = self.simulate_user_response(job)
        llm_reward = self.reward_scheme.get(response, 0.0)
        
        # Combine rewards based on the weight parameter
        combined_reward = (self.cosine_weight * cosine_reward) + (self.llm_weight * llm_reward)
        
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
            "llm_weight": self.llm_weight,
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
        self.llm_weight = 1.0 - weight
        logger.info(f"Set cosine_weight to {weight}, llm_weight to {self.llm_weight}") 