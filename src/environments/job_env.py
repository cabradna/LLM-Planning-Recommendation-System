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
    
    def __init__(self,
                 tensor_cache = None,
                 reward_scheme: Optional[Dict[str, float]] = None,
                 reward_strategy: str = "cosine",
                 random_seed: int = 42):
        """
        Initialize the environment. Requires a TensorCache.
        
        Args:
            tensor_cache: Initialized TensorCache instance. Cannot be None.
            reward_scheme: Dictionary mapping response types to reward values.
            reward_strategy: Strategy for generating rewards ("cosine", "llm", or "hybrid").
            random_seed: Random seed for reproducibility.
        """
        # Tensor cache for fast data access - Now mandatory
        if tensor_cache is None or not hasattr(tensor_cache, 'initialized') or not tensor_cache.initialized:
             raise ValueError("An initialized TensorCache instance must be provided.")
        self.tensor_cache = tensor_cache
        self.using_cache = True # Always true now
        
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
        
        logger.info(f"Initialized JobRecommendationEnv with {reward_strategy} reward strategy using TensorCache")
        logger.info(f"Using tensor cache with {len(self.tensor_cache)} cached jobs")
    
    def reset(self, applicant_id: Optional[str] = None) -> torch.Tensor:
        """
        Reset the environment with a new applicant using TensorCache.
        
        Args:
            applicant_id: Specific applicant ID to use. Must be provided.
            
        Returns:
            torch.Tensor: Initial state vector.
            
        Raises:
            ValueError: If applicant_id is None or not found in cache.
            RuntimeError: If TensorCache is not initialized.
        """
        if applicant_id is None:
            raise ValueError("applicant_id must be provided for reset")
            
        if not self.tensor_cache or not self.tensor_cache.initialized:
             raise RuntimeError("TensorCache is not initialized. Cannot reset environment.")
        
        # Set current applicant ID
        self.current_applicant_id = applicant_id
        
        # Use tensor cache for faster data access - No DB fallback
        try:
            # Get state vector from cache
            self.current_state = self.tensor_cache.get_applicant_state(applicant_id)
            
            # Sample candidate jobs from cache
            # sample_jobs returns (sampled_docs, sampled_vectors, sampled_job_ids)
            sampled_docs, sampled_vectors, sampled_job_ids = self.tensor_cache.sample_jobs(n=100) 
            
            if not sampled_docs:
                 logger.warning(f"TensorCache returned no sampled jobs for applicant {applicant_id}. Action space will be empty.")
                 # Continue, but expect issues downstream if no actions are available

            self.candidate_jobs = sampled_docs
            self.job_vectors = sampled_vectors
            self.job_id_to_idx = {job_id: idx for idx, job_id in enumerate(sampled_job_ids)}
            
            logger.info(f"Environment reset with cached data for applicant {applicant_id}, {len(self.candidate_jobs)} candidate jobs sampled.")

        except KeyError as e:
            # Applicant or potentially job data not found in cache
            logger.error(f"Cache Key Error during reset for applicant {applicant_id}: {e}")
            raise ValueError(f"Failed to reset environment: Data for applicant {applicant_id} not found in TensorCache.") from e
        except Exception as e:
            # Catch other potential cache errors
            logger.error(f"Unexpected error during cache access in reset: {e}", exc_info=True)
            raise RuntimeError("Failed to reset environment due to an unexpected TensorCache error.") from e
            
        return self.current_state
    
    def step(self, action_idx: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Take a step in the environment by recommending a job.
        
        Args:
            action_idx: Index of the job to recommend from the **currently sampled** candidate_jobs list.
            
        Returns:
            Tuple containing:
                - torch.Tensor: Next state (same as current_state)
                - float: Reward value
                - bool: Done flag (always False)
                - Dict: Additional info
        """
        if not self.candidate_jobs:
             # Handle case where reset resulted in no candidate jobs
             logger.warning("Step called with no candidate jobs available.")
             # Return zero reward, same state, not done. Or could raise error.
             return self.current_state, 0.0, False, {"error": "No candidate jobs"}
             
        if action_idx < 0 or action_idx >= len(self.candidate_jobs):
            raise ValueError(f"Invalid action index: {action_idx}. Must be between 0 and {len(self.candidate_jobs)-1} for the current sample.")
        
        # Get the recommended job from the current sample
        job = self.candidate_jobs[action_idx]
        # Use the _id directly from the cached doc (should be correct type, e.g., ObjectId)
        job_id = job["_id"] 
        
        # Get reward based on the selected reward strategy
        if self.reward_strategy == "cosine":
            reward = self.calculate_cosine_reward(action_idx)
        elif self.reward_strategy in ["llm", "hybrid"]:
             # This relies on subclasses implementing LLM/Hybrid logic
             # We need to ensure the subclasses also don't use DB
             # This call will route to LLMSimulatorEnv.step or HybridEnv.step if applicable
             # We handle reward calculation within those subclass methods or here if base class needs extension
             # For now, assume subclasses handle it correctly without DB. If called on base class, raise error.
            raise NotImplementedError(
                f"Reward strategy '{self.reward_strategy}' requires LLMSimulatorEnv or HybridEnv subclass."
            )
        else:
            raise ValueError(f"Unsupported reward strategy: {self.reward_strategy}")

        # In the current phase, state doesn't change after action
        next_state = self.current_state
        
        # Episode doesn't end
        done = False
        
        # Additional info
        info = {
            "job_id": str(job_id), # Convert to string for consistent logging/info
            "reward_strategy": self.reward_strategy,
            "job_title": job.get("job_title", "N/A"),
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
                logger.debug(f"Dimension mismatch: job vector {job_vector.shape[0]}, applicant state {self.current_state.shape[0]}")
                
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
                logger.debug(f"Dimension mismatch: job vector {job_vector.shape[0]}, applicant state {self.current_state.shape[0]}")
                
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
                logger.debug(f"Dimension mismatch in calculate_all_cosine_rewards: job vector {job_vectors_tensor.shape[1]}, applicant state {self.current_state.shape[0]}")
                
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
        Clean up resources (Cache clearing is handled elsewhere).
        """
        logger.info("JobRecommendationEnv closed (No DB connection to close).")
            
    def setup_llm(self, llm_model, tokenizer, device="auto"):
        """
        Set up an LLM for reward calculation. Switches to HybridEnv if current strategy is cosine.
        
        Args:
            llm_model: The LLM model to use for simulation.
            tokenizer: Tokenizer for the LLM model.
            device: Device to run the LLM on.
            
        Returns:
            The potentially new environment instance (HybridEnv or LLMSimulatorEnv).
        """
        # Determine the appropriate environment class based on the reward strategy
        if self.reward_strategy == "cosine":
            # Create a new hybrid environment with the provided LLM
            # Pass only necessary non-DB args
            hybrid_env = HybridEnv(
                # No db_connector
                tensor_cache=self.tensor_cache,
                reward_scheme=self.reward_scheme,
                llm_model=llm_model,
                tokenizer=tokenizer,
                device=device,
                cosine_weight=1.0,  # Start with cosine only
                random_seed=self.rng.randint(0, 2**32-1)
            )
            
            # Transfer state (carefully, ensure no DB-related state)
            hybrid_env.current_applicant_id = self.current_applicant_id
            hybrid_env.current_state = self.current_state
            hybrid_env.candidate_jobs = self.candidate_jobs
            hybrid_env.job_vectors = self.job_vectors
            hybrid_env.job_id_to_idx = self.job_id_to_idx
            # Ensure applicant_profiles are handled if needed (LLMSimulatorEnv part)
            
            logger.info("Switching to hybrid environment with LLM")
            return hybrid_env
        
        elif isinstance(self, LLMSimulatorEnv): # Check if already an LLM env
            # Update the LLM components directly
            self.llm_model = llm_model
            self.tokenizer = tokenizer
            self.device = device
            logger.info(f"Updated LLM components in {self.__class__.__name__}")
            return self # Return self as we modified in-place
        else:
             # This case should ideally not be hit if logic follows cosine -> hybrid
             logger.warning(f"setup_llm called on base JobRecommendationEnv with non-cosine strategy '{self.reward_strategy}'. Cannot setup LLM.")
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
    Environment that uses an LLM to simulate user responses, using only TensorCache.
    """
    
    def __init__(self,
                 tensor_cache = None,
                 reward_scheme: Optional[Dict[str, float]] = None,
                 llm_model=None, tokenizer=None, device="cpu",
                 random_seed: int = 42):
        """
        Initialize the LLM-based simulation environment. Requires TensorCache.
        
        Args:
            tensor_cache: Initialized TensorCache instance.
            reward_scheme: Dictionary mapping response types to reward values.
            llm_model: The LLM model to use for simulation.
            tokenizer: Tokenizer for the LLM model.
            device: Device to run the LLM on.
            random_seed: Random seed for reproducibility.
        """
        # Call parent __init__ WITHOUT db_connector
        super().__init__(
            # No db_connector
            tensor_cache=tensor_cache,
            reward_scheme=reward_scheme,
            reward_strategy="llm", # Explicitly set strategy for LLM env
            random_seed=random_seed
        )
        
        # LLM components for reward calculation
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.device = device
        
        # Cache for applicant profiles (Text data needs to be handled)
        # This cache might need rethinking if text profiles aren't in TensorCache
        self.applicant_profiles = {} 
        
        logger.info("Initialized LLMSimulatorEnv with LLM-based reward strategy (Cache-only)")
    
    def reset(self, applicant_id: Optional[str] = None) -> torch.Tensor:
        """
        Reset the environment with a new applicant. Loads state and text profile from cache.
        
        Args:
            applicant_id: Specific applicant ID to use.
            
        Returns:
            torch.Tensor: Initial state vector.
            
        Raises:
            ValueError: If applicant state or text profile is missing from cache (via KeyError).
        """
        # Call parent reset first to get state and sample jobs from cache
        state = super().reset(applicant_id)
        
        # --- Get Applicant Profile Text from Cache ---
        try:
            # Use string representation of applicant_id for lookup
            applicant_id_str = str(applicant_id)
            # This will raise KeyError if the applicant was skipped during cache load
            profile_text_data = self.tensor_cache.get_applicant_profile_text(applicant_id_str)
            
            # Store the successfully loaded profile text
            self.applicant_profiles[applicant_id_str] = profile_text_data
            logger.debug(f"Successfully loaded applicant text profile for {applicant_id_str} from cache.")

        except KeyError as e:
             # Profile text not found in cache (applicant was skipped during load)
             logger.error(f"Applicant profile text for {applicant_id_str} not found in TensorCache. Cannot proceed with LLM simulation for this applicant.")
             # Raise an error because LLM simulation depends on the profile text
             raise ValueError(f"Missing applicant text profile in cache for {applicant_id_str}. Cannot reset LLMSimulatorEnv.") from e
        except Exception as e:
             logger.error(f"Unexpected error fetching applicant profile text from cache for {applicant_id_str}: {e}", exc_info=True)
             raise RuntimeError("Failed to reset LLMSimulatorEnv due to cache error retrieving profile text.") from e
        # ------------------------------------
        
        return state

    def step(self, action_idx: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Take a step using LLM for reward calculation. Relies on TensorCache for job details.
        """
        if not self.candidate_jobs:
             logger.warning("LLMSimulatorEnv Step called with no candidate jobs.")
             return self.current_state, 0.0, False, {"error": "No candidate jobs"}
             
        if action_idx < 0 or action_idx >= len(self.candidate_jobs):
            raise ValueError(f"Invalid action index: {action_idx}. Must be between 0 and {len(self.candidate_jobs)-1}")
             
        # Get the recommended job from the current sample
        job = self.candidate_jobs[action_idx]
        job_id = job["_id"] # Use original type (_id)
        
        # Calculate reward using LLM simulation
        try:
            response = self.simulate_user_response(job) # Pass the job doc from cache sample
            reward = self.reward_scheme.get(response, 0.0)
        except Exception as e:
            logger.error(f"Error during LLM simulation or reward calculation: {e}", exc_info=True)
            reward = 0.0 # Default reward on error
            response = "ERROR"

        # State doesn't change
        next_state = self.current_state
        done = False
        
        # Additional info
        info = {
            "job_id": str(job_id),
            "reward_strategy": self.reward_strategy,
            "llm_reward": reward,
            "response": response,
            "job_title": job.get("job_title", "N/A"),
            "applicant_id": self.current_applicant_id
        }
        
        return next_state, reward, done, info

    
    def simulate_user_response(self, job: Dict) -> str:
        """
        Use an LLM to simulate user response. Gets job details from TensorCache.
        Handles potentially missing applicant profile text.
        
        Args:
            job: Job document (dict) from the current `candidate_jobs` sample.
            
        Returns:
            str: Response type ('APPLY', 'SAVE', 'CLICK', 'IGNORE')
            
        Raises:
            RuntimeError: If LLM components missing or essential job data missing from cache.
        """
        if not self.llm_model or not self.tokenizer:
            raise RuntimeError("LLM model or tokenizer not provided for simulation.")
        
        job_id = job["_id"] # Original type
        job_id_str = str(job_id) # String for metadata lookup if needed

        # Get job details from the provided job dict (already fetched during sampling)
        # If more details are needed, they MUST come from cache metadata.
        job_title = job.get("job_title", None)
        job_description = job.get("description", None)

        # Attempt to get more details from metadata cache if needed and not in job doc
        if job_title is None or job_description is None:
             try:
                  metadata = self.tensor_cache.get_job_metadata(job_id_str) # Use string ID for lookup
                  job_title = job_title or metadata.get("job_title", "Unknown Position")
                  job_description = job_description or metadata.get("description", "No description available.")
             except KeyError:
                  logger.error(f"Metadata not found in TensorCache for job {job_id_str}. Using available data.")
                  job_title = job_title or "Unknown Position"
                  job_description = job_description or "No description available."
             except Exception as e:
                  logger.error(f"Error retrieving metadata for job {job_id_str} from cache: {e}")
                  job_title = job_title or "Unknown Position"
                  job_description = job_description or "No description available."

        if not job_title or not job_description:
             # If essential details are still missing after checking cache
             raise RuntimeError(f"Essential details (title, description) missing for job {job_id_str} even after checking cache.")
             
        # Get applicant profile from self.applicant_profiles (populated during reset)
        applicant_id_str = str(self.current_applicant_id) # Ensure string key
        applicant_profile = self.applicant_profiles.get(applicant_id_str)

        if not applicant_profile:
             # This should not happen if reset succeeded, but handle defensively
             logger.error(f"Internal Error: Applicant profile for {applicant_id_str} not found in self.applicant_profiles despite successful reset.")
             return "IGNORE" # Default response

        # No need to check for _load_error anymore, as only valid profiles are stored

        # Build prompt for LLM using the fetched/cached profile
        prompt = self._build_llm_prompt(
            applicant_bio=applicant_profile.get('bio', 'Bio unavailable.'), # Use default only if field missing in valid doc
            applicant_resume=applicant_profile.get('resume_string', 'Resume unavailable.'), # Use default only if field missing
            job_title=job_title,
            job_description=job_description
        )
        
        # Get LLM response
        response = self._get_llm_decision(prompt)
        
        # Map LLM response to predefined response types
        mapped_response = self._map_llm_response(response)
        
        logger.debug(f"LLM simulation: {mapped_response} for job {job_title}")
        
        return mapped_response
    
    def _build_llm_prompt(self, applicant_bio: str, applicant_resume: str, job_title: str, job_description: str) -> str:
        """
        Build a prompt for the LLM to simulate user response.
        
        Args:
            applicant_bio: Applicant bio.
            applicant_resume: Applicant resume.
            job_title: Job title.
            job_description: Job description.
            
        Returns:
            str: Prompt for the LLM.
        """
        return f"""
You are simulating a job seeker's response to a job recommendation.

JOB SEEKER PROFILE:
{applicant_bio}

RESUME:
{applicant_resume}

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
        Get a decision from the LLM, ensuring input tensors are on the correct device.
        
        Args:
            prompt: Prompt for the LLM.
            
        Returns:
            str: LLM response text.
            
        Raises:
            Exception: Any errors during LLM processing are propagated up
        """
        # Tokenize the prompt (creates tensors on CPU by default)
        inputs = self.tokenizer(prompt, return_tensors="pt") 

        # --- Move input tensors to the LLM's primary device --- 
        try:
            # Determine the target device from the LLM model itself
            # This handles models loaded with device_map="auto"
            model_device = self.llm_model.device 
            # Move all tensors in the inputs dictionary to the model's device
            inputs = {key: tensor.to(model_device) for key, tensor in inputs.items()}
            logger.debug(f"Moved tokenizer inputs to device: {model_device}")
        except Exception as e:
            # Fallback or error handling if device determination/moving fails
            logger.error(f"Could not move inputs to model device: {e}. Using environment device: {self.device}")
            # As a fallback, try the environment's device (might still cause issues)
            inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
        # --- End device move --- 

        # Generate response
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.2, # Keep temp low
                do_sample=False, # CHANGE: Use greedy decoding
                pad_token_id=self.tokenizer.eos_token_id # Ensure pad token is set
            )
        
        # Decode the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the part after the prompt
        response_only = response[len(prompt):].strip()
        
        return response_only
    
    def _map_llm_response(self, response: str) -> str:
        """
        Map the LLM response to a predefined response type.
        Falls back to IGNORE if mapping is unclear.
        
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
            # CHANGE: Log warning and default to IGNORE instead of raising error
            logger.warning(f"Unclear LLM response: '{response}'. Defaulting to IGNORE.")
            return "IGNORE"


class HybridEnv(LLMSimulatorEnv):
    """
    Hybrid environment combining cosine and LLM rewards, using only TensorCache.
    """
    
    def __init__(self,
                 tensor_cache = None,
                 reward_scheme: Optional[Dict[str, float]] = None,
                 llm_model=None, tokenizer=None, device="cpu",
                 cosine_weight: float = 0.5,
                 random_seed: int = 42):
        """
        Initialize the hybrid environment. Requires TensorCache.
        
        Args:
            tensor_cache: Initialized TensorCache instance.
            reward_scheme: Dictionary mapping response types to reward values.
            llm_model: The LLM model to use for simulation.
            tokenizer: Tokenizer for the LLM model.
            device: Device to run the LLM on.
            cosine_weight: Weight for cosine similarity [0.0, 1.0].
            random_seed: Random seed for reproducibility.
        """
        # Call LLMSimulatorEnv init WITHOUT db_connector
        super().__init__(
            # No db_connector
            tensor_cache=tensor_cache,
            reward_scheme=reward_scheme,
            llm_model=llm_model,
            tokenizer=tokenizer,
            device=device,
            random_seed=random_seed
        )
        
        # Set hybrid parameters
        if not (0.0 <= cosine_weight <= 1.0):
            raise ValueError(f"Cosine weight must be between 0.0 and 1.0, got {cosine_weight}")
        self.cosine_weight = cosine_weight
        self.reward_strategy = "hybrid" # Override strategy from LLMSimulatorEnv
        
        logger.info(f"Initialized HybridEnv with cosine_weight={cosine_weight} (Cache-only)")
    
    def step(self, action_idx: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Take a step, combining cosine and LLM rewards using TensorCache.
        """
        if not self.candidate_jobs:
             logger.warning("HybridEnv Step called with no candidate jobs.")
             return self.current_state, 0.0, False, {"error": "No candidate jobs"}
             
        if action_idx < 0 or action_idx >= len(self.candidate_jobs):
            raise ValueError(f"Invalid action index: {action_idx}. Must be between 0 and {len(self.candidate_jobs)-1}")
        
        # Get the recommended job from the current sample
        job = self.candidate_jobs[action_idx]
        job_id = job["_id"] # Original type
        
        # Calculate cosine similarity reward (uses self.calculate_cosine_reward from base class)
        # This method should already be cache-only
        try:
             cosine_reward = self.calculate_cosine_reward(action_idx)
        except Exception as e:
             logger.error(f"Error calculating cosine reward for action {action_idx}: {e}", exc_info=True)
             cosine_reward = 0.0 # Default on error

        # Get LLM feedback reward (uses self.simulate_user_response from LLMSimulatorEnv)
        # This method is now modified to be cache-only
        try:
             response = self.simulate_user_response(job)
             llm_reward = self.reward_scheme.get(response, 0.0)
        except Exception as e:
             logger.error(f"Error during LLM simulation for hybrid reward: {e}", exc_info=True)
             llm_reward = 0.0 # Default reward on error
             response = "ERROR"
             
        # Combine rewards based on the weight parameter
        # Ensure weights are valid
        w_cos = max(0.0, min(1.0, self.cosine_weight))
        w_llm = 1.0 - w_cos
        combined_reward = (w_cos * cosine_reward) + (w_llm * llm_reward)
        
        # State doesn't change
        next_state = self.current_state
        done = False
        
        # Additional info
        info = {
            "job_id": str(job_id),
            "reward_strategy": "hybrid",
            "cosine_reward": cosine_reward,
            "llm_reward": llm_reward,
            "combined_reward": combined_reward,
            "cosine_weight": self.cosine_weight,
            "response": response,
            "job_title": job.get("job_title", "N/A"),
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