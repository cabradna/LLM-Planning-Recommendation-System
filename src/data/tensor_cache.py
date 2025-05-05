"""
Tensor Cache module for efficient data management during training.

This module provides the TensorCache class which:
1. Preloads data from MongoDB to GPU memory
2. Stores job and applicant data as optimized tensors
3. Provides fast sampling and lookup during training
"""

import torch
import numpy as np
import logging
import random
from typing import Dict, List, Tuple, Optional, Any
from bson import ObjectId
import time

logger = logging.getLogger(__name__)

class TensorCache:
    """
    Efficient tensor-based cache for job recommendation data.
    
    Preloads and stores data in optimized PyTorch tensors for fast access
    during training, significantly reducing database queries.
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the tensor cache.
        
        Args:
            device: Device to store tensors on, defaults to CUDA if available
        """
        self.device = torch.device(device)
        
        # Applicant data
        self.applicant_states = {}  # applicant_id -> tensor
        self.applicant_profiles_text = {}  # applicant_id -> text profile
        
        # Job data
        self.job_vectors = None     # tensor [num_jobs, embedding_dim]
        self.job_text_ids = []           # list of job IDs in embeddings collection - ObejctId type
        self.job_metadata = {}      # job_id -> metadata dict
        self.job_embeddings_ids = []  # list of original job IDs (maps indices to IDs) - ObjectId type
        
        # Stats for monitoring
        self.cache_hits = 0
        self.cache_misses = 0
        self.initialized = False
        self.initialization_time = 0
        
        logger.info(f"Initialized TensorCache on device: {self.device}")
    
    def copy_from_database(self, db_connector, applicant_ids=None):
        """
        Load all valid jobs and specified applicants from database to GPU memory.
        
        Args:
            db_connector: DatabaseConnector instance for fetching data
            applicant_ids: Optional list of applicant IDs to preload
        
        Returns:
            self: For method chaining
        """
        start_time = time.time()
        logger.info("Starting data copy from database to tensor cache")
        
        # 1. Load applicant data if provided
        if applicant_ids:
            self._load_applicants(db_connector, applicant_ids)
        
        # 2. Load all valid jobs
        self._load_all_valid_jobs(db_connector)
        
        self.initialized = True
        self.initialization_time = time.time() - start_time
        logger.info(f"Cache initialization completed in {self.initialization_time:.2f} seconds")
        logger.info(f"Cached {len(self.job_text_ids)} jobs and {len(self.applicant_states)} applicants")
        
        return self
    
    def _load_applicants(self, db_connector, applicant_ids):
        """
        Load applicant state vectors and text profiles from database.
        Skips applicants if either state vector or text profile cannot be loaded successfully.
        
        Args:
            db_connector: DatabaseConnector instance
            applicant_ids: List of applicant IDs to load
        """
        logger.info(f"Attempting to load {len(applicant_ids)} applicants (state vectors and text profiles) into tensor cache...")
        
        candidates_text_collection = db_connector.db[db_connector.collections["candidates_text"]]
        successfully_loaded_count = 0
        
        for applicant_id_obj in applicant_ids: 
            applicant_id_str = str(applicant_id_obj)
            applicant_state = None
            candidate_doc = None
            
            # --- 1. Load State Vector --- 
            try:
                 applicant_state = db_connector.get_applicant_state(applicant_id_obj)
                 if applicant_state is None:
                     # Handle case where get_applicant_state returns None without error
                     logger.warning(f"State vector not found for applicant {applicant_id_str}. Skipping applicant.")
                     continue
            except Exception as e:
                 logger.error(f"Error fetching state vector for applicant {applicant_id_str}: {e}. Skipping applicant.")
                 continue # Skip this applicant entirely
                 
            # --- 2. Load Text Profile --- 
            try:
                # Find the text document using the _id which should match applicant_id_obj
                candidate_doc = candidates_text_collection.find_one({"_id": applicant_id_obj})
                
                if candidate_doc is None:
                    logger.warning(f"No text data found in '{db_connector.collections['candidates_text']}' for applicant ID {applicant_id_str}. Skipping applicant.")
                    continue # Skip this applicant

            except Exception as e:
                logger.error(f"Error fetching text profile for applicant {applicant_id_str}: {e}. Skipping applicant.", exc_info=True)
                continue # Skip this applicant

            # --- 3. Format Profile and Store (only if both loaded successfully) --- 
            # Format candidate profile for storage
            try:
                profile = {
                    "bio": candidate_doc.get("bio", "Bio not available."), # Use default only if field missing, not if doc missing
                    "skills": candidate_doc.get("skills", []), 
                    "experience": candidate_doc.get("experience", [])
                }
                # Format the resume string
                resume_parts = []
                if profile['skills']:
                    resume_parts.append("Skills:\n- " + ', '.join(map(str, profile['skills']))) 
                if profile['experience']:
                    exp_strings = []
                    for exp in profile['experience']:
                        title = exp.get('title', 'N/A')
                        company = exp.get('company', 'N/A')
                        exp_strings.append(f"- {title} at {company}")
                    if exp_strings:
                        resume_parts.append("\nExperience:\n" + '\n'.join(exp_strings))
                profile['resume_string'] = '\n\n'.join(resume_parts) if resume_parts else "Resume details not available."
                
                # Store state vector (on device)
                self.applicant_states[applicant_id_str] = applicant_state.to(self.device)
                # Store formatted text profile (on CPU)
                self.applicant_profiles_text[applicant_id_str] = profile
                successfully_loaded_count += 1
                
            except Exception as e:
                 logger.error(f"Error formatting profile or storing data for applicant {applicant_id_str} even after finding documents: {e}. Skipping applicant.", exc_info=True)
                 # Clean up potentially partially stored data for this applicant just in case
                 if applicant_id_str in self.applicant_states:
                     del self.applicant_states[applicant_id_str]
                 if applicant_id_str in self.applicant_profiles_text:
                     del self.applicant_profiles_text[applicant_id_str]
                 continue # Skip this applicant
            
        logger.info(f"Successfully loaded state vectors and text profiles for {successfully_loaded_count} out of {len(applicant_ids)} requested applicants.")
    
    def _load_all_valid_jobs(self, db_connector):
        """
        Load all jobs with valid embeddings from database.
        
        Args:
            db_connector: DatabaseConnector instance
        """
        logger.info("Loading all jobs and embeddings from database")
        
        # 1. Get all job documents
        all_jobs = list(db_connector.db[db_connector.collections["jobs_text"]].find({}))
        logger.info(f"Retrieved {len(all_jobs)} total jobs from database")
        
        # 2. Get all job embeddings
        all_embeddings = list(db_connector.db[db_connector.collections["job_embeddings"]].find({}))
        logger.info(f"Retrieved {len(all_embeddings)} job embeddings")
        
        # 3. Filter for jobs with complete embeddings
        valid_jobs = []
        self.job_embeddings_ids = []
        for job in all_jobs: # Iterating over job text collection
            job_id = str(job["_id"])
            # Find matching embedding
            # matching_embedding will be the embedding item that corresponds to the job (text)
            matching_embedding = next((emb for emb in all_embeddings if str(emb["original_job_id"]) == job_id), None)
            
            if matching_embedding:
                # Check if all required embedding vectors are present and have non-zero length
                if (len(matching_embedding.get("job_title_embeddings", [])) > 0 and 
                    len(matching_embedding.get("tech_skills_vectors", [])) > 0 and 
                    len(matching_embedding.get("soft_skills_embeddings", [])) > 0 and
                    len(matching_embedding.get("experience_requirements_embeddings", [])) > 0):
                    valid_jobs.append(job) # append the job (text info) if the embeddings are found
                    self.job_embeddings_ids.append(matching_embedding["_id"]) # append the embedding ID
        
        logger.info(f"Found {len(valid_jobs)} jobs with complete embeddings")
        
        # 4. Store job IDs for later reference
        self.job_text_ids = [job["_id"] for job in valid_jobs]
        
        # 5. Store job metadata
        self.job_metadata = {job["_id"]: job for job in valid_jobs}
        
        # 6. Get and store job vectors
        job_vectors = db_connector.get_job_vectors(self.job_text_ids)
        self.job_vectors = torch.stack(job_vectors).to(self.device)
        logger.info(f"Created job vectors tensor with shape {self.job_vectors.shape}")
    
    def get_applicant_state(self, applicant_id):
        """
        Get applicant state vector from cache.
        
        Args:
            applicant_id: ID of the applicant (can be ObjectId or str).
            
        Returns:
            torch.Tensor: Applicant state vector.
            
        Raises:
            KeyError: If applicant not in cache.
        """
        # Ensure we use the string representation for dictionary lookup
        applicant_id_str = str(applicant_id)
        
        if applicant_id_str in self.applicant_states:
            self.cache_hits += 1
            return self.applicant_states[applicant_id_str]
        else:
            self.cache_misses += 1
            # Raise error using the string representation for clarity
            raise KeyError(f"Applicant {applicant_id_str} not found in cache")
    
    def sample_jobs(self, n=100, exclude_ids=None):
        """
        Sample n random jobs from the cache using optimized index access.
        
        Args:
            n: Number of jobs to sample
            exclude_ids: Optional list of job IDs to exclude from sampling
            
        Returns:
            Tuple containing:
                - List[Dict]: List of sampled job metadata documents
                - torch.Tensor: Tensor of corresponding job vectors [n, embedding_dim]
                - List[ObjectId]: List of sampled job ObjectIds
        """
        if not self.initialized:
            raise RuntimeError("Cache not initialized, call copy_from_database first")
        
        num_cached_jobs = len(self.job_text_ids)
        if num_cached_jobs == 0:
            logger.warning("sample_jobs: Cache contains no jobs to sample from.")
            return [], torch.empty((0, self.job_vectors.shape[1] if self.job_vectors is not None else 0), device=self.device), []
            
        # Determine available indices (indices corresponding to positions in job_text_ids/job_vectors)
        available_indices = list(range(num_cached_jobs))
        
        # Exclude specific job IDs if provided
        if exclude_ids:
            # Find indices to exclude efficiently (might need optimization if exclude_ids is large)
            # For moderate sizes, list comprehension is okay.
            # Ensure exclude_ids contains ObjectIds if that's what job_text_ids holds.
            exclude_indices_set = {i for i, job_id in enumerate(self.job_text_ids) if job_id in exclude_ids}
            available_indices = [i for i in available_indices if i not in exclude_indices_set]
        
        if not available_indices:
             logger.warning("sample_jobs: No available jobs after exclusion.")
             return [], torch.empty((0, self.job_vectors.shape[1]), device=self.device), []
             
        # Sample n indices (or all if fewer than n are available)
        sample_size = min(n, len(available_indices))
        sampled_indices = random.sample(available_indices, sample_size)
        
        # --- Optimized Retrieval using Indices --- 
        # Get vectors for sampled jobs using tensor indexing
        sampled_vectors = self.job_vectors[sampled_indices]
        
        # Get job IDs for sampled jobs using list comprehension
        sampled_job_ids = [self.job_text_ids[i] for i in sampled_indices]
        
        # Get job metadata documents using list comprehension and the parallel structure
        # Assumes job_metadata keys are ObjectIds matching job_text_ids
        # Directly fetch the metadata dict corresponding to the sampled index's job_id
        sampled_docs = []
        for index in sampled_indices:
            job_id = self.job_text_ids[index] # Get the ObjectId
            try:
                 metadata_doc = self.job_metadata[job_id] # Lookup using ObjectId key
                 # We can return the full metadata doc, or create a simplified one if needed
                 # Let's return the full one for now, as env relies on fields like 'description'
                 sampled_docs.append(metadata_doc)
            except KeyError:
                 # This should ideally not happen if loading was correct, but log defensively
                 logger.error(f"sample_jobs: Internal inconsistency! Job ID {job_id} (index {index}) not found in metadata keys during optimized sampling.")
                 # Decide how to handle: skip? add placeholder? For now, skip.
                 pass 
                 
        # Check if the number of docs matches vectors/ids (due to potential KeyErrors above)
        if len(sampled_docs) != len(sampled_job_ids):
            logger.warning(f"sample_jobs: Mismatch after metadata lookup - sampled {len(sampled_job_ids)} IDs/vectors but found {len(sampled_docs)} docs.")
            # This might require re-filtering sampled_vectors and sampled_job_ids to match sampled_docs
            # Or deciding if the inconsistency is acceptable.
            # For now, returning potentially mismatched lists.
            
        logger.debug(f"sample_jobs: Finished optimized sampling. Found {len(sampled_docs)} documents.")
        return sampled_docs, sampled_vectors, sampled_job_ids
    
    def calculate_cosine_similarities(self, applicant_state):
        """
        Calculate cosine similarities between applicant state and all jobs.
        
        Handles potential dimension mismatch by comparing only the overlapping
        initial dimensions (e.g., comparing applicant skills to job skills).
        
        Args:
            applicant_state: Applicant state tensor [state_dim]
            
        Returns:
            torch.Tensor: Cosine similarities for all jobs [num_jobs]
        """
        if not self.initialized:
            raise RuntimeError("Cache not initialized, call copy_from_database first")
        
        # Ensure applicant_state is on the correct device
        applicant_state = applicant_state.to(self.device)
        job_vectors = self.job_vectors # Shape: [num_jobs, job_dim]
        
        # --- Handle Dimension Mismatch --- 
        state_dim = applicant_state.shape[0]
        job_dim = job_vectors.shape[1]
        
        # Determine the dimension to use for comparison (usually the smaller one)
        # This handles the case where job vectors might have more dimensions (e.g., title, experience)
        # than the applicant state (e.g., just skills).
        comparison_dim = min(state_dim, job_dim)
        
        if state_dim != job_dim:
            logger.debug(f"Dimension mismatch detected: Applicant State ({state_dim}) vs Job Vectors ({job_dim}). Comparing first {comparison_dim} dimensions.")
            # Slice both tensors to the comparison dimension
            sliced_applicant_state = applicant_state[:comparison_dim]
            sliced_job_vectors = job_vectors[:, :comparison_dim]
        else:
            # Dimensions match, use tensors as is
            sliced_applicant_state = applicant_state
            sliced_job_vectors = job_vectors
        # --- End Dimension Handling ---
            
        # Calculate cosine similarity using the potentially sliced vectors
        # Normalize vectors
        normalized_applicant = sliced_applicant_state / (sliced_applicant_state.norm() + 1e-8) # Add epsilon for numerical stability
        normalized_jobs = sliced_job_vectors / (sliced_job_vectors.norm(dim=1, keepdim=True) + 1e-8) # Add epsilon
        
        # Calculate dot product of normalized vectors (which equals cosine similarity)
        # Ensure normalized_applicant is treated as a column vector for matmul
        similarities = torch.matmul(normalized_jobs, normalized_applicant.unsqueeze(1)).squeeze() 
        
        return similarities
    
    def get_job_vector(self, job_id):
        """
        Get vector for a specific job from cache.
        
        Args:
            job_id: ID of the job
            
        Returns:
            torch.Tensor: Job vector
            
        Raises:
            KeyError: If job not in cache
        """
        if not self.initialized:
            raise RuntimeError("Cache not initialized, call copy_from_database first")
        
        try:
            idx = self.job_text_ids.index(job_id)
            self.cache_hits += 1
            return self.job_vectors[idx]
        except ValueError:
            self.cache_misses += 1
            raise KeyError(f"Job {job_id} not found in cache")
    
    def get_job_metadata(self, job_id):
        """
        Get metadata for a specific job from cache.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Dict: Job metadata
            
        Raises:
            KeyError: If job not in cache
        """
        if job_id in self.job_metadata:
            self.cache_hits += 1
            return self.job_metadata[job_id]
        else:
            self.cache_misses += 1
            raise KeyError(f"Job {job_id} not found in cache")
    
    def get_valid_job_indices(self):
        """
        Get a list of valid job indices that can be used for lookup in the cache.
        
        Returns:
            List[int]: List of valid job indices
        """
        if not self.initialized:
            raise RuntimeError("Cache not initialized, call copy_from_database first")
        
        return list(range(len(self.job_text_ids)))
    
    def get_job_id(self, index):
        """
        Get job ID for a specific index.
        
        Args:
            index: Index of the job in the cache
            
        Returns:
            str: Job ID
            
        Raises:
            IndexError: If index is out of range
        """
        if not self.initialized:
            raise RuntimeError("Cache not initialized, call copy_from_database first")
        
        if index < 0 or index >= len(self.job_text_ids):
            raise IndexError(f"Job index {index} out of range (0-{len(self.job_text_ids)-1})")
        
        return self.job_text_ids[index]
    
    def get_job_vector_by_index(self, index):
        """
        Get job vector for a specific index in the cache.
        
        Args:
            index: Index of the job in the cache
            
        Returns:
            torch.Tensor: Job vector
            
        Raises:
            IndexError: If index is out of range
        """
        if not self.initialized:
            raise RuntimeError("Cache not initialized, call copy_from_database first")
        
        if index < 0 or index >= len(self.job_text_ids):
            raise IndexError(f"Job index {index} out of range (0-{len(self.job_text_ids)-1})")
        
        self.cache_hits += 1
        return self.job_vectors[index]
    
    def clear(self):
        """
        Clear all cached data to free memory.
        """
        logger.info("Clearing tensor cache...")
        
        # Clear applicant data
        self.applicant_states = {}
        self.applicant_profiles_text = {}
        
        # Clear job data
        if self.job_vectors is not None:
            self.job_vectors = None
        
        self.job_text_ids = []
        self.job_metadata = {}
        
        # Reset stats
        self.initialized = False
        
        logger.info("Tensor cache cleared successfully")
        
        return self
    
    def __len__(self):
        """Return the number of cached jobs."""
        return len(self.job_text_ids)
    
    def cache_stats(self):
        """Return cache statistics."""
        return {
            "job_count": len(self.job_text_ids),
            "applicant_state_count": len(self.applicant_states),
            "applicant_profile_count": len(self.applicant_profiles_text),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            "initialized": self.initialized,
            "initialization_time": self.initialization_time,
            "device": str(self.device),
        } 

    def get_applicant_profile_text(self, applicant_id):
        """
        Get applicant text profile (bio, skills, experience) from cache.
        Assumes profile exists if key is present, as placeholders are not stored.
        
        Args:
            applicant_id: ID of the applicant (string)
            
        Returns:
            Dict: Applicant text profile dictionary
            
        Raises:
            KeyError: If applicant profile text not in cache (means applicant was skipped during load)
        """
        applicant_id_str = str(applicant_id) # Ensure string key
        if applicant_id_str in self.applicant_profiles_text:
            self.cache_hits += 1 # Count as cache hit
            return self.applicant_profiles_text[applicant_id_str]
        else:
            self.cache_misses += 1
            # This key error now signifies the applicant was skipped during load
            raise KeyError(f"Applicant text profile for {applicant_id_str} not found in cache (applicant likely skipped during load due to missing state/text data).") 