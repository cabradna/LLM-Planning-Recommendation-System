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
        
        # Job data
        self.job_vectors = None     # tensor [num_jobs, embedding_dim]
        self.job_ids = []           # list of job IDs (maps indices to IDs)
        self.job_metadata = {}      # job_id -> metadata dict
        
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
        logger.info(f"Cached {len(self.job_ids)} jobs and {len(self.applicant_states)} applicants")
        
        return self
    
    def _load_applicants(self, db_connector, applicant_ids):
        """
        Load applicant state vectors from database.
        
        Args:
            db_connector: DatabaseConnector instance
            applicant_ids: List of applicant IDs to load
        """
        logger.info(f"Loading {len(applicant_ids)} applicants into tensor cache")
        
        for applicant_id in applicant_ids:
            # Get applicant state from database
            applicant_state = db_connector.get_applicant_state(applicant_id)
            
            # Store in cache on the correct device
            self.applicant_states[applicant_id] = applicant_state.to(self.device)
            
        logger.info(f"Loaded {len(self.applicant_states)} applicant states into cache")
    
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
        for job in all_jobs:
            job_id = str(job["_id"])
            # Find matching embedding
            matching_embedding = next((emb for emb in all_embeddings if str(emb["_id"]) == job_id), None)
            
            if matching_embedding:
                # Check if all 4 embedding vectors are present and have non-zero length
                if (len(matching_embedding.get("description_embedding", [])) > 0 and 
                    len(matching_embedding.get("requirements_embedding", [])) > 0 and 
                    len(matching_embedding.get("technical_skills_embedding", [])) > 0 and 
                    len(matching_embedding.get("soft_skills_embedding", [])) > 0):
                    valid_jobs.append(job)
        
        logger.info(f"Found {len(valid_jobs)} jobs with complete embeddings")
        
        # 4. Store job IDs for later reference
        self.job_ids = [str(job["_id"]) for job in valid_jobs]
        
        # 5. Store job metadata for each valid job
        for job in valid_jobs:
            job_id = str(job["_id"])
            self.job_metadata[job_id] = {
                "job_title": job.get("job_title", ""),
                "description": job.get("description", ""),
                "technical_skills": job.get("technical_skills", []),
                "soft_skills": job.get("soft_skills", [])
            }
        
        logger.info(f"Stored metadata for {len(self.job_metadata)} jobs")
        
        # 6. Get and store job vectors
        job_vectors = db_connector.get_job_vectors(self.job_ids)
        self.job_vectors = torch.stack(job_vectors).to(self.device)
        logger.info(f"Created job vectors tensor with shape {self.job_vectors.shape}")
    
    def get_applicant_state(self, applicant_id):
        """
        Get applicant state vector from cache.
        
        Args:
            applicant_id: ID of the applicant
            
        Returns:
            torch.Tensor: Applicant state vector
            
        Raises:
            KeyError: If applicant not in cache
        """
        if applicant_id in self.applicant_states:
            self.cache_hits += 1
            return self.applicant_states[applicant_id]
        else:
            self.cache_misses += 1
            raise KeyError(f"Applicant {applicant_id} not found in cache")
    
    def sample_jobs(self, n=100, exclude_ids=None):
        """
        Sample n random jobs from the cache.
        
        Args:
            n: Number of jobs to sample
            exclude_ids: Optional list of job IDs to exclude from sampling
            
        Returns:
            Tuple containing:
                - List[Dict]: List of job documents
                - torch.Tensor: Tensor of job vectors [n, embedding_dim]
                - List[str]: List of sampled job IDs
        """
        if not self.initialized:
            raise RuntimeError("Cache not initialized, call copy_from_database first")
        
        # Determine available indices
        available_indices = list(range(len(self.job_ids)))
        
        # Exclude specific job IDs if provided
        if exclude_ids:
            exclude_indices = [i for i, job_id in enumerate(self.job_ids) if job_id in exclude_ids]
            available_indices = [i for i in available_indices if i not in exclude_indices]
        
        # Sample n indices (or all if fewer than n are available)
        sample_size = min(n, len(available_indices))
        sampled_indices = random.sample(available_indices, sample_size)
        
        # Get vectors for sampled jobs
        sampled_vectors = self.job_vectors[sampled_indices]
        
        # Get job IDs for sampled jobs
        sampled_job_ids = [self.job_ids[i] for i in sampled_indices]
        
        # Get job documents for sampled jobs
        sampled_docs = []
        for job_id in sampled_job_ids:
            if job_id in self.job_metadata:
                # Create a document similar to what would be returned from the database
                metadata = self.job_metadata[job_id]
                doc = {
                    "_id": job_id,
                    "job_title": metadata["job_title"],
                    "description": metadata["description"],
                    "technical_skills": metadata.get("technical_skills", []),
                    "soft_skills": metadata.get("soft_skills", [])
                }
                sampled_docs.append(doc)
        
        return sampled_docs, sampled_vectors, sampled_job_ids
    
    def calculate_cosine_similarities(self, applicant_state):
        """
        Calculate cosine similarities between applicant state and all jobs.
        
        This is a vectorized implementation that computes similarities for all
        jobs in a single operation.
        
        Args:
            applicant_state: Applicant state tensor [embedding_dim]
            
        Returns:
            torch.Tensor: Cosine similarities for all jobs [num_jobs]
        """
        if not self.initialized:
            raise RuntimeError("Cache not initialized, call copy_from_database first")
        
        # Ensure applicant_state is on the correct device
        applicant_state = applicant_state.to(self.device)
        
        # Calculate cosine similarity for all jobs at once
        # Normalize vectors
        normalized_applicant = applicant_state / applicant_state.norm()
        normalized_jobs = self.job_vectors / self.job_vectors.norm(dim=1, keepdim=True)
        
        # Calculate dot product of normalized vectors (which equals cosine similarity)
        similarities = torch.matmul(normalized_jobs, normalized_applicant)
        
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
            idx = self.job_ids.index(job_id)
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
        
        return list(range(len(self.job_ids)))
    
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
        
        if index < 0 or index >= len(self.job_ids):
            raise IndexError(f"Job index {index} out of range (0-{len(self.job_ids)-1})")
        
        return self.job_ids[index]
    
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
        
        if index < 0 or index >= len(self.job_ids):
            raise IndexError(f"Job index {index} out of range (0-{len(self.job_ids)-1})")
        
        self.cache_hits += 1
        return self.job_vectors[index]
    
    def clear(self):
        """
        Clear all cached data to free memory.
        """
        logger.info("Clearing tensor cache...")
        
        # Clear applicant data
        self.applicant_states = {}
        
        # Clear job data
        if self.job_vectors is not None:
            self.job_vectors = None
        
        self.job_ids = []
        self.job_metadata = {}
        
        # Reset stats
        self.initialized = False
        
        logger.info("Tensor cache cleared successfully")
        
        return self
    
    def __len__(self):
        """Return the number of cached jobs."""
        return len(self.job_ids)
    
    def cache_stats(self):
        """Return cache statistics."""
        return {
            "job_count": len(self.job_ids),
            "applicant_count": len(self.applicant_states),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            "initialized": self.initialized,
            "initialization_time": self.initialization_time,
            "device": str(self.device),
        } 