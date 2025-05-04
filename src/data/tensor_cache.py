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
        
        Filters for jobs that have all required embedding fields as specified in
        the database_structure.md documentation.
        
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
        
        # 3. Filter for jobs with complete embeddings (as per database_structure.md)
        # Required fields: job_title_embeddings, tech_skills_vectors, soft_skills_embeddings
        valid_embeddings = []
        valid_job_ids = []
        
        for emb in all_embeddings:
            # Check for required fields
            if (
                "job_title_embeddings" in emb and len(emb.get("job_title_embeddings", [])) > 0 and
                "tech_skills_vectors" in emb and len(emb.get("tech_skills_vectors", [])) > 0 and
                "soft_skills_embeddings" in emb and len(emb.get("soft_skills_embeddings", [])) > 0
            ):
                job_id = emb["original_job_id"]
                valid_embeddings.append(emb)
                valid_job_ids.append(str(job_id))
        
        logger.info(f"Found {len(valid_job_ids)} jobs with complete embeddings")
        
        # 4. Store job IDs for later reference
        self.job_ids = valid_job_ids
        
        # 5. Store job metadata for each valid job
        job_id_to_doc = {str(job["_id"]): job for job in all_jobs}
        
        for job_id in valid_job_ids:
            if job_id in job_id_to_doc:
                job_doc = job_id_to_doc[job_id]
                self.job_metadata[job_id] = {
                    "job_title": job_doc.get("job_title", ""),
                    "description": job_doc.get("description", ""),
                    "technical_skills": job_doc.get("technical_skills", []),
                    "soft_skills": job_doc.get("soft_skills", [])
                }
        
        logger.info(f"Stored metadata for {len(self.job_metadata)} jobs")
        
        # 6. Create mapping from job ID to embedding
        emb_by_id = {str(e["original_job_id"]): e for e in valid_embeddings}
        
        # 7. Determine embedding dimension from the data
        sample_emb = valid_embeddings[0]
        job_title_dim = len(sample_emb.get("job_title_embeddings", []))
        tech_skills_dim = len(sample_emb.get("tech_skills_vectors", []))
        soft_skills_dim = len(sample_emb.get("soft_skills_embeddings", []))
        
        total_dim = job_title_dim + tech_skills_dim + soft_skills_dim
        logger.info(f"Job vector dimension: {total_dim} (title: {job_title_dim}, tech: {tech_skills_dim}, soft: {soft_skills_dim})")
        
        # 8. Initialize the job vectors tensor
        self.job_vectors = torch.zeros((len(valid_job_ids), total_dim), device=self.device)
        
        # 9. Fill the job vectors tensor
        for idx, job_id in enumerate(valid_job_ids):
            if job_id in emb_by_id:
                emb = emb_by_id[job_id]
                
                # Get embedding components
                job_title_emb = torch.tensor(emb.get("job_title_embeddings", []), dtype=torch.float32)
                tech_skills_emb = torch.tensor(emb.get("tech_skills_vectors", []), dtype=torch.float32)
                soft_skills_emb = torch.tensor(emb.get("soft_skills_embeddings", []), dtype=torch.float32)
                
                # Concatenate embeddings
                combined_vector = torch.cat([job_title_emb, tech_skills_emb, soft_skills_emb])
                
                # Store in the tensor
                self.job_vectors[idx] = combined_vector.to(self.device)
        
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