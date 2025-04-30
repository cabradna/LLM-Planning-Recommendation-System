"""
MongoDB database connector for the Dyna-Q Job Recommender.
Handles connections and queries to the MongoDB Atlas database.
"""

import pymongo
import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
import os

# Import configuration
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config.config import DB_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseConnector:
    """
    Handles connections and queries to the MongoDB Atlas database.
    
    This class provides methods to:
    1. Connect to the MongoDB Atlas database
    2. Retrieve candidate (applicant) embeddings
    3. Retrieve job embeddings
    4. Sample candidate jobs for the action space
    """
    
    def __init__(self, connection_string: Optional[str] = None, db_name: Optional[str] = None):
        """
        Initialize the database connector.
        
        Args:
            connection_string: MongoDB connection string. If None, uses the value from config.
            db_name: Database name. If None, uses the value from config.
        """
        self.connection_string = connection_string or self._get_connection_string()
        self.db_name = db_name or DB_CONFIG["database"]
        self.collections = DB_CONFIG["collections"]
        self.client = None
        self.db = None
        self._connect()
    
    def _get_connection_string(self) -> str:
        """
        Get the MongoDB connection string with authentication.
        
        Returns:
            str: Complete MongoDB connection string with authentication.
        """
        try:
            # Get the path to the auth file
            auth_file = DB_CONFIG["auth_file"]
            if not os.path.exists(auth_file):
                raise FileNotFoundError(f"Auth file not found at {auth_file}")
            
            # Read credentials from file
            with open(auth_file, 'r') as f:
                mongo_user = f.readline().strip()
                mongo_password = f.readline().strip()
            
            # Construct connection string
            base_uri = DB_CONFIG["connection_string"]
            if base_uri.startswith("mongodb+srv://"):
                # For Atlas connection
                return f"mongodb+srv://{mongo_user}:{mongo_password}@cluster0.zqzq6hs.mongodb.net/"
            else:
                # For local connection
                return f"mongodb://{mongo_user}:{mongo_password}@{base_uri.split('://')[1]}"
                
        except Exception as e:
            logger.error(f"Error getting connection string: {e}")
            raise
    
    def _connect(self) -> None:
        """
        Establish connection to MongoDB.
        
        Raises:
            ConnectionError: If connection to MongoDB fails.
        """
        try:
            self.client = pymongo.MongoClient(self.connection_string)
            self.db = self.client[self.db_name]
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB Atlas database: {self.db_name}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise ConnectionError(f"Could not connect to MongoDB: {e}")
    
    def close(self) -> None:
        """Close the database connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def get_applicant_state(self, applicant_id: str) -> torch.Tensor:
        """
        Retrieve applicant embeddings and construct the state vector.
        
        Args:
            applicant_id: ID of the applicant.
            
        Returns:
            torch.Tensor: State vector representing the applicant.
            
        Raises:
            ValueError: If applicant embeddings not found.
        """
        try:
            # Query the candidates_embeddings collection
            collection = self.db[self.collections["candidates_embeddings"]]
            candidate_data = collection.find_one({"original_candidate_id": applicant_id})
            
            if not candidate_data:
                raise ValueError(f"No embeddings found for applicant ID: {applicant_id}")
            
            # Extract and combine embeddings
            hard_skills_embedding = torch.tensor(candidate_data.get("hard_skill_embeddings", []))
            soft_skills_embedding = torch.tensor(candidate_data.get("soft_skills_embeddings", []))
            
            # Concatenate embeddings
            applicant_state = torch.cat([
                hard_skills_embedding, 
                soft_skills_embedding
            ], dim=0)
            
            return applicant_state
            
        except Exception as e:
            logger.error(f"Error retrieving applicant state: {e}")
            raise
    
    def _validate_job_embeddings(self, job_id: str) -> bool:
        """
        Check if a job has all the required embedding fields.
        
        Args:
            job_id: ID of the job to validate
            
        Returns:
            bool: True if the job has all required embeddings, False otherwise
        """
        try:
            # Query the job_embeddings collection
            collection = self.db[self.collections["job_embeddings"]]
            job_doc = collection.find_one({"original_job_id": job_id})
            
            if not job_doc:
                logger.warning(f"No embedding document found for job ID: {job_id}")
                return False
            
            # Check for required fields according to documentation
            required_fields = ["job_title_embeddings", "tech_skills_vectors", "soft_skills_embeddings"]
            
            for field in required_fields:
                if field not in job_doc or not job_doc[field] or len(job_doc[field]) == 0:
                    logger.warning(f"Job {job_id} missing required field: {field}")
                    return False
            
            # All required fields are present
            return True
            
        except Exception as e:
            logger.error(f"Error validating job embeddings for {job_id}: {e}")
            return False
            
    def sample_candidate_jobs(self, n: int = 100, filter_criteria: Optional[Dict] = None, 
                              validate_embeddings: bool = False) -> List[Dict]:
        """
        Sample a subset of jobs to be considered as candidate actions.
        
        Args:
            n: Number of candidate jobs to sample.
            filter_criteria: Optional criteria to filter jobs.
            validate_embeddings: If True, verify sampled jobs have all required embedding fields.
            
        Returns:
            List[Dict]: List of job documents with valid embeddings.
        """
        try:
            # Define collection to query
            collection_name = self.collections["jobs_text"]
            collection = self.db[collection_name]
            
            # Apply filter if provided, otherwise get random sample
            query = filter_criteria or {}
            
            # If validation is requested, sample more jobs than needed to account for potential invalid ones
            sample_size = n * 2 if validate_embeddings else n
            
            # Perform aggregation to get random sample
            pipeline = [
                {"$match": query},
                {"$sample": {"size": sample_size}}
            ]
            
            sampled_jobs = list(collection.aggregate(pipeline))
            
            if validate_embeddings:
                # Filter to jobs with valid embeddings
                valid_jobs = []
                for job in sampled_jobs:
                    if self._validate_job_embeddings(job["_id"]):
                        valid_jobs.append(job)
                        if len(valid_jobs) >= n:
                            break
                
                if len(valid_jobs) < n:
                    logger.warning(f"Only found {len(valid_jobs)} valid jobs out of {len(sampled_jobs)} sampled. "
                                  f"Requested {n} jobs.")
                    
                sampled_jobs = valid_jobs
            
            logger.info(f"Sampled {len(sampled_jobs)} candidate jobs from {collection_name}")
            return sampled_jobs
            
        except Exception as e:
            logger.error(f"Error sampling candidate jobs: {e}")
            raise
    
    def get_job_vectors(self, job_ids: List[str]) -> List[torch.Tensor]:
        """
        Retrieve job embeddings for a list of job IDs.
        
        Args:
            job_ids: List of job IDs.
            
        Returns:
            List[torch.Tensor]: List of job embedding vectors.
        """
        try:
            # Query the job_embeddings collection
            collection = self.db[self.collections["job_embeddings"]]
            job_embeddings_cursor = collection.find({"original_job_id": {"$in": job_ids}})
            
            # Process the embeddings
            job_vectors = []
            job_id_to_idx = {job_id: idx for idx, job_id in enumerate(job_ids)}
            retrieved_ids = set()
            
            for emb_doc in job_embeddings_cursor:
                job_id = emb_doc["original_job_id"]
                retrieved_ids.add(job_id)
                
                # Extract embeddings from the document
                v_job_title = torch.tensor(emb_doc.get("job_title_embeddings", []), dtype=torch.float)
                v_job_skills = torch.tensor(emb_doc.get("tech_skills_vectors", []), dtype=torch.float)
                v_soft_skills = torch.tensor(emb_doc.get("soft_skills_embeddings", []), dtype=torch.float)
                
                # Handle potentially missing experience_requirements_embeddings
                if "experience_requirements_embeddings" in emb_doc and len(emb_doc["experience_requirements_embeddings"]) > 0:
                    v_experience = torch.tensor(emb_doc["experience_requirements_embeddings"], dtype=torch.float)
                else:
                    # Fallback: use zero vector of expected dimension
                    logger.warning(f"Missing experience_requirements_embeddings for job {job_id}. Using zero vector.")
                    v_experience = torch.zeros(384, dtype=torch.float)
                
                # Ensure each vector has correct dimensions (384) for consistent concatenation
                # If vectors are non-empty but incorrect dimension, this would be a data quality issue
                if len(v_job_title) > 0 and v_job_title.shape[0] != 384:
                    raise ValueError(f"Job title embedding dimension mismatch for job {job_id}: {v_job_title.shape[0]} vs expected 384")
                    
                if len(v_job_skills) > 0 and v_job_skills.shape[0] != 384:
                    raise ValueError(f"Tech skills embedding dimension mismatch for job {job_id}: {v_job_skills.shape[0]} vs expected 384")
                    
                if len(v_soft_skills) > 0 and v_soft_skills.shape[0] != 384:
                    raise ValueError(f"Soft skills embedding dimension mismatch for job {job_id}: {v_soft_skills.shape[0]} vs expected 384")
                
                # If vectors are empty but required, this is a critical issue
                if len(v_job_title) == 0:
                    raise ValueError(f"Missing required job_title_embeddings for job {job_id}")
                    
                if len(v_job_skills) == 0:
                    raise ValueError(f"Missing required tech_skills_vectors for job {job_id}")
                    
                if len(v_soft_skills) == 0:
                    raise ValueError(f"Missing required soft_skills_embeddings for job {job_id}")
                
                # Combine into a single vector
                v_job = torch.cat([
                    v_job_title,
                    v_job_skills,
                    v_experience,
                    v_soft_skills
                ], dim=0)
                
                # Verify final dimension
                if v_job.shape[0] != 384 * 4:  # 4 embeddings of 384 dimensions each
                    raise ValueError(f"Invalid final job vector dimension: got {v_job.shape[0]}, expected {384 * 4}")
                
                job_vectors.append((job_id_to_idx[job_id], v_job))
            
            # Check if all job IDs were retrieved
            missing_ids = set(job_ids) - retrieved_ids
            if missing_ids:
                logger.warning(f"Could not find embeddings for {len(missing_ids)} job IDs")
            
            # Sort by the original index and extract just the tensors
            job_vectors.sort(key=lambda x: x[0])
            return [v for _, v in job_vectors]
            
        except Exception as e:
            logger.error(f"Error retrieving job vectors: {e}")
            raise
    
    def get_job_details(self, job_id: str) -> Dict:
        """
        Retrieve detailed information about a job.
        
        Args:
            job_id: Job ID.
            
        Returns:
            Dict: Job details.
        """
        try:
            collection = self.db[self.collections["jobs_text"]]
            job = collection.find_one({"_id": job_id})
            
            if not job:
                raise ValueError(f"No job found with ID: {job_id}")
                
            return job
            
        except Exception as e:
            logger.error(f"Error retrieving job details: {e}")
            raise 