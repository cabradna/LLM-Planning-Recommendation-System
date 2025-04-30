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
    
    def sample_candidate_jobs(self, n: int = 100, filter_criteria: Optional[Dict] = None) -> List[Dict]:
        """
        Sample a subset of jobs to be considered as candidate actions.
        
        Args:
            n: Number of candidate jobs to sample.
            filter_criteria: Optional criteria to filter jobs.
            
        Returns:
            List[Dict]: List of job documents.
        """
        try:
            # Define collection to query
            collection_name = self.collections["jobs_text"]
            collection = self.db[collection_name]
            
            # Apply filter if provided, otherwise get random sample
            query = filter_criteria or {}
            
            # Perform aggregation to get random sample
            pipeline = [
                {"$match": query},
                {"$sample": {"size": n}}
            ]
            
            jobs = list(collection.aggregate(pipeline))
            logger.info(f"Sampled {len(jobs)} candidate jobs from {collection_name}")
            return jobs
            
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
                
                # Extract and combine embeddings
                v_job_title = torch.tensor(emb_doc.get("job_title_embeddings", []))
                v_job_skills = torch.tensor(emb_doc.get("tech_skills_vectors", []))
                v_experience = torch.tensor(emb_doc.get("experience_requirements_vectors", []))
                v_soft_skills = torch.tensor(emb_doc.get("soft_skills_vectors", []))
                
                # Verify dimensions
                expected_dim = 384  # Each embedding should be 384-dimensional
                for name, vec in [
                    ("job title", v_job_title),
                    ("tech skills", v_job_skills),
                    ("experience", v_experience),
                    ("soft skills", v_soft_skills)
                ]:
                    if vec.shape[0] != expected_dim:
                        raise ValueError(f"Invalid dimension for {name} embedding: got {vec.shape[0]}, expected {expected_dim}")
                
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