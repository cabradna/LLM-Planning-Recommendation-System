"""
Tests for the database connector.
"""

import unittest
import sys
import os
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data.database import DatabaseConnector

class TestDatabaseConnector(unittest.TestCase):
    """Test cases for the DatabaseConnector class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock pymongo client
        self.mock_client = MagicMock()
        self.mock_db = MagicMock()
        self.mock_collections = MagicMock()
        
        # Configure the mock to return data
        self.mock_client.__getitem__.return_value = self.mock_db
        self.mock_db.__getitem__.side_effect = self.mock_get_collection
        
        # Create test data
        self.test_applicant_id = "test_applicant_1"
        self.test_applicant_embedding = torch.rand(384)
        
        self.test_jobs = [
            {"original_job_id": f"job_{i}", "job_title": f"Job {i}"} for i in range(10)
        ]
        
        self.test_job_embeddings = [
            {
                "original_job_id": f"job_{i}",
                "job_title_embeddings": torch.rand(192).numpy().tolist(),
                "tech_skills_vectors": torch.rand(192).numpy().tolist()
            }
            for i in range(10)
        ]
        
        # Set up collection return values
        self.collections = {
            "candidates_embeddings": MagicMock(),
            "job_embeddings": MagicMock(),
            "all_jobs": MagicMock(),
            "skilled_jobs": MagicMock()
        }
        
        # Configure mock collections
        self.collections["candidates_embeddings"].find_one.return_value = {
            "candidate_id": self.test_applicant_id,
            "hard_skills_embedding": self.test_applicant_embedding[:192].numpy().tolist(),
            "soft_skills_embedding": self.test_applicant_embedding[192:].numpy().tolist()
        }
        
        self.collections["skilled_jobs"].aggregate.return_value = self.test_jobs
        
        # Configure job_embeddings find
        def mock_job_embeddings_find(query):
            job_ids = query.get("original_job_id", {}).get("$in", [])
            return [emb for emb in self.test_job_embeddings if emb["original_job_id"] in job_ids]
        
        self.collections["job_embeddings"].find.side_effect = mock_job_embeddings_find
        
        self.collections["all_jobs"].find_one.return_value = self.test_jobs[0]
        
        # Patch pymongo.MongoClient
        self.patcher = patch('pymongo.MongoClient', return_value=self.mock_client)
        self.mock_mongo_client = self.patcher.start()
    
    def tearDown(self):
        """Tear down test environment."""
        self.patcher.stop()
    
    def mock_get_collection(self, collection_name):
        """Mock method to get a collection."""
        return self.collections.get(collection_name, MagicMock())
    
    def test_init_and_connect(self):
        """Test initialization and connection."""
        # Create connector
        connector = DatabaseConnector()
        
        # Check that client and db are set
        self.assertEqual(connector.client, self.mock_client)
        self.assertEqual(connector.db, self.mock_db)
    
    def test_close(self):
        """Test closing the connection."""
        # Create connector
        connector = DatabaseConnector()
        
        # Close connection
        connector.close()
        
        # Check that client.close was called
        self.mock_client.close.assert_called_once()
    
    def test_get_applicant_state(self):
        """Test getting applicant state."""
        # Create connector
        connector = DatabaseConnector()
        
        # Get applicant state
        state = connector.get_applicant_state(self.test_applicant_id)
        
        # Check that find_one was called with correct args
        self.collections["candidates_embeddings"].find_one.assert_called_once_with({
            "candidate_id": self.test_applicant_id
        })
        
        # Check state
        self.assertIsInstance(state, torch.Tensor)
        self.assertEqual(state.shape, torch.Size([384]))
    
    def test_sample_candidate_jobs(self):
        """Test sampling candidate jobs."""
        # Create connector
        connector = DatabaseConnector()
        
        # Sample jobs
        jobs = connector.sample_candidate_jobs(n=10)
        
        # Check that aggregate was called with correct args
        self.collections["skilled_jobs"].aggregate.assert_called_once()
        
        # Check jobs
        self.assertEqual(len(jobs), len(self.test_jobs))
        self.assertEqual(jobs, self.test_jobs)
    
    def test_get_job_vectors(self):
        """Test getting job vectors."""
        # Create connector
        connector = DatabaseConnector()
        
        # Get job IDs
        job_ids = [job["original_job_id"] for job in self.test_jobs]
        
        # Get job vectors
        job_vectors = connector.get_job_vectors(job_ids)
        
        # Check that find was called
        self.collections["job_embeddings"].find.assert_called_once()
        
        # Check job vectors
        self.assertEqual(len(job_vectors), len(job_ids))
        for vector in job_vectors:
            self.assertIsInstance(vector, torch.Tensor)
            self.assertEqual(vector.shape, torch.Size([384]))  # 192 + 192
    
    def test_get_job_details(self):
        """Test getting job details."""
        # Create connector
        connector = DatabaseConnector()
        
        # Get job details
        job = connector.get_job_details("job_1")
        
        # Check that find_one was called with correct args
        self.collections["all_jobs"].find_one.assert_called_once_with({
            "original_job_id": "job_1"
        })
        
        # Check job
        self.assertEqual(job, self.test_jobs[0])

if __name__ == '__main__':
    unittest.main() 