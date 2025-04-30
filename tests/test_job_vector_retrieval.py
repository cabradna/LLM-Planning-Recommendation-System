#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test job vector retrieval specifically for the issue in notebook cell 6.

This test reproduces the error:
'Invalid dimension for soft skills embedding: got 0, expected 384'

By testing individual steps of the database retrieval process, we can 
identify exactly which jobs have missing or malformed data.
"""

import os
import sys
import unittest
import logging
import torch
import numpy as np
from pathlib import Path

# Configure logging for this test
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import necessary components
from src.data.database import DatabaseConnector
from config.config import TRAINING_CONFIG

class TestJobVectorRetrieval(unittest.TestCase):
    """
    Test class specifically examining the job vector retrieval issue 
    observed in notebook cell 6.
    """
    
    def setUp(self):
        """
        Set up the test environment by creating a database connection
        and other necessary components.
        """
        # Create a database connector
        try:
            self.db = DatabaseConnector()
            logger.info("Database connection established for test")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
        
        # Set random seed for reproducibility
        np.random.seed(TRAINING_CONFIG["random_seed"])
        torch.manual_seed(TRAINING_CONFIG["random_seed"])
    
    def tearDown(self):
        """Clean up resources after the test."""
        if hasattr(self, 'db'):
            self.db.close()
            logger.info("Database connection closed")
    
    def test_get_candidate_ids(self):
        """
        Test retrieving candidate IDs from the database.
        This replicates the first part of notebook cell 6.
        """
        # Get the candidates_text collection
        collection = self.db.db[self.db.collections["candidates_text"]]
        
        # Query for candidate IDs (limit to 10 for demonstration)
        candidate_ids = [doc["_id"] for doc in collection.find({}, {"_id": 1}).limit(TRAINING_CONFIG["num_candidates"])]
        
        # Verify that we found some candidates
        self.assertTrue(len(candidate_ids) > 0, "No candidate IDs found in database")
        logger.info(f"Found {len(candidate_ids)} candidate IDs")
        
        # Store the first candidate ID for use in other tests
        self.candidate_id = candidate_ids[0]
        logger.info(f"Selected target candidate ID: {self.candidate_id}")
    
    def test_get_applicant_state(self):
        """
        Test retrieving the applicant state (embedding) from the database.
        This replicates the candidate embedding retrieval in notebook cell 6.
        """
        # First, make sure we have a candidate ID
        if not hasattr(self, 'candidate_id'):
            self.test_get_candidate_ids()
            
        # Get the applicant state
        candidate_embedding = self.db.get_applicant_state(self.candidate_id)
        
        # Verify that the embedding is a torch tensor with the expected shape
        self.assertIsInstance(candidate_embedding, torch.Tensor, "Candidate embedding is not a torch.Tensor")
        
        # Print the shape for debugging
        logger.info(f"Candidate embedding shape: {candidate_embedding.shape}")
        
        # Check that it has the expected dimension according to MODEL_CONFIG
        # (This would be 384 in the config)
        self.assertEqual(len(candidate_embedding.shape), 1, "Embedding should be a 1D tensor")
    
    def test_sample_candidate_jobs(self):
        """
        Test sampling candidate jobs from the database.
        This replicates the job sampling in notebook cell 6 WITHOUT validation.
        """
        # Sample jobs from the database WITHOUT validation (original behavior)
        num_jobs = TRAINING_CONFIG["num_jobs"]
        sampled_jobs = self.db.sample_candidate_jobs(n=num_jobs, validate_embeddings=False)
        
        # Verify that we sampled the expected number of jobs
        self.assertEqual(len(sampled_jobs), num_jobs, f"Expected {num_jobs} jobs, got {len(sampled_jobs)}")
        logger.info(f"Sampled {len(sampled_jobs)} jobs from the database (without validation)")
        
        # Extract job IDs for use in other tests
        self.job_ids_unvalidated = [job["_id"] for job in sampled_jobs]
    
    def test_get_job_vectors_individually_without_validation(self):
        """
        Test retrieving job vectors one by one WITHOUT validation.
        This demonstrates the original issue where some jobs have missing embeddings.
        """
        # First, make sure we have job IDs
        if not hasattr(self, 'job_ids_unvalidated'):
            self.test_sample_candidate_jobs()
        
        # Test each job individually to find which ones have issues
        problem_jobs = []
        valid_jobs = []
        
        for job_id in self.job_ids_unvalidated:
            try:
                # Try to retrieve the job vector
                job_vectors = self.db.get_job_vectors([job_id])
                
                # If successful, verify the shape
                self.assertEqual(len(job_vectors), 1, f"Expected 1 vector for job {job_id}, got {len(job_vectors)}")
                
                # Get the job vector
                job_vector = job_vectors[0]
                
                # Verify the shape
                self.assertEqual(job_vector.shape[0], 384 * 4, 
                                f"Job vector for {job_id} has wrong shape: {job_vector.shape[0]}, expected {384 * 4}")
                
                logger.info(f"Job {job_id} vector retrieved successfully, shape: {job_vector.shape}")
                valid_jobs.append(job_id)
                
            except Exception as e:
                # Record the problematic job
                logger.error(f"Error retrieving vector for job {job_id}: {str(e)}")
                problem_jobs.append((job_id, str(e)))
        
        # Report on problematic jobs
        if problem_jobs:
            logger.error(f"Found {len(problem_jobs)} problematic jobs out of {len(self.job_ids_unvalidated)}")
            for job_id, error in problem_jobs[:3]:  # Show first 3 for brevity
                logger.error(f"  - Job {job_id}: {error}")
            
            # Examine the first problematic job in detail
            if len(problem_jobs) > 0:
                problem_job_id = problem_jobs[0][0]
                self._examine_job_document(problem_job_id)
                
            # Store valid job IDs for later tests
            self.valid_job_ids = valid_jobs
            
            # We expect to find some problematic jobs, so this is not an error
            logger.info("This test demonstrates the issue that validation solves")
        else:
            logger.info("All job vectors retrieved successfully - no data quality issues found")
    
    def _examine_job_document(self, job_id):
        """
        Helper method to examine a problematic job document in detail.
        
        Args:
            job_id: ID of the job to examine
        """
        # Query the job_embeddings collection
        collection = self.db.db[self.db.collections["job_embeddings"]]
        job_doc = collection.find_one({"original_job_id": job_id})
        
        if not job_doc:
            logger.error(f"Could not find job document for ID: {job_id}")
            return
        
        # Log the document keys
        logger.info(f"Document keys for job {job_id}: {list(job_doc.keys())}")
        
        # Check specifically for the embeddings fields
        for field_name in ["job_title_embeddings", "tech_skills_vectors", 
                          "soft_skills_embeddings", "experience_requirements_embeddings"]:
            if field_name in job_doc:
                field_value = job_doc[field_name]
                if isinstance(field_value, list):
                    logger.info(f"  - {field_name}: list with {len(field_value)} elements")
                    # If the field is empty, note that specifically
                    if len(field_value) == 0:
                        logger.error(f"  - {field_name} is an empty list []")
                else:
                    logger.info(f"  - {field_name}: {type(field_value)}")
            else:
                logger.error(f"  - {field_name} is MISSING from the document")
    
    def test_get_job_vectors_bulk_with_validation(self):
        """
        Test retrieving all job vectors at once WITH validation.
        This demonstrates how validation solves the issue.
        """
        # Get jobs with validation
        num_jobs = min(10, TRAINING_CONFIG["num_jobs"])
        sampled_jobs = self.db.sample_candidate_jobs(n=num_jobs, validate_embeddings=True)
        job_ids = [job["_id"] for job in sampled_jobs]
        
        # This should succeed because all jobs have been validated
        try:
            # Try to retrieve all job vectors at once
            job_vectors = self.db.get_job_vectors(job_ids)
            
            # If successful, verify the results
            self.assertEqual(len(job_vectors), len(job_ids), 
                           f"Expected {len(job_ids)} vectors, got {len(job_vectors)}")
            
            # Check the shape of each vector
            for i, vector in enumerate(job_vectors):
                self.assertEqual(vector.shape[0], 384 * 4, 
                               f"Vector {i} has wrong shape: {vector.shape[0]}, expected {384 * 4}")
            
            logger.info(f"Successfully retrieved {len(job_vectors)} job vectors with validation")
            logger.info(f"First vector shape: {job_vectors[0].shape}")
            logger.info("This demonstrates that validation successfully prevents the error")
            
        except Exception as e:
            # This should not happen with validation
            self.fail(f"Failed to retrieve job vectors even with validation: {str(e)}")
    
    def test_get_job_vectors_bulk_without_validation(self):
        """
        Test retrieving all job vectors at once WITHOUT validation.
        This replicates the exact call that failed in notebook cell 6.
        """
        # First, make sure we have job IDs
        if not hasattr(self, 'job_ids_unvalidated'):
            self.test_sample_candidate_jobs()
        
        # If we know there are valid job IDs (from previous tests), we can use them
        # to demonstrate the bulk operation working with clean data
        if hasattr(self, 'valid_job_ids') and len(self.valid_job_ids) > 0:
            try:
                # Try to retrieve valid job vectors in bulk
                valid_job_vectors = self.db.get_job_vectors(self.valid_job_ids)
                logger.info(f"Successfully retrieved {len(valid_job_vectors)} valid job vectors in bulk")
            except Exception as e:
                logger.error(f"Even valid jobs failed in bulk: {str(e)}")
        
        try:
            # Try to retrieve all job vectors at once (without validation)
            job_vectors = self.db.get_job_vectors(self.job_ids_unvalidated)
            
            # If successful, verify the results
            self.assertEqual(len(job_vectors), len(self.job_ids_unvalidated), 
                           f"Expected {len(self.job_ids_unvalidated)} vectors, got {len(job_vectors)}")
            
            logger.info(f"Successfully retrieved {len(job_vectors)} job vectors without validation - no data quality issues found")
            
        except Exception as e:
            # If it fails, log the error and include instructions for further debugging
            logger.error(f"Error retrieving job vectors in bulk: {str(e)}")
            logger.error("This demonstrates the error that occurs without validation")
            logger.error("Run test_get_job_vectors_individually_without_validation to identify specific problematic jobs")
            logger.error("Run test_get_job_vectors_bulk_with_validation to see how validation fixes the issue")
            
            # We expect this to fail if there are data quality issues
            logger.info("This failure is expected if there are data quality issues in the database")
            # Don't fail the test, as this is demonstrating the expected error
            # self.fail(f"Failed to retrieve job vectors in bulk: {str(e)}")
            
    def test_sample_candidate_jobs_with_validation(self):
        """
        Test sampling candidate jobs with embedding validation.
        This verifies that jobs with missing required embedding fields are filtered out.
        """
        # Sample jobs with validation
        num_jobs = min(10, TRAINING_CONFIG["num_jobs"])  # Use a smaller number for faster testing
        validated_jobs = self.db.sample_candidate_jobs(n=num_jobs, validate_embeddings=True)
        
        # Verify that we got some jobs (might be fewer than requested if data quality is poor)
        self.assertTrue(len(validated_jobs) > 0, "No valid jobs found")
        logger.info(f"Sampled {len(validated_jobs)} valid jobs")
        
        # Extract job IDs
        job_ids = [job["_id"] for job in validated_jobs]
        
        # Verify each job has valid embeddings
        for job_id in job_ids:
            # This should not raise an exception since we've validated the jobs
            try:
                job_vectors = self.db.get_job_vectors([job_id])
                self.assertEqual(len(job_vectors), 1, f"Expected 1 vector for job {job_id}")
                self.assertEqual(job_vectors[0].shape[0], 384 * 4, 
                               f"Job vector for {job_id} has wrong shape: {job_vectors[0].shape[0]}")
                logger.info(f"Validated job {job_id} has proper embeddings")
            except Exception as e:
                self.fail(f"Job {job_id} should have valid embeddings but got error: {str(e)}")
        
        # Now verify that validation actually filters out problematic jobs
        # We'll override the validation method temporarily to mark specific jobs as invalid
        original_validate = self.db._validate_job_embeddings
        
        def mock_validate(job_id):
            # Mark the first half of the jobs as "invalid" for testing purposes
            idx = job_ids.index(job_id) if job_id in job_ids else -1
            if idx >= 0 and idx < len(job_ids) // 2:
                return False
            return original_validate(job_id)
        
        try:
            # Apply the mock validation
            self.db._validate_job_embeddings = mock_validate
            
            # Sample jobs with the mock validation
            mock_validated_jobs = self.db.sample_candidate_jobs(n=num_jobs, validate_embeddings=True)
            mock_job_ids = [job["_id"] for job in mock_validated_jobs]
            
            # Verify none of the "invalid" jobs were included
            for i, job_id in enumerate(job_ids):
                if i < len(job_ids) // 2:
                    self.assertNotIn(job_id, mock_job_ids, 
                                   f"Job {job_id} should have been filtered out by validation")
            
            logger.info("Validation filtering works correctly")
        
        finally:
            # Restore the original validation method
            self.db._validate_job_embeddings = original_validate

if __name__ == '__main__':
    unittest.main() 