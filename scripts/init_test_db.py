#!/usr/bin/env python
"""
Initialize a test MongoDB database with dummy data for testing.

This script creates a local MongoDB database with test collections and data
that can be used for testing the Dyna-Q job recommender neural model.
"""

import os
import sys
import torch
import numpy as np
import json
import argparse
from typing import List, Dict, Any

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_dummy_applicants(num_applicants: int = 20) -> List[Dict[str, Any]]:
    """
    Create dummy applicant data.
    
    Args:
        num_applicants: Number of applicants to create.
        
    Returns:
        List of applicant documents.
    """
    applicants_text = []
    applicants_embeddings = []
    
    # Define skill pools
    technical_skills = [
        "Python", "Java", "C++", "JavaScript", "SQL", "Machine Learning",
        "Data Analysis", "Cloud Computing", "DevOps", "Web Development"
    ]
    
    soft_skills = [
        "Communication", "Team Work", "Problem Solving", "Leadership",
        "Time Management", "Adaptability", "Critical Thinking"
    ]
    
    for i in range(num_applicants):
        # Generate random embeddings
        hard_skill_embeddings = torch.rand(192).numpy().tolist()
        soft_skill_embeddings = torch.rand(192).numpy().tolist()
        
        # Pick random skills
        num_tech_skills = np.random.randint(2, 6)
        num_soft_skills = np.random.randint(2, 5)
        
        tech_skills = np.random.choice(technical_skills, size=num_tech_skills, replace=False).tolist()
        soft_skills = np.random.choice(soft_skills, size=num_soft_skills, replace=False).tolist()
        
        # Create candidate text document
        candidate_id = f"applicant_{i}"
        candidate_text = {
            "_id": candidate_id,
            "bio": f"Experienced professional with background in {', '.join(tech_skills[:2])}",
            "skills": tech_skills + soft_skills,
            "experience": [
                {
                    "title": f"{np.random.choice(['Junior', 'Senior', 'Lead'])} Developer",
                    "years": np.random.randint(1, 5)
                }
            ]
        }
        
        # Create embedding document
        candidate_embedding = {
            "_id": f"emb_{candidate_id}",
            "original_candidate_id": candidate_id,
            "hard_skill_embeddings": hard_skill_embeddings,
            "soft_skill_embeddings": soft_skill_embeddings
        }
        
        applicants_text.append(candidate_text)
        applicants_embeddings.append(candidate_embedding)
    
    return applicants_text, applicants_embeddings

def create_dummy_jobs(num_jobs: int = 100) -> List[Dict[str, Any]]:
    """
    Create dummy job data.
    
    Args:
        num_jobs: Number of jobs to create.
        
    Returns:
        List of job documents.
    """
    jobs = []
    
    # Define job titles and descriptions
    job_titles = [
        "Software Engineer", "Data Scientist", "Product Manager",
        "DevOps Engineer", "UI/UX Designer", "Data Analyst",
        "Machine Learning Engineer", "Full Stack Developer",
        "QA Engineer", "Project Manager"
    ]
    
    job_descriptions = [
        "Developing software applications and systems",
        "Analyzing large datasets to extract insights",
        "Managing product development lifecycle",
        "Building and maintaining CI/CD pipelines",
        "Creating user-friendly interfaces and experiences",
        "Processing and interpreting complex data",
        "Building and deploying machine learning models",
        "Developing both frontend and backend components",
        "Ensuring software quality through testing",
        "Leading project teams to successful delivery"
    ]
    
    for i in range(num_jobs):
        # Select random title and description
        title_idx = np.random.randint(0, len(job_titles))
        
        # Create document
        job = {
            "_id": f"job_{i}",
            "doc_id": f"doc_{i}",
            "source_file": f"source_{np.random.randint(1, 5)}.csv",
            "original_index": i,
            "job_title": f"{job_titles[title_idx]} {i}",
            "description": job_descriptions[title_idx],
            "metadata": {
                "location": np.random.choice(["New York", "San Francisco", "Seattle", "Boston", "Austin"]),
                "company": f"Company {np.random.randint(1, 20)}",
                "salary_range": f"${np.random.randint(70, 150)}K - ${np.random.randint(150, 200)}K"
            },
            "technical_skills": np.random.choice(["Python", "Java", "SQL", "JavaScript", "Cloud"], 
                                               size=np.random.randint(2, 5), 
                                               replace=False).tolist(),
            "soft_skills": np.random.choice(["Communication", "Team Work", "Problem Solving"], 
                                          size=np.random.randint(1, 3), 
                                          replace=False).tolist(),
            "experience_requirements": [f"{np.random.randint(1, 5)} years of experience"]
        }
        
        jobs.append(job)
    
    return jobs

def create_dummy_job_embeddings(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create dummy job embeddings.
    
    Args:
        jobs: List of job documents.
        
    Returns:
        List of job embedding documents.
    """
    job_embeddings = []
    
    for job in jobs:
        # Generate random embeddings
        job_title_embeddings = torch.rand(192).numpy().tolist()
        tech_skills_vectors = torch.rand(192).numpy().tolist()
        soft_skills_embeddings = torch.rand(192).numpy().tolist()
        
        # Create document
        embedding = {
            "_id": f"emb_{job['_id']}",
            "original_job_id": job["_id"],
            "job_title_embeddings": job_title_embeddings,
            "tech_skills_vectors": tech_skills_vectors,
            "soft_skills_embeddings": soft_skills_embeddings
        }
        
        job_embeddings.append(embedding)
    
    return job_embeddings

def initialize_mongo_db(
    uri: str = "mongodb://localhost:27017",
    db_name: str = "test_rl_jobsdb",
    drop_existing: bool = True
) -> None:
    """
    Initialize MongoDB with test data.
    
    Args:
        uri: MongoDB connection URI.
        db_name: Database name.
        drop_existing: Whether to drop existing database.
    """
    try:
        import pymongo
        client = pymongo.MongoClient(uri)
        
        # Drop existing database if requested
        if drop_existing and db_name in client.list_database_names():
            client.drop_database(db_name)
            print(f"Dropped existing database: {db_name}")
        
        # Create database
        db = client[db_name]
        
        # Create collections
        candidates_text = db["candidates_text"]
        candidates_embeddings = db["candidates_embeddings"]
        jobs_text = db["jobs_text"]
        job_embeddings = db["job_embeddings"]
        
        # Create dummy data
        applicants_text, applicants_embeddings = create_dummy_applicants(num_applicants=20)
        jobs = create_dummy_jobs(num_jobs=100)
        job_embs = create_dummy_job_embeddings(jobs)
        
        # Insert data
        if applicants_text:
            candidates_text.insert_many(applicants_text)
            candidates_embeddings.insert_many(applicants_embeddings)
            print(f"Inserted {len(applicants_text)} applicant documents and their embeddings")
        
        if jobs:
            jobs_text.insert_many(jobs)
            print(f"Inserted {len(jobs)} job documents")
        
        if job_embs:
            job_embeddings.insert_many(job_embs)
            print(f"Inserted {len(job_embs)} job embedding documents")
        
        # Create indexes
        candidates_text.create_index("_id")
        candidates_embeddings.create_index("original_candidate_id")
        jobs_text.create_index("_id")
        job_embeddings.create_index("original_job_id")
        
        print(f"Successfully initialized test database: {db_name}")
        
    except Exception as e:
        print(f"Error initializing MongoDB: {e}")
        raise

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Initialize a test MongoDB database")
    parser.add_argument("--uri", type=str, default="mongodb://localhost:27017",
                        help="MongoDB connection URI")
    parser.add_argument("--db-name", type=str, default="test_rl_jobsdb",
                        help="Database name")
    parser.add_argument("--no-drop", action="store_true",
                        help="Don't drop existing database")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Initialize MongoDB
    initialize_mongo_db(
        uri=args.uri,
        db_name=args.db_name,
        drop_existing=not args.no_drop
    )

if __name__ == "__main__":
    main() 