# Test Data Generation

This document describes the approach to generating test data for the Dyna-Q Job Recommender Neural Model.

## Mock Data vs. Real Data

The test suite uses both mock data and synthetic data that mimics the structure of real data:

1. **Mock Data**: Using Python's `unittest.mock` library, we mock database connections, environment responses, and other external dependencies to isolate tests from actual infrastructure.

2. **Synthetic Data**: For integration and functional tests, we generate synthetic data that follows the structure of real production data.

## Database Setup for Testing

For MongoDB testing, the `init_test_db.py` script in the `scripts` directory creates a test database with the following collections:

- `candidates_embeddings`: Applicant profiles with skill embeddings
- `all_jobs`: All job postings
- `skilled_jobs`: Job postings with skill requirements
- `job_embeddings`: Vector representations of jobs

## Synthetic Data Generation

The main approaches to synthetic data generation include:

### 1. Random Tensor Generation

For neural network inputs, we generate random tensors with appropriate dimensions:

```python
# Generate random state vector
state = torch.randn(384)  # 384-dimensional state vector

# Generate random action vector
action = torch.randn(384)  # 384-dimensional action vector
```

### 2. Structured Mock Data

For MongoDB documents, we create structured mock data:

```python
# Example applicant document
applicant = {
    "candidate_id": f"applicant_{i}",
    "hard_skills_embedding": torch.rand(192).numpy().tolist(),
    "soft_skills_embedding": torch.rand(192).numpy().tolist(),
    "technical_skills": ["Python", "SQL", "Machine Learning"],
    "soft_skills": ["Communication", "Team Work"],
    "experience_years": 3,
    "education_level": "Master's"
}
```

### 3. LLM-Generated Data

For realistic text data like job descriptions or user feedback, we can use LLM-generated content (not implemented in the test suite but mentioned for completeness).

## Test Parameters

Tests use standardized parameters to ensure consistency:

- State dimension: 384
- Action dimension: 384
- Hidden layer dimensions: [256, 128]
- Batch size: 32
- Random seed: 42 