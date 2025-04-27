# Dyna-Q Job Recommender Test Suite

This directory contains the test suite for the Dyna-Q Job Recommender Neural Model.

## Test Organization

The tests are organized as follows:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Mock Tests**: Use mock objects to simulate dependencies

## Running Tests

You can run the tests using the `run_tests.py` script:

```bash
# Run all tests
python run_tests.py

# Run only unittest tests
python run_tests.py --unittest

# Run only pytest tests
python run_tests.py --pytest
```

## Test Files

- `test_q_network.py`: Tests for the Q-Network model
- `test_world_model.py`: Tests for the World Model
- `test_dyna_q_agent.py`: Tests for the DynaQ Agent
- `test_job_env.py`: Tests for the job recommendation environment
- `test_data_loader.py`: Tests for the data loader module
- `test_database.py`: Tests for the database connector
- `test_pretraining.py`: Tests for the pretraining module
- `test_evaluate.py`: Tests for the evaluation module
- `test_integration.py`: Integration tests for the complete system

## Test Configuration

- `conftest.py`: Pytest configuration with fixtures
- `test_data.md`: Documentation about test data generation

## Prerequisites

Before running the tests, ensure:

1. You have installed all requirements:
   ```bash
   pip install -r ../requirements.txt
   ```

2. MongoDB is installed and running (for database-related tests):
   ```bash
   # Initialize test database
   python ../scripts/init_test_db.py
   ```

## Test Data

The tests use a combination of:

- Generated random data (for neural network inputs)
- Mock objects (to isolate from external dependencies)
- Synthetic structured data (for MongoDB documents)

See `test_data.md` for more information about the test data generation approach. 