#!/bin/bash
# Setup script for Dyna-Q Job Recommender Neural Model environment

# Display header
echo "====================================="
echo "Dyna-Q Job Recommender Setup"
echo "====================================="

# Check Python version
echo "Checking Python version..."
python --version

# Create and activate virtual environment
echo "Setting up virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists"
else
    python -m venv venv
    echo "Created virtual environment: venv"
fi

# Activate virtual environment
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        # Windows
        echo "Detected Windows OS"
        source venv/Scripts/activate
        ;;
    *)
        # Unix-like (Linux, macOS)
        source venv/bin/activate
        ;;
esac

echo "Activated virtual environment"

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install package in development mode
echo "Installing package in development mode..."
pip install -e .

# Check if MongoDB is running
echo "Checking MongoDB..."
if command -v mongod &> /dev/null; then
    echo "MongoDB is installed"
    
    # Try to connect to MongoDB
    if command -v mongosh &> /dev/null; then
        if mongosh --eval "db.version()" --quiet &> /dev/null; then
            echo "MongoDB is running"
        else
            echo "WARNING: MongoDB is installed but not running"
            echo "Please start MongoDB service before running the application"
        fi
    else
        echo "WARNING: MongoDB shell (mongosh) not found"
        echo "Cannot verify if MongoDB is running"
    fi
else
    echo "WARNING: MongoDB is not installed or not in PATH"
    echo "Please install MongoDB from https://www.mongodb.com/try/download/community"
fi

# Initialize test database
echo "Initializing test database..."
python scripts/init_test_db.py

# Run tests
echo "Running tests..."
python tests/run_tests.py

echo "====================================="
echo "Setup complete!"
echo "====================================="
echo "Next steps:"
echo "1. Explore available commands: python src/main.py info"
echo "2. Run pretraining: python src/main.py pretraining --generate_data"
echo "3. Train models: python src/main.py train --train_baseline --num_episodes 100"
echo "=====================================" 