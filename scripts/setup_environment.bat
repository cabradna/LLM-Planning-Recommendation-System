@echo off
REM Setup script for Dyna-Q Job Recommender Neural Model environment (Windows)

echo =====================================
echo Dyna-Q Job Recommender Setup
echo =====================================

REM Check Python version
echo Checking Python version...
python --version

REM Create and activate virtual environment
echo Setting up virtual environment...
if exist venv\ (
    echo Virtual environment already exists
) else (
    python -m venv venv
    echo Created virtual environment: venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Activated virtual environment

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Install package in development mode
echo Installing package in development mode...
pip install -e .

REM Check if MongoDB is running
echo Checking MongoDB...
where mongod >nul 2>nul
if %ERRORLEVEL% == 0 (
    echo MongoDB is installed
    
    REM Try to connect to MongoDB
    where mongosh >nul 2>nul
    if %ERRORLEVEL% == 0 (
        mongosh --eval "db.version()" --quiet >nul 2>nul
        if %ERRORLEVEL% == 0 (
            echo MongoDB is running
        ) else (
            echo WARNING: MongoDB is installed but not running
            echo Please start MongoDB service before running the application
        )
    ) else (
        echo WARNING: MongoDB shell (mongosh) not found
        echo Cannot verify if MongoDB is running
    )
) else (
    echo WARNING: MongoDB is not installed or not in PATH
    echo Please install MongoDB from https://www.mongodb.com/try/download/community
)

REM Initialize test database
echo Initializing test database...
python scripts\init_test_db.py

REM Run tests
echo Running tests...
python tests\run_tests.py

echo =====================================
echo Setup complete!
echo =====================================
echo Next steps:
echo 1. Explore available commands: python src\main.py info
echo 2. Run pretraining: python src\main.py pretraining --generate_data
echo 3. Train models: python src\main.py train --train_baseline --num_episodes 100
echo =====================================

REM Keep the window open
pause 