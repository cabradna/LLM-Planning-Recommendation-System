#!/usr/bin/env python
"""
Test runner script for the Dyna-Q job recommender neural model.
"""

import unittest
import os
import sys
import argparse

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_unittest_tests():
    """Run all unittest tests."""
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern="test_*.py")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)

def run_pytest_tests():
    """Run all pytest tests."""
    import pytest
    
    # Get the directory containing this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run pytest
    return pytest.main(["-xvs", test_dir])

def main():
    """Main function to run tests."""
    parser = argparse.ArgumentParser(description="Run tests for the Dyna-Q job recommender")
    parser.add_argument("--unittest", action="store_true", help="Run unittest tests")
    parser.add_argument("--pytest", action="store_true", help="Run pytest tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # If no arguments provided, run all tests
    if not (args.unittest or args.pytest or args.all):
        args.all = True
    
    # Run tests
    if args.unittest or args.all:
        print("\n========= Running unittest tests =========\n")
        results = run_unittest_tests()
        
        if not results.wasSuccessful():
            print("\n========= unittest tests failed =========\n")
            sys.exit(1)
    
    if args.pytest or args.all:
        try:
            import pytest
            print("\n========= Running pytest tests =========\n")
            result = run_pytest_tests()
            
            if result != 0:
                print("\n========= pytest tests failed =========\n")
                sys.exit(1)
        except ImportError:
            print("pytest not installed. Skipping pytest tests.")
    
    print("\n========= All tests passed! =========\n")

if __name__ == "__main__":
    main() 