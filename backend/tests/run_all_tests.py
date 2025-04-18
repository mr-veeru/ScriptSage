#!/usr/bin/env python
"""
Unified test runner for ScriptSage backend tests
Run this script to execute all tests in the test suite
"""

import os
import sys
import unittest
from importlib import util

# Add the parent directory to the path so we can import from the backend modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_all_tests():
    """Run all tests in the tests directory"""
    print("=== Running ScriptSage Backend Tests ===")
    print("Loading test modules...")
    
    # Discover and run all test_*.py files in the current directory
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(__file__), pattern="test_*.py")
    
    print(f"Discovered {test_suite.countTestCases()} tests in {len(list(test_suite))} modules")
    
    # Run the tests
    print("\n--- Running Tests ---")
    result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    # Run specific test files directly if they're not using unittest
    test_files = [
        "test_ast.py",
        "test_api.py",
        "test_ml.py"
    ]
    
    for test_file in test_files:
        if not os.path.exists(os.path.join(os.path.dirname(__file__), test_file)):
            continue
            
        print(f"\n--- Running {test_file} directly ---")
        
        # Check if the file has an appropriate test function
        module_name = test_file[:-3]  # remove .py
        spec = util.spec_from_file_location(module_name, os.path.join(os.path.dirname(__file__), test_file))
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Call test functions if they exist
        if hasattr(module, "run_tests"):
            module.run_tests()
        
    print("\n=== Test Run Complete ===")
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 