import sys
import os
import json
import requests

# Add the parent directory to sys.path so we can import backend modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Base URL for the API
BASE_URL = "http://localhost:5000"

# Test Python code
python_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

class MathUtils:
    def __init__(self):
        self.cache = {}
    
    def calculate(self, n):
        if n in self.cache:
            return self.cache[n]
        result = factorial(n)
        self.cache[n] = result
        return result

from math import sqrt
import os, sys

# Example usage
utils = MathUtils()
print(utils.calculate(5))
"""

def analyze_code(code, filename=None):
    """Send code to the API for analysis"""
    url = f"{BASE_URL}/api/analyze"
    
    data = {
        "code": code
    }
    
    if filename:
        data["filename"] = filename
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        return None

def pretty_print(data):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=2))

def run_tests():
    """Run API tests"""
    print("=== Testing API Integration ===")

    # Test Python code analysis
    print("\n--- Python Code Analysis ---")
    result = analyze_code(python_code, "example.py")

    if result:
        print(f"Language: {result.get('language')}")
        print(f"Type: {result.get('analysis', {}).get('type')}")
        print(f"Confidence: {result.get('analysis', {}).get('confidence')}")
        
        # Print AST analysis if available
        ast_analysis = result.get('analysis', {}).get('ast_analysis')
        if ast_analysis:
            print("\nAST Analysis:")
            print(f"Functions: {ast_analysis.get('structure', {}).get('functions')}")
            print(f"Classes: {ast_analysis.get('structure', {}).get('classes')}")
            print(f"Imports: {ast_analysis.get('structure', {}).get('imports')}")
            
            complexity = ast_analysis.get('complexity', {})
            if complexity:
                print("\nComplexity Metrics:")
                print(f"Node Count: {complexity.get('node_count')}")
                print(f"Cyclomatic Complexity: {complexity.get('cyclomatic_complexity')}")
        else:
            print("\nAST analysis not available")
        
        return True
    else:
        print("Failed to get analysis results from API")
        return False

if __name__ == "__main__":
    run_tests() 