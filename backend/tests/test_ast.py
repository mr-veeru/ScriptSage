import sys
import os
import json

# Add the parent directory to sys.path so we can import backend modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Updated import paths
from core.ast_analyzer import parse_code_structure, calculate_complexity_metrics

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

# Test JavaScript code (for basic analysis)
js_code = """
function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n-1);
}

class MathUtils {
    constructor() {
        this.cache = {};
    }
    
    calculate(n) {
        if (this.cache[n]) return this.cache[n];
        const result = factorial(n);
        this.cache[n] = result;
        return result;
    }
}

// Example usage
const utils = new MathUtils();
console.log(utils.calculate(5));
"""

def pretty_print(data):
    """Pretty print a dictionary"""
    print(json.dumps(data, indent=2))

def run_tests():
    """Run AST analyzer tests"""
    print("=== Testing AST analyzer ===")

    # Test Python code analysis
    print("\n--- Python Code Analysis ---")
    python_structure = parse_code_structure(python_code, "Python")
    python_metrics = calculate_complexity_metrics(python_structure)

    # Print high-level summary
    print(f"Functions: {len(python_structure.get('ast', {}).get('functions', []))}")
    print(f"Classes: {len(python_structure.get('ast', {}).get('classes', []))}")
    print(f"Imports: {len(python_structure.get('ast', {}).get('imports', []))}")
    print(f"Cyclomatic Complexity: {python_metrics.get('cyclomatic_complexity')}")
    print(f"Node Count: {python_metrics.get('node_count')}")

    # Test JavaScript code analysis (basic fallback)
    print("\n--- JavaScript Code Analysis ---")
    js_structure = parse_code_structure(js_code, "JavaScript")
    js_metrics = calculate_complexity_metrics(js_structure)

    # Print high-level summary
    print(f"Line Count: {js_structure.get('basic', {}).get('line_count')}")
    print(f"Character Count: {js_structure.get('basic', {}).get('character_count')}")

    # Print full analysis details (uncomment if you want to see all data)
    # print("\n--- Detailed Analysis ---")
    # pretty_print(python_structure)
    # pretty_print(python_metrics)
    
    return True

if __name__ == "__main__":
    run_tests() 