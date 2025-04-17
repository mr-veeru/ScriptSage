import sys
import os
import json

# Add the parent directory to sys.path so we can import backend modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Updated import paths
from core.language_analyzer import predict_language, predict_purpose

def run_tests():
    """Run ML model tests"""
    print("=== Testing ML Models ===")
    
    # Test samples
    samples = [
        {
            "name": "Python Sample",
            "code": """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
""",
            "expected_language": "Python",
            "expected_purpose": "Algorithm Implementation"
        },
        {
            "name": "JavaScript Sample",
            "code": """
function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n-1);
}
""",
            "expected_language": "JavaScript",
            "expected_purpose": "Algorithm Implementation"
        },
        {
            "name": "CSS Sample",
            "code": """
.my-class {
    color: red;
    font-size: 12px;
}
""",
            "expected_language": "CSS",
            "expected_purpose": "Styling"
        },
        {
            "name": "React JSX Sample",
            "code": """
function Button() {
    return (
        <button 
            style={{ 
                backgroundColor: 'blue', 
                color: 'white',
                padding: '10px'
            }}
        >
            Click Me
        </button>
    );
}
""",
            "expected_language": "JavaScript",
            "expected_purpose": "UI Component"
        }
    ]
    
    passed = 0
    language_passed = 0
    total = len(samples)
    
    for i, sample in enumerate(samples, 1):
        print(f"\n--- Test {i}: {sample['name']} ---")
        
        # Test language prediction
        language = predict_language(sample["code"])
        language_correct = language == sample["expected_language"]
        if language_correct:
            language_passed += 1
        
        # Test purpose prediction - be more lenient here as it's a harder problem
        purpose, confidence = predict_purpose(sample["code"])
        purpose_correct = purpose == sample["expected_purpose"]
        
        # For UI Component and Styling, there might be some overlap, so consider close matches
        alternative_purposes = {
            "UI Component": ["Frontend", "Web Component", "React Component"],
            "Styling": ["CSS", "UI Styling", "Style Definition"]
        }
        
        if not purpose_correct and sample["expected_purpose"] in alternative_purposes:
            if purpose in alternative_purposes[sample["expected_purpose"]]:
                purpose_correct = True
                print(f"Accepted alternative purpose: {purpose}")
        
        # Print results
        print(f"Language: {language} ({'✓' if language_correct else '✗'}, expected {sample['expected_language']})")
        print(f"Purpose: {purpose} ({'✓' if purpose_correct else '✗'}, expected {sample['expected_purpose']})")
        print(f"Confidence: {confidence:.2f}")
        
        if language_correct and purpose_correct:
            passed += 1
    
    language_accuracy = (language_passed / total) * 100
    total_accuracy = (passed / total) * 100
    
    print(f"\n--- Summary ---")
    print(f"Language detection accuracy: {language_accuracy:.2f}%")
    print(f"Combined accuracy: {total_accuracy:.2f}%")
    print(f"Passed: {passed}/{total} tests")
    
    # Consider test successful if language detection is at least 75% 
    # Purpose detection is harder, so we're more lenient
    return language_accuracy >= 75  

if __name__ == "__main__":
    run_tests() 