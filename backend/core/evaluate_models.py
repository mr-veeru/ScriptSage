#!/usr/bin/env python
"""
ScriptSage Model Evaluation Script
---------------------------------
This script evaluates the trained models against test data.
"""

import os
import sys
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix,
    classification_report
)
import joblib

# Add the parent directory to sys.path to allow absolute imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import paths and constants
from core.language_analyzer import (
    MODEL_DIR, 
    LANGUAGE_MODEL_PATH,
    PURPOSE_MODEL_PATH,
    predict_language,
    predict_purpose
)

# Import data preprocessing
from core.data_preprocessing import load_augmented_samples

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_language_classifier():
    """Evaluate the language classifier against test samples"""
    logger.info("Evaluating language classifier...")
    
    # Check if model exists
    if not os.path.exists(LANGUAGE_MODEL_PATH):
        logger.error(f"Language classifier model not found at {LANGUAGE_MODEL_PATH}")
        return False
    
    # Load the model
    model = joblib.load(LANGUAGE_MODEL_PATH)
    
    # Load test samples
    samples = load_augmented_samples()
    
    if not samples:
        logger.error("No samples found for evaluation")
        return False
    
    # Create test set (we'll use a small portion of the samples)
    test_samples = samples[-int(len(samples) * 0.2):]  # Use the last 20% as test set
    
    X_test = [sample.get('content', '') for sample in test_samples]
    y_true = [sample.get('language', 'Unknown') for sample in test_samples]
    
    # Predict
    logger.info(f"Making predictions on {len(X_test)} test samples...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    logger.info(f"Language classifier accuracy: {accuracy:.4f}")
    
    # Detailed report
    report = classification_report(y_true, y_pred)
    logger.info(f"Classification report:\n{report}")
    
    # Calculate confusion matrix
    labels = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create a figure for the confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Language Classification Confusion Matrix')
    
    # Save the figure
    cm_path = os.path.join(MODEL_DIR, "language_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    logger.info(f"Confusion matrix saved to {cm_path}")
    
    return accuracy

def evaluate_purpose_classifier():
    """Evaluate the purpose classifier against test samples"""
    logger.info("Evaluating purpose classifier...")
    
    # Check if model exists
    if not os.path.exists(PURPOSE_MODEL_PATH):
        logger.error(f"Purpose classifier model not found at {PURPOSE_MODEL_PATH}")
        return False
    
    # Load the model
    model = joblib.load(PURPOSE_MODEL_PATH)
    
    # Load test samples
    samples = load_augmented_samples()
    
    if not samples:
        logger.error("No samples found for evaluation")
        return False
    
    # Filter samples with purpose information
    purpose_samples = []
    for sample in samples:
        source = sample.get('source', '')
        if 'web_ui' in source:
            purpose = 'Web UI'
        elif 'data_processing' in source:
            purpose = 'Data Processing'
        elif 'api' in source:
            purpose = 'API'
        elif 'utility' in source:
            purpose = 'Utility'
        else:
            purpose = 'Other'
        
        purpose_samples.append({
            'content': sample.get('content', ''),
            'purpose': purpose
        })
    
    # Create test set (we'll use a small portion of the samples)
    test_samples = purpose_samples[-int(len(purpose_samples) * 0.2):]
    
    X_test = [sample.get('content', '') for sample in test_samples]
    y_true = [sample.get('purpose', 'Other') for sample in test_samples]
    
    # Predict
    logger.info(f"Making predictions on {len(X_test)} test samples...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    logger.info(f"Purpose classifier accuracy: {accuracy:.4f}")
    
    # Detailed report
    report = classification_report(y_true, y_pred)
    logger.info(f"Classification report:\n{report}")
    
    # Calculate confusion matrix
    labels = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create a figure for the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Purpose Classification Confusion Matrix')
    
    # Save the figure
    cm_path = os.path.join(MODEL_DIR, "purpose_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    logger.info(f"Confusion matrix saved to {cm_path}")
    
    return accuracy

def evaluate_with_examples():
    """Test the models with real-world examples"""
    logger.info("Evaluating models with example code snippets...")
    
    # Test examples for different languages
    examples = {
        "Python": """
def calculate_fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

if __name__ == "__main__":
    print(calculate_fibonacci(10))
        """,
        
        "JavaScript": """
function fetchUserData() {
    return fetch('/api/users')
        .then(response => response.json())
        .then(data => {
            renderUserList(data);
            return data;
        })
        .catch(error => console.error('Error fetching users:', error));
}

document.getElementById('user-button').addEventListener('click', fetchUserData);
        """,
        
        "HTML": """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Web Page</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <h1>Welcome to my website</h1>
    <p>This is a paragraph of text.</p>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
    </ul>
    <script src="app.js"></script>
</body>
</html>
        """
    }
    
    results = []
    
    for true_language, code in examples.items():
        # Predict language
        predicted_language = predict_language(code)
        
        # Predict purpose
        predicted_purpose = predict_purpose(code)
        
        # Add to results
        results.append({
            'True Language': true_language,
            'Predicted Language': predicted_language,
            'Predicted Purpose': predicted_purpose,
            'Language Correct': predicted_language == true_language,
            'Code Sample': code[:50] + "..." if len(code) > 50 else code
        })
    
    # Display results
    for i, result in enumerate(results):
        logger.info(f"Example {i+1}:")
        logger.info(f"  True Language: {result['True Language']}")
        logger.info(f"  Predicted Language: {result['Predicted Language']}")
        logger.info(f"  Predicted Purpose: {result['Predicted Purpose']}")
        logger.info(f"  Language Prediction Correct: {result['Language Correct']}")
        logger.info("")
    
    # Calculate accuracy
    language_accuracy = sum(1 for r in results if r['Language Correct']) / len(results)
    logger.info(f"Example language prediction accuracy: {language_accuracy:.2f}")
    
    return results

def evaluate_models():
    """Run all evaluation methods"""
    logger.info("Starting model evaluation...")
    
    start_time = time.time()
    
    # Evaluate language classifier
    lang_accuracy = evaluate_language_classifier()
    
    # Evaluate purpose classifier
    purpose_accuracy = evaluate_purpose_classifier()
    
    # Test with examples
    example_results = evaluate_with_examples()
    
    elapsed = time.time() - start_time
    logger.info(f"Model evaluation completed in {elapsed:.2f} seconds")
    
    return {
        "language_accuracy": lang_accuracy,
        "purpose_accuracy": purpose_accuracy,
        "examples": len(example_results)
    }

def main():
    """Main function"""
    result = evaluate_models()
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main()) 