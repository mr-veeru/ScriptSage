#!/usr/bin/env python
"""
ScriptSage Model Training Script
--------------------------------
This script trains optimized language and purpose classifiers.
"""

import os
import logging
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import sys

# Add the parent directory to sys.path to allow absolute imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import paths and constants using absolute imports
from core.language_analyzer import (
    LANGUAGE_MODEL_PATH,
    PURPOSE_MODEL_PATH
)

# Import data loading function
from core.data_preprocessing import load_augmented_samples

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_language_classifier(samples, optimize=True):
    """Train a language classifier on the provided samples"""
    logger.info("Training language classifier...")
    
    # Prepare the dataset
    X = [sample.get('content', '') for sample in samples]
    y = [sample.get('language', 'Unknown') for sample in samples]
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    
    # Create a pipeline with TF-IDF vectorizer and classifier
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            lowercase=True,
            max_features=10000,
            ngram_range=(1, 2),
            analyzer='char',
            min_df=3
        )),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # If optimization is requested, use GridSearchCV
    if optimize:
        logger.info("Performing hyperparameter optimization...")
        param_grid = {
            'vectorizer__max_features': [5000, 10000],
            'vectorizer__ngram_range': [(1, 2), (1, 3)],
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 20, 50]
        }
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        pipeline = grid_search.best_estimator_
    else:
        # Otherwise, just fit the pipeline with default parameters
        pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Language classifier accuracy: {accuracy:.4f}")
    
    # Print detailed metrics
    logger.info("Classification Report:\n" + 
                classification_report(y_test, y_pred))
    
    # Save the model
    joblib.dump(pipeline, LANGUAGE_MODEL_PATH)
    logger.info(f"Language classifier saved to {LANGUAGE_MODEL_PATH}")
    
    return pipeline, accuracy

def train_purpose_classifier(samples, optimize=True):
    """Train a code purpose classifier on the provided samples"""
    logger.info("Training purpose classifier...")
    
    # Filter samples and determine purpose based on content patterns
    purpose_samples = []
    for sample in samples:
        content = sample.get('content', '').lower()
        language = sample.get('language', '').lower()
        
        # Determine purpose based on content patterns
        purpose = 'Other'  # Default purpose
        
        # Algorithm Implementation patterns
        if any(pattern in content for pattern in [
            'factorial', 'fibonacci', 'sort', 'search', 'palindrome',
            'prime', 'binary', 'algorithm', 'recursive', 'recursion',
            'bubble sort', 'quick sort', 'merge sort', 'binary search',
            'depth first', 'breadth first', 'dijkstra', 'dynamic programming',
            'backtracking', 'memoization', 'divide and conquer'
        ]):
            purpose = 'Algorithm Implementation'
            
        # UI Component patterns
        elif any(pattern in content for pattern in [
            '<div', '<button', '<input', 'react.component', 'render()',
            'return (', 'jsx', 'className=', 'style=', 'onclick=',
            'vue.component', 'angular.component', 'svelte.component',
            'template', 'props', 'state', 'useState', 'useEffect',
            'componentDidMount', 'componentWillUnmount', 'render()',
            'return <', 'export default function', 'const Component'
        ]):
            purpose = 'UI Component'
            
        # Styling patterns
        elif any(pattern in content for pattern in [
            'color:', 'font-size:', 'margin:', 'padding:', 'background:',
            'display:', 'flex:', 'grid:', 'border:', 'width:', 'height:',
            '@media', '@keyframes', '@import', 'animation:', 'transition:',
            'transform:', 'box-shadow:', 'border-radius:', 'position:',
            'z-index:', 'opacity:', 'visibility:', 'cursor:', 'text-align:'
        ]) and language in ['css', 'scss', 'less', 'sass', 'stylus']:
            purpose = 'Styling'
            
        # API patterns
        elif any(pattern in content for pattern in [
            'fetch(', 'axios.get', 'http.get', 'api/', 'endpoint',
            'route(', '@app.route', 'res.json', 'return json',
            'express.Router', 'app.get', 'app.post', 'app.put', 'app.delete',
            'restController', '@RestController', '@RequestMapping',
            'graphql', 'apollo', 'resolver', 'mutation', 'query',
            'grpc', 'protobuf', 'service', 'rpc'
        ]):
            purpose = 'API'
            
        # Data Processing patterns
        elif any(pattern in content for pattern in [
            'process_data', 'transform', 'filter', 'map', 'reduce',
            'groupby', 'aggregate', 'pandas', 'numpy', 'dataframe',
            'spark', 'pyspark', 'rdd', 'dataset', 'dataframe',
            'etl', 'extract', 'transform', 'load', 'data pipeline',
            'data cleaning', 'data validation', 'data normalization',
            'data aggregation', 'data analysis', 'data visualization'
        ]):
            purpose = 'Data Processing'
            
        # Utility patterns
        elif any(pattern in content for pattern in [
            'utils.', 'helper', 'utility', 'format', 'parse',
            'convert', 'validate', 'check', 'is_', 'has_',
            'sanitize', 'normalize', 'encode', 'decode',
            'encrypt', 'decrypt', 'hash', 'verify',
            'logger', 'logging', 'error handling', 'exception',
            'config', 'configuration', 'settings', 'options'
        ]):
            purpose = 'Utility'
            
        # Database patterns
        elif any(pattern in content for pattern in [
            'sql', 'select', 'insert', 'update', 'delete',
            'create table', 'alter table', 'drop table',
            'mongodb', 'mongoose', 'collection', 'document',
            'redis', 'key-value', 'cache', 'session',
            'orm', 'sequelize', 'typeorm', 'prisma',
            'migration', 'schema', 'model', 'entity'
        ]):
            purpose = 'Database'
            
        # Testing patterns
        elif any(pattern in content for pattern in [
            'test(', 'describe(', 'it(', 'expect(', 'assert(',
            'jest', 'mocha', 'chai', 'cypress', 'selenium',
            'unit test', 'integration test', 'e2e test',
            'mock', 'stub', 'spy', 'fixture', 'setup',
            'teardown', 'beforeEach', 'afterEach'
        ]):
            purpose = 'Testing'
        
        purpose_samples.append({
            'content': sample.get('content', ''),
            'purpose': purpose,
            'language': sample.get('language', 'Unknown')
        })
    
    # Prepare the dataset
    X = [sample.get('content', '') for sample in purpose_samples]
    y = [sample.get('purpose', 'Other') for sample in purpose_samples]
    
    # Check if we have multiple purpose classes - if not, we need to add synthetic samples
    unique_purposes = set(y)
    if len(unique_purposes) < 2:
        logger.warning(f"Only found {len(unique_purposes)} purpose class(es). Adding synthetic samples for model training.")
        # Add synthetic samples for each purpose type
        synthetic_samples = [
            {"content": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)", "purpose": "Algorithm Implementation"},
            {"content": "function Button() {\n    return (\n        <button onClick={() => console.log('clicked')}>\n            Click Me\n        </button>\n    );\n}", "purpose": "UI Component"},
            {"content": ".my-class {\n    color: red;\n    font-size: 12px;\n}", "purpose": "Styling"},
            {"content": "async function fetchData() {\n    const response = await fetch('/api/data');\n    return response.json();\n}", "purpose": "API"},
            {"content": "def process_data(df):\n    return df.groupby('category').mean()", "purpose": "Data Processing"},
            {"content": "function formatDate(date) {\n    return date.toLocaleDateString();\n}", "purpose": "Utility"},
            {"content": "SELECT * FROM users WHERE id = ?", "purpose": "Database"},
            {"content": "test('should add numbers correctly', () => {\n    expect(add(1, 2)).toBe(3);\n});", "purpose": "Testing"}
        ]
        
        for sample in synthetic_samples:
            if sample["purpose"] not in unique_purposes:
                X.append(sample["content"])
                y.append(sample["purpose"])
                logger.info(f"Added synthetic sample for purpose: {sample['purpose']}")
    
    logger.info(f"Training purpose classifier with {len(set(y))} distinct purpose classes")
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    
    # Create a pipeline with enhanced TF-IDF vectorizer and ensemble classifier
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            lowercase=True,
            max_features=20000,
            ngram_range=(1, 4),
            analyzer='word',
            min_df=2,
            stop_words='english'
        )),
        ('classifier', GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ))
    ])
    
    # If optimization is requested, use GridSearchCV with expanded parameter grid
    if optimize:
        logger.info("Performing hyperparameter optimization...")
        param_grid = {
            'vectorizer__max_features': [10000, 20000, 30000],
            'vectorizer__ngram_range': [(1, 2), (1, 3), (1, 4)],
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        }
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,  # Increased from 3 to 5 for better cross-validation
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        pipeline = grid_search.best_estimator_
    else:
        # Otherwise, just fit the pipeline with default parameters
        pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Purpose classifier accuracy: {accuracy:.4f}")
    
    # Print detailed metrics
    logger.info("Classification Report:\n" + 
                classification_report(y_test, y_pred))
    
    # Save the model
    joblib.dump(pipeline, PURPOSE_MODEL_PATH)
    logger.info(f"Purpose classifier saved to {PURPOSE_MODEL_PATH}")
    
    return pipeline, accuracy

def main():
    """Train all models"""
    start_time = time.time()
    
    # Load augmented samples
    samples = load_augmented_samples()
    
    if not samples:
        logger.error("No samples found for training. Run data collection and preprocessing first.")
        return False
    
    # Log available languages
    languages = set(sample.get('language', 'Unknown') for sample in samples)
    logger.info(f"Training with samples from {len(languages)} languages: {', '.join(languages)}")
    
    # Train language classifier
    lang_model, lang_accuracy = train_language_classifier(samples, optimize=True)
    
    # Train purpose classifier
    purpose_model, purpose_accuracy = train_purpose_classifier(samples, optimize=True)
    
    elapsed = time.time() - start_time
    logger.info(f"Model training completed in {elapsed:.2f} seconds")
    
    return {
        "language_accuracy": lang_accuracy,
        "purpose_accuracy": purpose_accuracy
    }

if __name__ == "__main__":
    main() 