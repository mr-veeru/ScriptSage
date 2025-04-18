#!/usr/bin/env python
"""
Fix Purpose Classifier Script
----------------------------
This script diagnoses and attempts to fix issues with the purpose classifier.
"""

import os
import sys
import logging
import pickle
import joblib

# Add the parent directory to sys.path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import paths from language_analyzer
from core.language_analyzer import PURPOSE_MODEL_PATH
from core.train_models import train_purpose_classifier
from core.data_preprocessing import load_augmented_samples

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_file(file_path):
    """Inspect the file and report issues"""
    try:
        # Try to read the first few bytes to check for corruption
        with open(file_path, 'rb') as f:
            header_bytes = f.read(20)
            logger.info(f"First 20 bytes of file: {header_bytes}")
            
            # Reset file pointer and attempt to load with joblib
            f.seek(0)
            try:
                joblib.load(f)
                logger.info("File successfully loaded with joblib")
                return True
            except Exception as joblib_err:
                logger.error(f"Error loading with joblib: {joblib_err}")
                
            # Try with pickle
            f.seek(0)
            try:
                pickle.load(f)
                logger.info("File successfully loaded with pickle")
                return True
            except Exception as pickle_err:
                logger.error(f"Error loading with pickle: {pickle_err}")
                
        return False
    except Exception as e:
        logger.error(f"Error inspecting file: {e}")
        return False

def fix_purpose_classifier():
    """Attempt to fix the purpose classifier"""
    # Check if file exists
    if not os.path.exists(PURPOSE_MODEL_PATH):
        logger.error(f"Purpose classifier file not found at {PURPOSE_MODEL_PATH}")
        return False
    
    # Check file size
    file_size = os.path.getsize(PURPOSE_MODEL_PATH)
    logger.info(f"Purpose classifier file size: {file_size} bytes")
    
    # Rename the problematic file
    backup_path = f"{PURPOSE_MODEL_PATH}.bak"
    try:
        os.rename(PURPOSE_MODEL_PATH, backup_path)
        logger.info(f"Renamed problematic file to {backup_path}")
    except Exception as e:
        logger.error(f"Error renaming file: {e}")
        return False
    
    # Attempt to retrain the model
    try:
        logger.info("Loading samples for retraining...")
        samples = load_augmented_samples()
        
        if not samples:
            logger.error("No samples found for training")
            return False
        
        logger.info(f"Retraining with {len(samples)} samples")
        purpose_model, purpose_accuracy = train_purpose_classifier(samples, optimize=True)
        
        logger.info(f"Purpose classifier retrained with accuracy: {purpose_accuracy:.4f}")
        
        # Verify the new model
        if not inspect_file(PURPOSE_MODEL_PATH):
            logger.error("Newly trained model also has issues")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error retraining purpose classifier: {e}")
        return False

if __name__ == "__main__":
    print("=== Purpose Classifier Diagnostic Tool ===")
    print("Inspecting purpose classifier file...")
    
    if not inspect_file(PURPOSE_MODEL_PATH):
        print("\nThe purpose classifier file appears to be corrupted.")
        
        user_input = input("Would you like to retrain the purpose classifier? (y/n): ")
        if user_input.lower() == 'y':
            success = fix_purpose_classifier()
            if success:
                print("\nPurpose classifier has been successfully retrained and fixed.")
            else:
                print("\nFailed to fix the purpose classifier.")
        else:
            print("\nNo changes were made to the purpose classifier.")
    else:
        print("\nNo issues detected with the purpose classifier file.") 