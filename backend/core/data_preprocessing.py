#!/usr/bin/env python
"""
ScriptSage Data Preprocessing Script
-----------------------------------
This script handles the preprocessing of collected code samples.
"""

import os
import re
import logging
import pickle
import random
from tqdm import tqdm
import time
import sys

# Add the parent directory to sys.path to allow absolute imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import paths and constants
from core.language_analyzer import (
    DATA_DIR
)

# Import data loading function
from core.data_collection import load_online_samples

# Create necessary directories
os.makedirs(os.path.join(DATA_DIR, 'augmented'), exist_ok=True)

def clean_code_sample(code, language):
    """Clean and normalize code samples"""
    # Remove long comment blocks
    if language == 'Python':
        # Remove triple-quoted docstrings
        code = re.sub(r'"""[\s\S]*?"""', '', code)
        code = re.sub(r"'''[\s\S]*?'''", '', code)
    elif language in ['JavaScript', 'TypeScript', 'Java', 'C/C++', 'C#']:
        # Remove /* */ style comments
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    
    # Remove excessive blank lines (more than 2 in a row)
    code = re.sub(r'\n{3,}', '\n\n', code)
    
    # Truncate if too long (focus on the first part which often contains more relevant info)
    if len(code) > 5000:
        code = code[:5000]
    
    return code

def preprocess_code_samples():
    """Clean and preprocess all collected code samples"""
    logger.info("Loading online samples...")
    samples = load_online_samples()
    
    if not samples:
        logger.warning("No samples to preprocess. Run data collection first.")
        return 0
    
    logger.info(f"Preprocessing {len(samples)} code samples...")
    processed_samples = []
    
    progress_bar = tqdm(total=len(samples), desc="Preprocessing Samples")
    
    for sample in samples:
        language = sample.get('language')
        content = sample.get('content', '')
        
        # Skip empty content
        if not content:
            progress_bar.update(1)
            continue
        
        # Clean the sample
        cleaned_content = clean_code_sample(content, language)
        
        # Only keep meaningful samples
        if len(cleaned_content) >= 50:  # Arbitrary minimum threshold
            processed_sample = {
                'language': language,
                'content': cleaned_content,
                'source': sample.get('source', 'unknown')
            }
            processed_samples.append(processed_sample)
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Save processed samples
    processed_path = os.path.join(DATA_DIR, 'processed_samples.pkl')
    with open(processed_path, 'wb') as f:
        pickle.dump(processed_samples, f)
    
    logger.info(f"Preprocessed {len(processed_samples)} samples saved to {processed_path}")
    return len(processed_samples)

def augment_training_data():
    """Create augmented samples from processed samples"""
    # Load processed samples
    processed_path = os.path.join(DATA_DIR, 'processed_samples.pkl')
    
    if not os.path.exists(processed_path):
        logger.warning("No processed samples found. Run preprocessing first.")
        return 0
    
    with open(processed_path, 'rb') as f:
        processed_samples = pickle.load(f)
    
    logger.info(f"Augmenting {len(processed_samples)} processed samples...")
    augmented_samples = []
    
    progress_bar = tqdm(total=len(processed_samples), desc="Augmenting Samples")
    
    for sample in processed_samples:
        language = sample.get('language')
        content = sample.get('content', '')
        
        # Skip empty content
        if not content:
            progress_bar.update(1)
            continue
        
        lines = content.split('\n')
        
        # Original sample
        augmented_samples.append(sample)
        
        # If sample is large enough, create truncated versions
        if len(lines) >= 20:
            # Take first half
            first_half = '\n'.join(lines[:len(lines)//2])
            augmented_samples.append({
                'language': language,
                'content': first_half,
                'source': f"{sample.get('source', 'unknown')}:truncated:first_half"
            })
            
            # Take random contiguous section
            if len(lines) >= 30:
                start_idx = random.randint(0, len(lines) - 15)  # At least 15 lines
                section_length = random.randint(15, min(len(lines) - start_idx, 30))
                section = '\n'.join(lines[start_idx:start_idx + section_length])
                augmented_samples.append({
                    'language': language,
                    'content': section,
                    'source': f"{sample.get('source', 'unknown')}:truncated:random_section"
                })
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Save augmented samples
    augmented_path = os.path.join(DATA_DIR, 'augmented', 'augmented_samples.pkl')
    with open(augmented_path, 'wb') as f:
        pickle.dump(augmented_samples, f)
    
    logger.info(f"Augmented to {len(augmented_samples)} samples saved to {augmented_path}")
    return len(augmented_samples)

def load_augmented_samples():
    """Load augmented samples for training"""
    augmented_path = os.path.join(DATA_DIR, 'augmented', 'augmented_samples.pkl')
    
    if not os.path.exists(augmented_path):
        logger.warning("No augmented samples found. Run data augmentation first.")
        return []
    
    with open(augmented_path, 'rb') as f:
        augmented_samples = pickle.load(f)
    
    logger.info(f"Loaded {len(augmented_samples)} augmented samples")
    return augmented_samples

def main():
    """Main function for data preprocessing"""
    start_time = time.time()
    
    # Preprocess the collected samples
    preprocess_code_samples()
    
    # Augment the preprocessed samples
    augment_training_data()
    
    elapsed = time.time() - start_time
    logger.info(f"Data preprocessing completed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main() 