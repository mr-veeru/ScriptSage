#!/usr/bin/env python
"""
ScriptSage Data Collection Script
---------------------------------
This script handles the collection of code samples from various sources.
"""

import os
import requests
import logging
import time
from bs4 import BeautifulSoup
from tqdm import tqdm
import pickle
from dotenv import load_dotenv
import sys

# Add the parent directory to sys.path to allow absolute imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
HEADERS = {
    'User-Agent': 'ScriptSage-ML-Trainer/1.0',
}

# Import paths and constants
from core.language_analyzer import (
    MODEL_DIR,
    DATA_DIR
)

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'online_samples'), exist_ok=True)

# Language extensions mapping
GITHUB_LANGUAGES = {
    'Python': ['py'],
    'JavaScript': ['js'],
    'TypeScript': ['ts', 'tsx'],
    'HTML': ['html', 'htm'],
    'CSS': ['css'],
    'Java': ['java'],
    'C/C++': ['c', 'cpp', 'h', 'hpp'],
    'Ruby': ['rb'],
    'Go': ['go'],
    'PHP': ['php'],
    'Shell/Bash': ['sh', 'bash'],
    'JSON': ['json'],
    'YAML': ['yml', 'yaml'],
    'Configuration': ['env', 'ini', 'cfg', 'conf', 'config'],
    'Rust': ['rs'],
    'SQL': ['sql'],
    'C#': ['cs']
}

# GitHub API details
GITHUB_API_BASE = "https://api.github.com/search/code"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com"

def fetch_github_code_samples(language, extensions, num_samples=10, min_size=100, max_size=10000):
    """Fetch code samples from GitHub for a specific language"""
    samples = []
    
    # Allow providing a GitHub token as environment variable for higher rate limits
    token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GITHUB_API_TOKEN', '')
    
    if token:
        logger.info(f"Using GitHub API token (length: {len(token)})")
    else:
        logger.warning("No GitHub token found in environment variables. API requests will be rate-limited.")
    
    headers = HEADERS.copy()
    if token:
        headers['Authorization'] = f'token {token}'

    for ext in extensions:
        query = f"extension:{ext} language:{language.lower()}"
        
        try:
            logger.info(f"Searching GitHub for {language} samples with extension .{ext}")
            response = requests.get(
                GITHUB_API_BASE,
                params={
                    'q': query,
                    'per_page': min(num_samples * 2, 100),  # Fetch more than needed in case some fail
                    'sort': 'stars',
                    'order': 'desc'
                },
                headers=headers
            )
            
            if response.status_code != 200:
                logger.warning(f"GitHub API request failed: {response.status_code} - {response.text}")
                continue
                
            search_results = response.json()
            
            if 'items' not in search_results or not search_results['items']:
                logger.warning(f"No results found for {language} with extension .{ext}")
                continue
                
            # Process the results
            for item in search_results['items'][:num_samples]:
                try:
                    # Extract repo and path information
                    repo_full_name = item['repository']['full_name']
                    file_path = item['path']
                    raw_url = f"{GITHUB_RAW_BASE}/{repo_full_name}/master/{file_path}"
                    
                    # Fetch the raw content
                    raw_response = requests.get(raw_url, headers=headers)
                    
                    if raw_response.status_code != 200:
                        logger.warning(f"Failed to fetch raw content: {raw_response.status_code}")
                        continue
                        
                    content = raw_response.text
                    
                    # Check the content size
                    if len(content) < min_size:
                        logger.debug(f"Content too small: {len(content)} bytes")
                        continue
                        
                    if len(content) > max_size:
                        # Truncate if too large
                        content = content[:max_size]
                        
                    # Add to samples
                    samples.append({
                        'language': language,
                        'extension': ext,
                        'content': content,
                        'source': f"github:{repo_full_name}/{file_path}"
                    })
                    
                    if len(samples) >= num_samples:
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing GitHub item: {e}")
            
        except Exception as e:
            logger.error(f"Error fetching {language} samples from GitHub: {e}")
    
    logger.info(f"Fetched {len(samples)} {language} samples from GitHub")
    return samples

def fetch_stackoverflow_samples(language, num_samples=10):
    """Fetch code samples from Stack Overflow for a specific language"""
    samples = []
    
    try:
        # Search for questions with the language tag
        tag = language.lower()
        # Handle special cases
        if tag == 'c/c++': 
            tag = 'c%2B%2B'
        elif tag == 'shell/bash':
            tag = 'bash'
        
        logger.info(f"Searching Stack Overflow for {language} code samples")
        
        # Fetch the questions page
        url = f"https://stackoverflow.com/questions/tagged/{tag}?tab=votes&pagesize=50"
        response = requests.get(url, headers=HEADERS)
        
        if response.status_code != 200:
            logger.warning(f"Stack Overflow request failed: {response.status_code}")
            return samples
            
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find question links
        question_links = []
        for question in soup.select('.s-post-summary'):
            link_elem = question.select_one('.s-link')
            if link_elem and link_elem.get('href'):
                question_links.append("https://stackoverflow.com" + link_elem.get('href'))
        
        # Visit each question to extract code blocks
        for link in question_links[:min(25, len(question_links))]:  # Limit to avoid too many requests
            try:
                q_response = requests.get(link, headers=HEADERS)
                
                if q_response.status_code != 200:
                    continue
                    
                q_soup = BeautifulSoup(q_response.text, 'html.parser')
                
                # Look for answers with code blocks
                answers = q_soup.select('.answer')
                
                for answer in answers:
                    # Check if this is an accepted or highly upvoted answer
                    vote_count_elem = answer.select_one('.js-vote-count')
                    vote_count = 0
                    if vote_count_elem and vote_count_elem.get_text().strip().isdigit():
                        vote_count = int(vote_count_elem.get_text().strip())
                    
                    # Only consider answers with some votes
                    if vote_count >= 3:
                        # Extract code blocks
                        code_blocks = answer.select('pre code')
                        
                        for code_block in code_blocks:
                            code = code_block.get_text()
                            
                            # Only consider non-trivial code blocks
                            if len(code) >= 50:
                                samples.append({
                                    'language': language,
                                    'content': code,
                                    'source': f"stackoverflow:{link}"
                                })
                                
                                if len(samples) >= num_samples:
                                    break
                    
                    if len(samples) >= num_samples:
                        break
                        
            except Exception as e:
                logger.error(f"Error processing Stack Overflow question: {e}")
                
            if len(samples) >= num_samples:
                break
                
    except Exception as e:
        logger.error(f"Error fetching {language} samples from Stack Overflow: {e}")
    
    logger.info(f"Fetched {len(samples)} {language} samples from Stack Overflow")
    return samples

def save_online_samples(samples, language):
    """Save collected samples to disk"""
    sample_path = os.path.join(DATA_DIR, 'online_samples', f"{language.lower().replace('/', '_')}_samples.pkl")
    
    try:
        with open(sample_path, 'wb') as f:
            pickle.dump(samples, f)
        logger.info(f"Saved {len(samples)} {language} samples to {sample_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving {language} samples: {e}")
        return False

def load_online_samples(language=None):
    """Load collected samples from disk"""
    samples = []
    sample_dir = os.path.join(DATA_DIR, 'online_samples')
    
    try:
        if language:
            # Load samples for a specific language
            sample_path = os.path.join(sample_dir, f"{language.lower().replace('/', '_')}_samples.pkl")
            if os.path.exists(sample_path):
                with open(sample_path, 'rb') as f:
                    lang_samples = pickle.load(f)
                samples.extend(lang_samples)
                logger.info(f"Loaded {len(lang_samples)} {language} samples")
        else:
            # Load all samples
            for sample_file in os.listdir(sample_dir):
                if sample_file.endswith('_samples.pkl'):
                    sample_path = os.path.join(sample_dir, sample_file)
                    with open(sample_path, 'rb') as f:
                        lang_samples = pickle.load(f)
                    samples.extend(lang_samples)
                    logger.info(f"Loaded {len(lang_samples)} samples from {sample_file}")
    except Exception as e:
        logger.error(f"Error loading samples: {e}")
    
    return samples

def fetch_all_language_samples(languages=None, samples_per_language=50):
    """Fetch code samples for all languages or specific languages"""
    if not languages:
        languages = list(GITHUB_LANGUAGES.keys())
    
    total_samples = 0
    progress_bar = tqdm(total=len(languages), desc="Collecting Language Samples")
    
    for language in languages:
        logger.info(f"Collecting samples for {language}")
        
        # Get extensions for this language
        extensions = GITHUB_LANGUAGES.get(language, [])
        
        # Fetch GitHub samples
        github_samples = fetch_github_code_samples(
            language, 
            extensions, 
            num_samples=samples_per_language,
            min_size=100
        )
        
        # Fetch Stack Overflow samples (fewer, as they're typically smaller snippets)
        stackoverflow_samples = fetch_stackoverflow_samples(
            language, 
            num_samples=samples_per_language // 5
        )
        
        # Combine samples
        combined_samples = github_samples + stackoverflow_samples
        
        # Save samples
        if combined_samples:
            save_online_samples(combined_samples, language)
            total_samples += len(combined_samples)
        
        progress_bar.update(1)
    
    progress_bar.close()
    logger.info(f"Collected a total of {total_samples} samples for {len(languages)} languages")
    return total_samples

def main():
    """Main function for data collection"""
    start_time = time.time()
    
    # Parse command line arguments (could be added for more flexibility)
    # For now, use default values
    languages = list(GITHUB_LANGUAGES.keys())
    samples_per_language = 30
    
    # Fetch samples for all languages
    fetch_all_language_samples(languages, samples_per_language)
    
    elapsed = time.time() - start_time
    logger.info(f"Data collection completed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main() 