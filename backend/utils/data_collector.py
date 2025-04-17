import os
import requests
import base64
import json
import time
import random
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# Configuration
GITHUB_API_TOKEN = os.environ.get('GITHUB_API_TOKEN', '')  # Get token from environment
if not GITHUB_API_TOKEN:
    print("Warning: GITHUB_API_TOKEN not found. API requests will be rate limited.")
else:
    print(f"GitHub API token loaded: {GITHUB_API_TOKEN[:4]}...{GITHUB_API_TOKEN[-4:]}")

LANGUAGES = [
    'Python', 'JavaScript', 'Java', 'C', 'C++', 'C#', 'Go', 'Ruby', 
    'PHP', 'TypeScript', 'Rust', 'Swift', 'Kotlin', 'Scala',
    'Shell', 'HTML', 'CSS', 'SQL'
]

# Use a single data directory at the backend level
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
SAMPLES_PER_LANGUAGE = 500  # Number of samples to collect per language

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def github_search_code(query, language, page=1, per_page=100):
    """Search for code on GitHub by language"""
    headers = {
        'Accept': 'application/vnd.github.v3+json'
    }
    
    if GITHUB_API_TOKEN:
        headers['Authorization'] = f'token {GITHUB_API_TOKEN}'
    
    search_url = 'https://api.github.com/search/code'
    params = {
        'q': f'{query} language:{language}',
        'page': page,
        'per_page': per_page,
        'sort': 'stars'
    }
    
    try:
        # Add timeout to prevent hanging
        response = requests.get(search_url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 403:
            # Rate limit exceeded
            print(f"Rate limit exceeded. Waiting for 60 seconds...")
            time.sleep(60)
            return []
            
        if response.status_code != 200:
            print(f"Error searching GitHub: {response.status_code}")
            print(response.json())
            return []
        
        return response.json()
    except requests.exceptions.Timeout:
        print(f"Request timed out for {language}/{query}. Skipping...")
        return []
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return []

def fetch_file_content(file_url):
    """Fetch file content from GitHub"""
    headers = {
        'Accept': 'application/vnd.github.v3+json'
    }
    
    if GITHUB_API_TOKEN:
        headers['Authorization'] = f'token {GITHUB_API_TOKEN}'
    
    try:
        # Add timeout to prevent hanging
        response = requests.get(file_url, headers=headers, timeout=20)
        
        if response.status_code != 200:
            print(f"Error fetching file: {response.status_code}")
            return None
        
        data = response.json()
        if data.get('encoding') == 'base64' and data.get('content'):
            content = base64.b64decode(data['content']).decode('utf-8', errors='replace')
            return content
        
        return None
    except requests.exceptions.Timeout:
        print(f"File fetch timed out for {file_url}. Skipping...")
        return None
    except requests.exceptions.RequestException as e:
        print(f"File fetch failed: {e}")
        return None

def collect_samples():
    """Collect code samples from GitHub for all languages"""
    all_samples = []
    
    for language in LANGUAGES:
        print(f"Collecting samples for {language}...")
        lang_samples = []
        
        # Different search queries to get diverse code samples
        search_queries = [
            'function', 'class', 'algorithm', 'util', 'helper',
            'data', 'api', 'test', 'model', 'service'
        ]
        
        # Create a progress bar for search queries
        progress_bar = tqdm(search_queries, desc=f"Processing {language} queries")
        
        for query in progress_bar:
            if len(lang_samples) >= SAMPLES_PER_LANGUAGE:
                break
                
            try:
                search_results = github_search_code(query, language)
                
                if 'items' not in search_results:
                    progress_bar.write(f"No items found for {language}/{query}")
                    continue
                    
                items_count = len(search_results.get('items', []))
                progress_bar.write(f"Found {items_count} items for {language}/{query}")
                
                # Create a nested progress bar for processing items
                item_bar = tqdm(search_results.get('items', []), 
                              desc=f"Processing {items_count} files",
                              leave=False)
                
                for item in item_bar:
                    if len(lang_samples) >= SAMPLES_PER_LANGUAGE:
                        break
                        
                    try:
                        file_url = item['url']
                        item_bar.set_description(f"Fetching {item['name']}")
                        content = fetch_file_content(file_url)
                        
                        if content and 50 < len(content) < 10000:  # Size limits
                            lang_samples.append({
                                'language': language,
                                'content': content,
                                'url': item['html_url'],
                                'repo': item['repository']['full_name']
                            })
                            
                            # Add a small delay to avoid hitting rate limits
                            time.sleep(0.5 + random.random())
                    except Exception as e:
                        item_bar.write(f"Error processing item: {str(e)}")
                        continue
            
            except Exception as e:
                progress_bar.write(f"Error processing {language}/{query}: {str(e)}")
            
            # Respect GitHub API rate limits
            rate_limit_delay = 2 + random.random() * 3
            progress_bar.write(f"Waiting {rate_limit_delay:.2f}s for rate limit...")
            time.sleep(rate_limit_delay)
        
        # Save samples after each language to avoid losing data if script crashes
        if lang_samples:
            print(f"Collected {len(lang_samples)} samples for {language}")
            all_samples.extend(lang_samples)
            
            # Save language-specific samples
            lang_df = pd.DataFrame(lang_samples)
            lang_path = os.path.join(OUTPUT_DIR, f'{language.lower()}_samples.csv')
            lang_df.to_csv(lang_path, index=False)
            print(f"Saved to {lang_path}")
        else:
            print(f"No samples collected for {language}")
    
    # Save all samples
    if all_samples:
        all_df = pd.DataFrame(all_samples)
        all_path = os.path.join(OUTPUT_DIR, 'all_samples.csv')
        all_df.to_csv(all_path, index=False)
        print(f"Saved all {len(all_samples)} samples to {all_path}")
    
    return all_samples

def collect_purpose_samples():
    """Collect code samples specifically for purpose classification"""
    purpose_categories = {
        'algorithm': ['sort', 'search', 'graph', 'tree', 'linkedlist', 'algorithm'],
        'web': ['api', 'http', 'server', 'router', 'endpoint', 'request'],
        'database': ['sql', 'query', 'database', 'crud', 'model', 'schema'],
        'machine_learning': ['model', 'train', 'predict', 'classifier', 'regression', 'neural'],
        'utility': ['util', 'helper', 'format', 'convert', 'validate', 'parse'],
        'testing': ['test', 'assert', 'mock', 'spec', 'fixture', 'benchmark']
    }
    
    purpose_samples = []
    
    for purpose, queries in purpose_categories.items():
        purpose_count = 0
        print(f"Collecting samples for {purpose} purpose...")
        
        for query in queries:
            if purpose_count >= 100:
                break
                
            for language in random.sample(LANGUAGES, 5):  # Sample from different languages
                try:
                    search_results = github_search_code(query, language)
                    
                    if 'items' not in search_results:
                        continue
                        
                    for item in search_results['items']:
                        if purpose_count >= 100:
                            break
                            
                        file_url = item['url']
                        content = fetch_file_content(file_url)
                        
                        if content and 50 < len(content) < 5000:  # Size limits
                            purpose_samples.append({
                                'purpose': purpose,
                                'language': language,
                                'content': content,
                                'url': item['html_url'],
                                'query': query
                            })
                            purpose_count += 1
                            
                            # Add a small delay to avoid hitting rate limits
                            time.sleep(0.5 + random.random())
                
                except Exception as e:
                    print(f"Error processing {purpose}/{query}/{language}: {str(e)}")
                
                # Respect GitHub API rate limits
                time.sleep(2 + random.random() * 3)
        
        print(f"Collected {purpose_count} samples for {purpose} purpose")
    
    # Save purpose samples
    purpose_df = pd.DataFrame(purpose_samples)
    purpose_df.to_csv(os.path.join(OUTPUT_DIR, 'purpose_samples.csv'), index=False)
    
    return purpose_samples

if __name__ == "__main__":
    print("Starting code sample collection from GitHub...")
    collect_samples()
    collect_purpose_samples()
    print(f"Collection complete. Data saved to {OUTPUT_DIR}") 