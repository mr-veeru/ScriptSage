import os
import re
import pickle
import requests
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

# Import patterns and constants from the main analyzer
from language_analyzer import (
    LANGUAGE_PATTERNS, 
    LANGUAGE_SAMPLES, 
    PURPOSE_SAMPLES,
    MODEL_DIR, 
    DATA_DIR,
    LANGUAGE_MODEL_PATH,
    PURPOSE_MODEL_PATH,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'augmented'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'online_samples'), exist_ok=True)

# Online sources for code samples
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

# GitHub search API base URL
GITHUB_API_BASE = "https://api.github.com/search/code"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com"

# Default headers for API requests
HEADERS = {
    'User-Agent': 'ScriptSage-ML-Trainer/1.0',
}

def fetch_github_code_samples(language, extensions, num_samples=10, min_size=100, max_size=10000):
    """Fetch code samples from GitHub for a specific language"""
    samples = []
    
    # Allow providing a GitHub token as environment variable for higher rate limits
    token = os.environ.get('GITHUB_TOKEN', '')
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
                    
                    # Only use answers with positive votes
                    if vote_count > 0 or answer.get('accepted-answer', False):
                        # Extract code blocks
                        code_blocks = answer.select('code')
                        
                        for code_block in code_blocks:
                            content = code_block.get_text()
                            
                            # Skip very short code blocks
                            if len(content) < 50:
                                continue
                                
                            samples.append({
                                'language': language,
                                'content': content,
                                'source': f"stackoverflow:{link}"
                            })
                            
                            if len(samples) >= num_samples:
                                break
                    
                    if len(samples) >= num_samples:
                        break
                
                if len(samples) >= num_samples:
                    break
                    
            except Exception as e:
                logger.error(f"Error processing Stack Overflow question: {e}")
    
    except Exception as e:
        logger.error(f"Error fetching {language} samples from Stack Overflow: {e}")
    
    logger.info(f"Fetched {len(samples)} {language} samples from Stack Overflow")
    return samples

def save_online_samples(samples, language):
    """Save fetched online samples to a CSV file"""
    if not samples:
        logger.warning(f"No samples to save for {language}")
        return
        
    # Create a DataFrame from the samples
    df = pd.DataFrame(samples)
    
    # Save to CSV
    output_path = os.path.join(DATA_DIR, 'online_samples', f"{language.replace('/', '_')}_samples.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(samples)} {language} samples to {output_path}")

def load_online_samples(language=None):
    """Load previously saved online samples"""
    samples_dir = os.path.join(DATA_DIR, 'online_samples')
    
    if not os.path.exists(samples_dir):
        logger.warning(f"Online samples directory does not exist: {samples_dir}")
        return []
    
    all_samples = []
    
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(samples_dir) if f.endswith('.csv')]
    
    if language:
        # Filter to only the requested language
        lang_file = f"{language.replace('/', '_')}_samples.csv"
        csv_files = [f for f in csv_files if f == lang_file]
    
    # Load each file
    for file in csv_files:
        try:
            df = pd.read_csv(os.path.join(samples_dir, file))
            samples = df.to_dict('records')
            all_samples.extend(samples)
            logger.info(f"Loaded {len(samples)} samples from {file}")
        except Exception as e:
            logger.error(f"Error loading samples from {file}: {e}")
    
    return all_samples

def fetch_all_language_samples(languages=None, samples_per_language=20):
    """Fetch code samples for all languages from online sources"""
    if languages is None:
        languages = list(GITHUB_LANGUAGES.keys())
    
    all_samples = []
    
    for language in languages:
        logger.info(f"Fetching samples for {language}")
        
        # Get GitHub samples
        if language in GITHUB_LANGUAGES:
            github_samples = fetch_github_code_samples(
                language, 
                GITHUB_LANGUAGES[language],
                num_samples=samples_per_language
            )
            all_samples.extend(github_samples)
            
            # Save these samples immediately
            save_online_samples(github_samples, language)
        
        # Get Stack Overflow samples
        stackoverflow_samples = fetch_stackoverflow_samples(
            language,
            num_samples=samples_per_language // 2  # Fewer from SO as they're usually shorter
        )
        all_samples.extend(stackoverflow_samples)
        
        # Save these samples immediately
        save_online_samples(stackoverflow_samples, language)
    
    return all_samples

def augment_training_data():
    """Augment the built-in training data with online samples"""
    # Load online samples if available
    online_samples = load_online_samples()
    
    if not online_samples:
        logger.warning("No online samples available, fetching new samples")
        online_samples = fetch_all_language_samples()
    
    # Combine with built-in samples
    all_samples = []
    
    # Add built-in samples
    for language, samples in LANGUAGE_SAMPLES.items():
        for sample in samples:
            all_samples.append({
                'language': language,
                'content': sample,
                'source': 'built-in'
            })
    
    # Add online samples
    all_samples.extend(online_samples)
    
    # Save the augmented dataset
    augmented_path = os.path.join(DATA_DIR, 'augmented', 'augmented_language_samples.csv')
    pd.DataFrame(all_samples).to_csv(augmented_path, index=False)
    
    logger.info(f"Created augmented dataset with {len(all_samples)} samples at {augmented_path}")
    return all_samples

def train_optimized_language_classifier(optimize=True):
    """Train an optimized language classifier with extended data"""
    # Augment the training data first
    augment_training_data()
    
    # Get the augmented training data
    X, y = [], []
    
    # Try to load augmented data
    augmented_path = os.path.join(DATA_DIR, 'augmented', 'augmented_language_samples.csv')
    if os.path.exists(augmented_path):
        df = pd.read_csv(augmented_path)
        X = df['content'].tolist()
        y = df['language'].tolist()
    else:
        # Fallback to built-in samples
        logger.warning("Augmented dataset not found, falling back to built-in samples")
        for language, samples in LANGUAGE_PATTERNS.items():
            for sample in samples:
                X.append(sample['example'])
                y.append(language)
    
    logger.info(f"Training language classifier with {len(X)} samples")
    
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if optimize:
        # Create a pipeline with parameter grid for optimization
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', MultinomialNB())
        ])
        
        # Define parameter grid for optimization
        param_grid = {
            'tfidf__analyzer': ['char', 'word'],
            'tfidf__ngram_range': [(2, 3), (2, 4), (1, 4)],
            'tfidf__max_features': [5000, 10000, 15000],
            'classifier__alpha': [0.01, 0.1, 0.5, 1.0],
        }
        
        # Count classes to check if we have enough samples for cross-validation
        class_counts = {}
        for label in y_train:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        min_class_count = min(class_counts.values()) if class_counts else 0
        
        # Check if we have enough samples for cross-validation
        if min_class_count < 3:
            logger.info("Sample size too small for cross-validation, using simple optimization")
            
            # Use a simpler approach - just train with best known parameters
            best_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    analyzer='char', 
                    ngram_range=(1, 4), 
                    max_features=5000
                )),
                ('classifier', MultinomialNB(alpha=0.01))
            ])
            
            best_pipeline.fit(X_train, y_train)
        else:
            # Adjust CV based on the smallest class size
            cv_folds = min(3, min_class_count)
            logger.info(f"Using {cv_folds}-fold cross validation")
            
            # Perform grid search
            logger.info("Performing grid search to find optimal parameters")
            grid_search = GridSearchCV(
                pipeline, 
                param_grid,
                cv=cv_folds,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get the best parameters
            logger.info(f"Best parameters: {grid_search.best_params_}")
            
            # Use the best estimator
            best_pipeline = grid_search.best_estimator_
    else:
        # Use a pre-configured pipeline with good default parameters
        best_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='char', 
                ngram_range=(2, 4), 
                max_features=10000
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        best_pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    accuracy = best_pipeline.score(X_test, y_test)
    logger.info(f"Language classifier accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    from sklearn.metrics import classification_report
    y_pred = best_pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    logger.info(f"Classification report:\n{report}")
    
    # Save the trained model
    with open(LANGUAGE_MODEL_PATH, 'wb') as f:
        pickle.dump(best_pipeline, f)
    
    logger.info(f"Saved optimized language classifier to {LANGUAGE_MODEL_PATH}")
    
    return best_pipeline, accuracy

def train_optimized_purpose_classifier(optimize=True):
    """Train an optimized purpose classifier"""
    # Get purpose training data
    X, y = [], []
    
    # Add built-in samples
    for purpose, samples in PURPOSE_SAMPLES.items():
        for sample in samples:
            X.append(sample)
            y.append(purpose)
    
    # Add more samples for common purposes
    additional_samples = {
        'Algorithm Implementation': [
            'def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x < pivot]\n    right = [x for x in arr[1:] if x >= pivot]\n    return quicksort(left) + [pivot] + quicksort(right)',
            'function fibonacci(n) {\n    if (n <= 1) return n;\n    return fibonacci(n-1) + fibonacci(n-2);\n}',
            'public int binarySearch(int[] arr, int target) {\n    int left = 0, right = arr.length - 1;\n    while (left <= right) {\n        int mid = left + (right - left) / 2;\n        if (arr[mid] == target) return mid;\n        if (arr[mid] < target) left = mid + 1;\n        else right = mid - 1;\n    }\n    return -1;\n}'
        ],
        'Data Processing': [
            'df = pd.read_csv("data.csv")\ndf_filtered = df[df["value"] > 10]\nresult = df_filtered.groupby("category").mean()',
            'const result = data.filter(item => item.price > 100).map(item => ({...item, tax: item.price * 0.1}))',
            'SELECT customer_id, SUM(order_amount) FROM orders GROUP BY customer_id HAVING COUNT(*) > 5'
        ],
        'UI Component': [
            'function Button({ onClick, children }) {\n  return <button className="btn primary" onClick={onClick}>{children}</button>;\n}',
            'class Navbar extends React.Component {\n  render() {\n    return (\n      <nav className="navbar">\n        <div className="logo">{this.props.logo}</div>\n        <ul className="menu">{this.props.children}</ul>\n      </nav>\n    );\n  }\n}',
            '<template>\n  <div class="card">\n    <div class="card-header">{{ title }}</div>\n    <div class="card-body">{{ content }}</div>\n  </div>\n</template>'
        ],
        'Utility Function': [
            'def validate_email(email):\n    import re\n    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"\n    return bool(re.match(pattern, email))',
            'function formatCurrency(amount, currency = "USD") {\n  return new Intl.NumberFormat("en-US", {\n    style: "currency",\n    currency\n  }).format(amount);\n}',
            'public static String generateRandomString(int length) {\n    String chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";\n    Random random = new Random();\n    StringBuilder sb = new StringBuilder(length);\n    for (int i = 0; i < length; i++) {\n        sb.append(chars.charAt(random.nextInt(chars.length())));\n    }\n    return sb.toString();\n}'
        ]
    }
    
    # Add additional samples
    for purpose, samples in additional_samples.items():
        for sample in samples:
            X.append(sample)
            y.append(purpose)
    
    logger.info(f"Training purpose classifier with {len(X)} samples")
    
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if optimize:
        # Create a pipeline with parameter grid for optimization
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', RandomForestClassifier())
        ])
        
        # Define parameter grid for optimization
        param_grid = {
            'tfidf__analyzer': ['word'],
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'tfidf__max_features': [3000, 5000, 7000],
            'classifier__n_estimators': [50, 100, 150],
            'classifier__max_depth': [None, 10, 20, 30],
        }
        
        # Count classes to check if we have enough samples for cross-validation
        class_counts = {}
        for label in y_train:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        min_class_count = min(class_counts.values()) if class_counts else 0
        
        # Check if we have enough samples for cross-validation
        if min_class_count < 3:
            logger.info("Sample size too small for cross-validation, using simple optimization")
            
            # Use a simpler approach - just train with best known parameters
            best_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    analyzer='word',
                    ngram_range=(1, 1), 
                    max_features=7000
                )),
                ('classifier', RandomForestClassifier(n_estimators=50, max_depth=None))
            ])
            
            best_pipeline.fit(X_train, y_train)
        else:
            # Adjust CV based on the smallest class size
            cv_folds = min(3, min_class_count)
            logger.info(f"Using {cv_folds}-fold cross validation")
            
            # Perform grid search
            logger.info("Performing grid search to find optimal parameters")
            grid_search = GridSearchCV(
                pipeline, 
                param_grid,
                cv=cv_folds,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get the best parameters
            logger.info(f"Best parameters: {grid_search.best_params_}")
            
            # Use the best estimator
            best_pipeline = grid_search.best_estimator_
    else:
        # Use a pre-configured pipeline
        best_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='word',
                ngram_range=(1, 2),
                max_features=5000
            )),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=20))
        ])
        
        best_pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    accuracy = best_pipeline.score(X_test, y_test)
    logger.info(f"Purpose classifier accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    from sklearn.metrics import classification_report
    y_pred = best_pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    logger.info(f"Classification report:\n{report}")
    
    # Save the trained model
    with open(PURPOSE_MODEL_PATH, 'wb') as f:
        pickle.dump(best_pipeline, f)
    
    logger.info(f"Saved optimized purpose classifier to {PURPOSE_MODEL_PATH}")
    
    return best_pipeline, accuracy

def main():
    """Main training function"""
    logger.info("Starting ML model training")
    
    # Fetch online samples if needed
    if not os.listdir(os.path.join(DATA_DIR, 'online_samples')):
        logger.info("Fetching online code samples")
        fetch_all_language_samples()
    
    # Train language classifier
    logger.info("Training language classifier")
    lang_pipeline, lang_accuracy = train_optimized_language_classifier(optimize=True)
    logger.info(f"Language classifier accuracy: {lang_accuracy:.4f}")
    
    # Train purpose classifier
    logger.info("Training purpose classifier")
    purpose_pipeline, purpose_accuracy = train_optimized_purpose_classifier(optimize=True)
    logger.info(f"Purpose classifier accuracy: {purpose_accuracy:.4f}")
    
    logger.info("ML model training completed")
    return {
        "language_accuracy": lang_accuracy,
        "purpose_accuracy": purpose_accuracy,
        "language_model_path": LANGUAGE_MODEL_PATH,
        "purpose_model_path": PURPOSE_MODEL_PATH
    }

if __name__ == "__main__":
    main() 