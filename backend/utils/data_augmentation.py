import os
import pandas as pd
import numpy as np
import re
import random
from tqdm import tqdm

# Use the same data directory as data_collector.py at the backend level
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
AUGMENTED_DIR = os.path.join(DATA_DIR, 'augmented')
os.makedirs(AUGMENTED_DIR, exist_ok=True)

def comment_modification(code, language):
    """Add or modify comments in code based on language"""
    if language == 'Python':
        lines = code.split('\n')
        for i in range(len(lines)):
            if random.random() < 0.2 and not lines[i].strip().startswith('#'):
                lines[i] = lines[i] + "  # Modified line"
        return '\n'.join(lines)
    
    elif language in ['JavaScript', 'TypeScript', 'Java', 'C++', 'C#']:
        if random.random() < 0.3:
            return re.sub(r'/\*.*?\*/', '/* Modified comment */', code)
        else:
            lines = code.split('\n')
            for i in range(len(lines)):
                if random.random() < 0.2 and not lines[i].strip().startswith('//'):
                    lines[i] = lines[i] + "  // Modified line"
            return '\n'.join(lines)
    
    return code  # Return unchanged for other languages

def whitespace_modification(code):
    """Modify whitespace in code"""
    # Replace tabs with spaces or vice versa
    if random.random() < 0.5:
        code = code.replace('\t', '    ')
    else:
        code = code.replace('    ', '\t')
    
    # Add or remove blank lines
    lines = code.split('\n')
    new_lines = []
    
    for line in lines:
        new_lines.append(line)
        if random.random() < 0.1:
            new_lines.append('')
    
    return '\n'.join(new_lines)

def variable_renaming(code, language):
    """Rename variables in code based on language"""
    # Simple variable name patterns by language
    patterns = {
        'Python': r'\b([a-z][a-z0-9_]*)\b(?=\s*=)',
        'JavaScript': r'\b(var|let|const)\s+([a-z][a-z0-9_]*)\b',
        'Java': r'\b(int|String|boolean|double|float)\s+([a-z][a-z0-9_]*)\b',
        'C++': r'\b(int|char|bool|double|float)\s+([a-z][a-z0-9_]*)\b',
    }
    
    if language not in patterns:
        return code
    
    pattern = patterns[language]
    variable_names = set(re.findall(pattern, code))
    
    new_code = code
    for var_name in variable_names:
        if isinstance(var_name, tuple):
            var_name = var_name[-1]  # Get the actual variable name from tuple
        
        if len(var_name) > 3 and random.random() < 0.3:
            # Only rename reasonably sized variable names
            new_name = var_name + '_mod'
            # Use word boundary to avoid partial replacements
            new_code = re.sub(r'\b' + re.escape(var_name) + r'\b', new_name, new_code)
    
    return new_code

def code_truncation(code):
    """Truncate code to create a partial sample"""
    lines = code.split('\n')
    if len(lines) < 10:
        return code  # Too short to truncate
    
    # Keep a random portion of the code (40-80%)
    keep_percentage = random.uniform(0.4, 0.8)
    num_lines = int(len(lines) * keep_percentage)
    
    start_idx = random.randint(0, len(lines) - num_lines)
    truncated_lines = lines[start_idx:start_idx + num_lines]
    
    return '\n'.join(truncated_lines)

def combine_code_snippets(codes, language, max_snippets=3):
    """Combine 2-3 code snippets from the same language"""
    if len(codes) < 2:
        return None
    
    num_snippets = min(random.randint(2, max_snippets), len(codes))
    selected_snippets = random.sample(codes, num_snippets)
    
    # Language-specific connectors
    connectors = {
        'Python': '\n\n# New function\n',
        'JavaScript': '\n\n// New function\n',
        'Java': '\n\n// New method\n',
        'C++': '\n\n// New function\n',
        'PHP': '\n\n// New function\n',
        'TypeScript': '\n\n// New component\n',
    }
    
    connector = connectors.get(language, '\n\n')
    combined = connector.join(selected_snippets)
    
    return combined

def augment_language_data():
    """Augment language classification data"""
    all_samples_path = os.path.join(DATA_DIR, 'all_samples.csv')
    
    if not os.path.exists(all_samples_path):
        print("Original data file not found. Run data_collector.py first.")
        return
    
    print("Augmenting language classification data...")
    df = pd.read_csv(all_samples_path)
    
    augmented_samples = []
    languages = df['language'].unique()
    
    for language in tqdm(languages):
        language_df = df[df['language'] == language]
        original_codes = language_df['content'].tolist()
        
        # Apply each augmentation technique
        for code in original_codes:
            # Skip very short or very long code snippets
            if len(code) < 50 or len(code) > 20000:
                continue
                
            # Original sample
            augmented_samples.append({
                'language': language,
                'content': code,
                'augmentation': 'original'
            })
            
            # Comment modification
            modified_code = comment_modification(code, language)
            augmented_samples.append({
                'language': language,
                'content': modified_code,
                'augmentation': 'comment_mod'
            })
            
            # Whitespace modification
            modified_code = whitespace_modification(code)
            augmented_samples.append({
                'language': language,
                'content': modified_code,
                'augmentation': 'whitespace_mod'
            })
            
            # Variable renaming
            modified_code = variable_renaming(code, language)
            augmented_samples.append({
                'language': language,
                'content': modified_code,
                'augmentation': 'var_rename'
            })
            
            # Code truncation (only for longer snippets)
            if len(code.split('\n')) > 15:
                modified_code = code_truncation(code)
                augmented_samples.append({
                    'language': language,
                    'content': modified_code,
                    'augmentation': 'truncation'
                })
        
        # Combine snippets (create 5 combined samples per language)
        for _ in range(min(5, len(original_codes) // 3)):
            combined_code = combine_code_snippets(original_codes, language)
            if combined_code:
                augmented_samples.append({
                    'language': language,
                    'content': combined_code,
                    'augmentation': 'combined'
                })
    
    # Save augmented data
    augmented_df = pd.DataFrame(augmented_samples)
    augmented_df.to_csv(os.path.join(AUGMENTED_DIR, 'augmented_language_samples.csv'), index=False)
    print(f"Created {len(augmented_df)} augmented language samples")
    
    return augmented_df

def augment_purpose_data():
    """Augment code purpose classification data"""
    purpose_samples_path = os.path.join(DATA_DIR, 'purpose_samples.csv')
    
    if not os.path.exists(purpose_samples_path):
        print("Original purpose data file not found. Run data_collector.py first.")
        return
    
    print("Augmenting purpose classification data...")
    df = pd.read_csv(purpose_samples_path)
    
    augmented_samples = []
    purposes = df['purpose'].unique()
    
    for purpose in tqdm(purposes):
        purpose_df = df[df['purpose'] == purpose]
        language_groups = purpose_df.groupby('language')
        
        for language, group in language_groups:
            original_codes = group['content'].tolist()
            
            # Skip if too few samples
            if len(original_codes) < 2:
                continue
                
            for code in original_codes:
                # Skip very short or very long code snippets
                if len(code) < 50 or len(code) > 10000:
                    continue
                    
                # Original sample
                augmented_samples.append({
                    'purpose': purpose,
                    'language': language,
                    'content': code,
                    'augmentation': 'original'
                })
                
                # Comment modification
                modified_code = comment_modification(code, language)
                augmented_samples.append({
                    'purpose': purpose,
                    'language': language,
                    'content': modified_code,
                    'augmentation': 'comment_mod'
                })
                
                # Whitespace modification
                modified_code = whitespace_modification(code)
                augmented_samples.append({
                    'purpose': purpose,
                    'language': language,
                    'content': modified_code,
                    'augmentation': 'whitespace_mod'
                })
                
                # Variable renaming
                modified_code = variable_renaming(code, language)
                augmented_samples.append({
                    'purpose': purpose,
                    'language': language,
                    'content': modified_code,
                    'augmentation': 'var_rename'
                })
            
            # Combine snippets (create 3 combined samples per purpose/language)
            for _ in range(min(3, len(original_codes) // 2)):
                combined_code = combine_code_snippets(original_codes, language, max_snippets=2)
                if combined_code:
                    augmented_samples.append({
                        'purpose': purpose,
                        'language': language,
                        'content': combined_code,
                        'augmentation': 'combined'
                    })
    
    # Save augmented data
    augmented_df = pd.DataFrame(augmented_samples)
    augmented_df.to_csv(os.path.join(AUGMENTED_DIR, 'augmented_purpose_samples.csv'), index=False)
    print(f"Created {len(augmented_df)} augmented purpose samples")
    
    return augmented_df

if __name__ == "__main__":
    print("Starting data augmentation...")
    augment_language_data()
    augment_purpose_data()
    print("Data augmentation complete.") 