# 🧙‍♂️ ScriptSage

<div align="center">
  
  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
  [![Machine Learning](https://img.shields.io/badge/ML-Powered-orange?style=for-the-badge&logo=tensorflow)](https://scikit-learn.org/)
  [![Code Analysis](https://img.shields.io/badge/Code-Analysis-green?style=for-the-badge&logo=code)](https://github.com)
  
  **Automated Code Understanding and Documentation System**
</div>

---

## 🔮 Overview

ScriptSage is an intelligent code analysis and documentation tool that automatically understands and documents your codebase. Using advanced machine learning models, it analyzes code patterns, identifies purposes, and generates comprehensive documentation - saving developers countless hours of manual documentation work.

<details>
<summary>✨ <b>Key Benefits</b></summary>
<br>

- **Time Saving**: Automate what would take hours manually
- **Consistency**: Generate standardized documentation
- **Insight**: Discover patterns in your codebase
- **Onboarding**: Help new developers understand the codebase faster
- **Maintenance**: Keep documentation in sync with code changes

</details>

## 🌟 Features

- **🔍 Intelligent Code Analysis**: Automatically detects code patterns and purposes
- **🌐 Multi-language Support**: Works with various programming languages
- **🧩 Purpose Classification**: Identifies code segments as:
  - 🧮 Algorithm Implementations
  - 🖌️ UI Components
  - 🎨 Styling
  - 🔌 API Endpoints
  - 📊 Data Processing
  - 🛠️ Utility Functions
  - 💾 Database Operations
  - 🧪 Testing Code
- **📝 Automated Documentation**: Generates clear and structured documentation
- **⚙️ Customizable Analysis**: Configurable parameters for different codebases

## 🚀 Quick Start

1. **Installation**
   ```bash
   git clone https://github.com/yourusername/scriptsage.git
   cd scriptsage
   pip install -r backend/requirements.txt
   ```

2. **Configuration**
   ```bash
   cp backend/.env.example backend/.env
   # Edit .env with your configuration
   ```

3. **Run the Application**
   ```bash
   python backend/run.py
   ```

## 📂 Project Structure

```
scriptsage/
├── backend/                  # Backend server and ML components
│   ├── api/                  # REST API endpoints
│   ├── core/                 # Core functionality
│   │   ├── language_analyzer.py    # Language detection logic
│   │   ├── train_models.py         # Model training code
│   │   └── data_preprocessing.py   # Data processing utilities
│   ├── data/                 # Training and test data
│   │   ├── raw/              # Raw collected code samples
│   │   └── processed/        # Preprocessed datasets
│   ├── models/               # Trained ML models
│   │   ├── language_classifier.pkl  # Language classification model
│   │   └── purpose_classifier.pkl   # Purpose classification model
│   ├── tests/                # Test suite
│   │   ├── test_api.py       # API tests
│   │   └── fix_purpose_classifier.py # Tests for purpose classifier
│   └── run.py                # Main entry point
└── frontend/                 # Web interface (if applicable)
```

## 🔄 Data Pipeline

### 📥 Data Collection

ScriptSage uses a comprehensive pipeline to collect code samples:

1. **Source Diversity**: Collects code from various repositories
2. **Language Coverage**: Ensures samples from multiple programming languages
3. **Purpose Variety**: Gathers examples of different code purposes
4. **Quality Filtering**: Removes low-quality or duplicate samples
5. **Metadata Tagging**: Preserves information about language, source, and initial purpose

**Commands:**

```bash
# Collect samples from GitHub repositories
python backend/scripts/collect_samples.py --source github --languages python,javascript,typescript --max-samples 5000

# Collect samples from local codebase
python backend/scripts/collect_samples.py --source local --path /path/to/your/codebase --output backend/data/raw

# Collect samples from public datasets
python backend/scripts/collect_samples.py --source datasets --datasets rosetta-code,code-jam --output backend/data/raw
```

Sample data collection workflow:
```python
# Sample data collection workflow
def collect_samples(sources, max_samples=10000):
    samples = []
    for source in sources:
        raw_samples = fetch_from_source(source)
        filtered_samples = filter_quality_samples(raw_samples)
        samples.extend(filtered_samples[:max_samples // len(sources)])
    return samples
```

### 🔧 Data Preprocessing

Before training, data undergoes several preprocessing steps:

1. **Cleaning**: Removes comments, extra whitespace, and irrelevant content
2. **Tokenization**: Breaks code into meaningful tokens
3. **Feature Extraction**: Converts code to numerical features
4. **Augmentation**: Generates additional training examples for rare classes
5. **Splitting**: Creates training, validation, and test sets

**Commands:**

```bash
# Preprocess all collected samples
python backend/scripts/preprocess_data.py --input backend/data/raw --output backend/data/processed

# Clean and normalize code samples
python backend/scripts/preprocess_data.py --clean-only --input backend/data/raw --output backend/data/cleaned

# Generate augmented samples for underrepresented classes
python backend/scripts/augment_data.py --input backend/data/processed --output backend/data/augmented

# Split data into training and testing sets
python backend/scripts/split_data.py --input backend/data/processed --output backend/data/split --test-size 0.2
```

Data preprocessing workflow:
```python
# Data preprocessing workflow
def preprocess_samples(samples):
    cleaned = [clean_code(sample) for sample in samples]
    augmented = augment_rare_classes(cleaned)
    return split_data(augmented)
```

### 🧠 Model Training

ScriptSage trains two primary models:

1. **Language Classifier**:
   - RandomForestClassifier with TF-IDF features
   - Character-level n-grams for language pattern recognition
   - Optimized hyperparameters via GridSearchCV

2. **Purpose Classifier**:
   - GradientBoostingClassifier with TF-IDF features
   - Word-level n-grams for semantic understanding
   - Pattern-based feature enrichment
   - Cross-validation to ensure generalization

**Commands:**

```bash
# Train all models with default settings
python backend/core/train_models.py

# Train only the language classifier with hyperparameter optimization
python backend/core/train_models.py --model language --optimize

# Train only the purpose classifier with custom settings
python backend/core/train_models.py --model purpose --max-features 30000 --n-estimators 300

# Train with a specific dataset
python backend/core/train_models.py --data backend/data/processed/custom_dataset.json
```

Model training with hyperparameter optimization:
```python
# Model training with hyperparameter optimization
def train_purpose_classifier(X, y):
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=20000)),
        ('classifier', GradientBoostingClassifier())
    ])
    
    param_grid = {
        'vectorizer__ngram_range': [(1, 2), (1, 3), (1, 4)],
        'classifier__n_estimators': [100, 200, 300]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X, y)
    return grid_search.best_estimator_
```

### 🧪 Testing and Evaluation

Both models undergo rigorous evaluation:

1. **Cross-Validation**: 5-fold cross-validation for robust metrics
2. **Accuracy Measurement**: Overall classification accuracy
3. **Per-Class Metrics**: Precision, recall, and F1-score for each class
4. **Confusion Matrix**: Detailed analysis of classification errors
5. **Real-World Testing**: Evaluation on unseen production code

**Commands:**

```bash
# Evaluate all models on test data
python backend/scripts/evaluate_models.py

# Evaluate specific model with detailed metrics
python backend/scripts/evaluate_models.py --model purpose --detailed

# Generate confusion matrix visualization
python backend/scripts/evaluate_models.py --model language --confusion-matrix --output plots/confusion_matrix.png

# Run cross-validation on models
python backend/scripts/cross_validate.py --model purpose --folds 5 --iterations 3

# Test models on custom dataset
python backend/scripts/evaluate_models.py --data path/to/custom/test/data.json
```

Example evaluation script:
```python
# Evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }
```

## 📊 Model Performance

The system uses advanced machine learning techniques:

- **Gradient Boosting Classifier** for purpose detection
  - 200 estimators for ensemble strength
  - Learning rate of 0.1 for optimal convergence
  - Max depth of 5 to prevent overfitting

- **TF-IDF Vectorization** with optimized parameters
  - 20,000 max features for comprehensive vocabulary
  - Word n-grams from 1 to 4 to capture phrases
  - Minimum document frequency of 2 to filter rare tokens
  - English stop words removal for cleaner features

- **Cross-validation** for robust evaluation
  - 5-fold stratified splits
  - Preserved class distribution
  - Multiple random seeds for stability

- **Hyperparameter Optimization**
  - Grid search over key parameters
  - Parallel processing for efficiency
  - Best parameter selection based on accuracy

## ⚙️ Configuration

Key configuration options in `backend/.env`:

```env
# Model settings
MODEL_PATH=./models
MAX_FEATURES=20000
NGRAM_RANGE=1,4

# Training settings
OPTIMIZE_HYPERPARAMS=True
CROSS_VALIDATION_FOLDS=5
TEST_SIZE=0.2

# Application settings
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=5000
```

## 📈 Future Improvements

ScriptSage is continuously evolving with planned enhancements:

- **Deep Learning Models**: Transformer-based code understanding
- **Multilingual Support**: Expanded language coverage
- **Interactive Documentation**: Dynamic exploration of code relationships
- **IDE Integration**: Real-time analysis within development environments
- **Code Generation**: Automated documentation string creation

## 🔗 Support

For support, please:
1. Check the documentation
2. Open an issue
3. Contact the maintainer 