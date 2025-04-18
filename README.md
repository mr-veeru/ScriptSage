# 🧙‍♂️ ScriptSage

> **Your intelligent code analysis companion**

![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![React](https://img.shields.io/badge/react-18.2.0-61DAFB.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## ✨ Overview

ScriptSage is an advanced code analysis tool that uses machine learning to understand, evaluate, and summarize code across multiple programming languages. Drop in your code and let ScriptSage work its magic!

<div align="center">
  <img src="https://via.placeholder.com/800x450?text=ScriptSage+Screenshot" alt="ScriptSage Screenshot" width="80%" />
</div>

## 🚀 Features

- **Multi-language Support** - Intelligently detects and analyzes Python, JavaScript, TypeScript, HTML, CSS, and more
- **Code Purpose Detection** - Identifies whether code is for data processing, web UI, API endpoints, etc.
- **Smart Summarization** - Generates concise, human-readable summaries of code functionality
- **Complexity Analysis** - Evaluates code structure and complexity metrics
- **Modern UI** - Beautiful, responsive interface for uploading and analyzing code
- **Advanced ML Models** - Leverages CodeBERT and other state-of-the-art models for deeper insights

## 🔧 Tech Stack

### Backend
- **Python** - Core application logic
- **Flask** - RESTful API server
- **scikit-learn** - Machine learning algorithms
- **Pygments** - Syntax highlighting and language detection
- **AST Parsing** - Code structure analysis

### Frontend
- **React** - UI components and state management
- **TypeScript** - Type-safe code
- **Material UI** - Modern component library
- **react-textarea-code-editor** - Code editing with syntax highlighting

## 📊 Data Pipeline

### Data Collection

ScriptSage's ML models require diverse code samples to train effectively. There are multiple ways to collect training data:

1. **Automated GitHub Collection**
   ```bash
   cd backend
   # Set GitHub token for higher rate limits (optional)
   export GITHUB_TOKEN=your_token_here
   
   # Collect samples for all supported languages
   python -m core.data_collection
   
   # Collect samples for specific languages (edit the script parameters)
   python -c "from core.data_collection import fetch_all_language_samples; fetch_all_language_samples(languages=['Python', 'JavaScript'], samples_per_language=30)"
   ```

2. **Manual Collection**
   - Place code samples in the appropriate directories:
     ```
     backend/data/
     ├── Python/
     │   ├── web_ui/
     │   ├── data_processing/
     │   └── api/
     ├── JavaScript/
     │   ├── web_ui/
     │   └── utility/
     └── ...
     ```

3. **Stack Overflow Collection**
   ```bash
   python -c "from core.data_collection import fetch_stackoverflow_samples; fetch_stackoverflow_samples('Python', num_samples=20)"
   ```

### Data Cleaning and Preprocessing

ScriptSage preprocesses raw code samples to prepare them for training:

1. **Clean Code Samples**
   ```bash
   # Preprocess all collected samples
   python -m core.data_preprocessing
   ```

2. **Data Augmentation**
   ```bash
   # Augment existing samples to improve model training
   python -c "from core.data_preprocessing import augment_training_data; augment_training_data()"
   ```

3. **Manual Cleaning**
   - Remove large comment blocks
   - Remove binary/non-text content
   - Ensure proper file extensions
   - Verify language labels

### Model Training

Train the ML models used by ScriptSage for code analysis:

1. **Train All Models**
   ```bash
   cd backend
   python -m core.train_models
   ```

2. **Train Language Classifier Only**
   ```bash
   python -c "from core.train_models import train_language_classifier; train_language_classifier(samples, optimize=True)"
   ```

3. **Train Purpose Classifier Only**
   ```bash
   python -c "from core.train_models import train_purpose_classifier; train_purpose_classifier(samples, optimize=True)"
   ```

### Model Testing and Evaluation

Evaluate the performance of your trained models:

1. **Test Language Classifier**
   ```bash
   python -c "from core.language_analyzer import predict_language; print(predict_language('def hello(): print(\"Hello, world!\")'))"
   ```

2. **Test Purpose Classifier**
   ```bash
   python -c "from core.language_analyzer import predict_purpose; print(predict_purpose('function fetchData() { return fetch(\"/api/data\").then(res => res.json()); }'))"
   ```

3. **Comprehensive Model Evaluation**
   ```bash
   python -m core.evaluate_models
   ```

4. **Test on Sample Files**
   ```bash
   python -c "from core.language_analyzer import analyze_code; print(analyze_code(open('path/to/sample.py').read()))"
   ```

5. **Run Dedicated Test Suite**
   ```bash
   # Run all tests
   cd backend
   python tests/run_all_tests.py
   
   # Run specific test modules
   python tests/test_ml.py     # Test ML models
   python tests/test_ast.py    # Test AST analysis
   python tests/test_api.py    # Test API endpoints
   ```

The test suite includes:
- `test_ml.py`: Tests language and purpose detection accuracy
- `test_ast.py`: Tests code structure parsing and complexity metrics
- `test_api.py`: Tests the backend API endpoints and integration

## 🏁 Running the Project

Get ScriptSage up and running:

### Quick Start

The easiest way to get ScriptSage running is to use the quickstart script:

```bash
# Clone the repository
git clone https://github.com/yourusername/scriptsage.git
cd scriptsage

# Run the quickstart script with frontend and server
python quickstart.py --with-frontend --start-server
```

### Manual Setup

#### Backend

```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Start the server
python run.py
```

#### Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

### Using ScriptSage

1. Open your browser to http://localhost:3000
2. Upload or paste your code
3. View the analysis results
4. Explore the visualizations and insights

### Troubleshooting

- **Import Errors**: Ensure your Python path includes the backend directory
  ```bash
  # In Windows
  set PYTHONPATH=%PYTHONPATH%;C:\path\to\scriptsage\backend
  
  # In Linux/Mac
  export PYTHONPATH=$PYTHONPATH:/path/to/scriptsage/backend
  ```

- **Model Loading Errors**: Ensure models are trained before starting the server
  ```bash
  cd backend
  python core/train_models.py
  ```

- **Frontend Connection Issues**: Check that backend API is running and CORS is enabled
  ```bash
  # Test backend API
  curl http://localhost:5000/api/health
  ```

## 🧠 How It Works

1. **Language Detection** - ScriptSage first analyzes your code to determine its programming language using pattern matching and machine learning.

2. **Purpose Identification** - The system then classifies the code's purpose (e.g., utility function, UI component, data processing).

3. **Structure Analysis** - AST (Abstract Syntax Tree) parsing extracts the code's structure, patterns, and complexity metrics.

4. **Advanced Processing** - For deeper insights, ScriptSage can use transformers-based models like CodeBERT to understand semantic meaning.

5. **Result Generation** - The analysis is compiled into a comprehensive report with helpful visualizations and suggestions.

## 📁 Project Structure

```
scriptsage/
├── backend/
│   ├── api/                  # Flask API endpoints
│   ├── core/                 # Code analysis engine
│   │   ├── language_analyzer.py  # Main analysis logic
│   │   ├── ast_analyzer.py   # Code structure analysis
│   │   ├── data_collection.py    # Data gathering
│   │   ├── data_preprocessing.py # Data cleaning and augmentation
│   │   ├── train_models.py   # Model training
│   │   ├── evaluate_models.py    # Model evaluation
│   │   └── predict.py        # Inference and prediction
│   ├── data/                 # Training data
│   ├── models/               # ML model storage
│   ├── tests/                # Backend tests
│   └── logs/                 # Application logs
├── frontend/
│   ├── public/               # Static assets
│   └── src/                  # React components
│       ├── components/       # UI components
│       └── App.tsx           # Main application
└── quickstart.py             # Setup automation
```

## 🔮 Future Plans

- **Code Suggestion** - AI-powered recommendations for code improvements
- **More Languages** - Expanded support for additional programming languages
- **Integration** - VS Code extension and Git integration
- **Custom Models** - Train models on your own codebase for personalized insights

## 👨‍💻 Author

This project was created and developed entirely by a single developer as a personal project.

## 🙏 Acknowledgements

- [CodeBERT](https://github.com/microsoft/CodeBERT) by Microsoft Research
- [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) for robust parsing
- All the open-source dependencies that made this project possible

---

<div align="center">
  <p>Made with ❤️ for developers who appreciate good code</p>
</div> 