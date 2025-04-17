# 🧙‍♂️ ScriptSage

<div align="center">

![ScriptSage Logo](https://img.shields.io/badge/🧙‍♂️-ScriptSage-6434eb?style=for-the-badge)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18.x-blue?style=flat&logo=react)](https://reactjs.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-lightgrey?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](https://opensource.org/licenses/MIT)

**The magical code analysis engine that deciphers your code's language, purpose, and structure**

</div>

## ✨ What is ScriptSage?

ScriptSage is a powerful code analysis engine that uses machine learning to understand source code on multiple levels:

- 🔍 **Detect programming languages** automatically across 15+ languages
- 🧠 **Analyze code purpose and functionality** without executing it
- 📊 **Measure complexity and structure** using abstract syntax tree analysis
- 🌐 **Provide a REST API** for seamless integration with other tools

Perfect for developers, educators, and tools that need to understand code without running it.

## 🚀 Key Features

- **Language Detection:** Instantly identifies programming languages through pattern recognition and ML models
- **Code Purpose Classification:** Determines what your code is trying to accomplish 
- **AST-Based Analysis:** Deep structural analysis for Python files
- **Multi-Language Support:** Basic structural analysis for 15+ languages
- **User-Friendly API:** Simple REST API for integration with your existing tools
- **Modern UI:** Clean interface for code submission and analysis visualization

## 📋 Supported Languages

<div align="center">

| Language | Detection | Basic Analysis | AST Analysis |
|:--------:|:---------:|:--------------:|:------------:|
| Python   |     ✅    |       ✅       |      ✅      |
| JavaScript |   ✅    |       ✅       |      ⚠️      |
| Java     |     ✅    |       ✅       |      ⚠️      |
| C/C++    |     ✅    |       ✅       |      ⚠️      |
| Go       |     ✅    |       ✅       |      ⚠️      |
| Ruby     |     ✅    |       ✅       |      ⚠️      |
| PHP      |     ✅    |       ✅       |      ⚠️      |
| Swift    |     ✅    |       ✅       |      ⚠️      |
| Kotlin   |     ✅    |       ✅       |      ⚠️      |
| + More   |     ✅    |       ✅       |      ⚠️      |

</div>

## 🔧 Quick Setup

### Prerequisites

- Python 3.8+ 
- Node.js 16+ (for frontend)
- npm or yarn

### Setup with Quickstart Script

The easiest way to set up ScriptSage is using our quickstart script:

```bash
# Run the quickstart script
python quickstart.py --no-venv --with-frontend --start-server
```

This will:
1. Check your system prerequisites
2. Install all dependencies globally
3. Download pre-trained models
4. Configure the environment
5. Start the backend and frontend servers

### Manual Setup

#### Backend Setup

```bash
# Clone repository
git clone https://github.com/yourusername/ScriptSage.git
cd ScriptSage

# Install dependencies globally
pip install -r backend/requirements.txt

# Start the API server
cd backend
python run.py
```

#### Frontend Setup

```bash
# In a new terminal
cd ScriptSage/frontend

# Install dependencies
npm install

# Start development server
npm start
```

Visit `http://localhost:3000` to use the application.

## 🔮 How It Works

1. **Data Collection:** ScriptSage trains on a diverse set of code samples from GitHub
2. **Feature Extraction:** Extracts linguistic and structural features from code
3. **Machine Learning:** Uses ML models to classify and analyze code
4. **AST Analysis:** For supported languages, builds and analyzes abstract syntax trees
5. **API Delivery:** Presents results through a clean REST API

## 📚 API Documentation

### Core Endpoints

- `GET /api/languages` - List all supported languages
- `POST /api/analyze` - Analyze code snippet or file
- `POST /api/train` - Retrain models with new data
- `POST /api/rebuild` - Rebuild models from scratch

### Example Usage

```bash
# Analyze Python code
curl -X POST http://127.0.0.1:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"code": "def hello(): print(\"Hello, world!\")"}'
```

Response:
```json
{
  "language": "Python",
  "confidence": 0.98,
  "purpose": "Output function",
  "complexity": "Simple",
  "structure": {
    "functions": 1,
    "classes": 0,
    "statements": 1,
    "imports": 0
  }
}
```

## 🧪 Running Tests

```bash
cd backend/tests
python run_all_tests.py
```

## 📁 Project Structure

```
ScriptSage/
├── backend/                 # Python Flask API
│   ├── api/                 # API endpoints
│   ├── core/                # Core analysis engine
│   ├── data/                # Training data
│   ├── models/              # Trained ML models
│   ├── tests/               # Test modules
│   ├── utils/               # Utility scripts
│   └── logs/                # Application logs
├── frontend/                # React-based UI
│   ├── public/              # Static assets
│   └── src/                 # React components
└── docs/                    # Documentation
```

## 🚧 Challenges & Solutions

ScriptSage acknowledges several challenges in code analysis and addresses them with practical solutions:

### AST Analysis Limitations
- **Problem**: Full AST analysis currently limited to Python
- **Solution**: Modular architecture allows adding new language parsers; check our [contribution guidelines](CONTRIBUTING.md) to help expand language support

### Training Data Requirements
- **Problem**: Initial setup requires GitHub data collection
- **Solution**: Pre-trained models now included in releases; offline mode available with `--use-bundled-models` flag

### Resource Optimization
- **Problem**: ML libraries can be resource-intensive
- **Solution**: Optimized model inference with reduced memory footprint; lazy-loading for less frequently used models

### Simplified Setup
- **Problem**: Complex multi-step setup process
- **Solution**: New `quickstart.py` script automates the entire setup process

### Accuracy Enhancements
- **Problem**: Language detection accuracy varies
- **Solution**: Confidence scoring system helps identify uncertain results (threshold configurable); continuous model improvement with active learning

### API Rate Limits
- **Problem**: GitHub API limitations during data collection
- **Solution**: Built-in rate limiting and retry mechanisms; local caching of previously downloaded samples

### Model Freshness
- **Problem**: Models become outdated as languages evolve
- **Solution**: Scheduled model updates available through `cron` job; incremental learning to update models without full retraining

### Error Handling
- **Problem**: Limited documentation on error cases
- **Solution**: Comprehensive error catalog now in docs/errors.md; graceful degradation when components fail

## 🤝 Contributing

Contributions are welcome! Check out our [contributing guide](CONTRIBUTING.md) to get started.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  Made with ❤️ by the ScriptSage Team
</div> 