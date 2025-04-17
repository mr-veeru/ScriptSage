import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile

# Update the import paths to match the new structure
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.language_analyzer import analyze_code, predict_language, train_models, rebuild_models
from core.ast_analyzer import parse_code_structure, calculate_complexity_metrics, AST_PARSING_AVAILABLE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze code endpoint"""
    data = request.json
    
    if not data or 'code' not in data:
        return jsonify({'error': 'No code provided'}), 400
    
    code = data['code']
    filename = data.get('filename', '')
    
    # First, detect the language
    language = predict_language(code)
    
    # Then analyze the code
    analysis_result = analyze_code(code)
    
    # Add AST analysis if available
    if AST_PARSING_AVAILABLE:
        try:
            ast_structure = parse_code_structure(code, language)
            complexity_metrics = calculate_complexity_metrics(ast_structure)
            
            analysis_result['ast_analysis'] = {
                'structure': {
                    'functions': len(ast_structure.get('ast', {}).get('functions', [])) if 'ast' in ast_structure else 0,
                    'classes': len(ast_structure.get('ast', {}).get('classes', [])) if 'ast' in ast_structure else 0,
                    'imports': len(ast_structure.get('ast', {}).get('imports', [])) if 'ast' in ast_structure else 0
                },
                'complexity': complexity_metrics
            }
        except Exception as e:
            logger.error(f"Error during AST analysis: {e}")
            analysis_result['ast_analysis'] = {
                'error': 'Failed to perform AST analysis',
                'detail': str(e)
            }
    
    response = {
        'language': language,
        'filename': filename,
        'analysis': analysis_result
    }
    
    return jsonify(response)

@app.route('/api/analyze-files', methods=['POST'])
def analyze_files():
    """Analyze uploaded files endpoint"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    uploaded_files = request.files.getlist('files')
    
    if not uploaded_files or len(uploaded_files) == 0:
        return jsonify({'error': 'No files uploaded'}), 400
    
    results = []
    
    for file in uploaded_files:
        try:
            # Create a temporary file to save the uploaded content
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                file.save(temp.name)
                with open(temp.name, 'r', encoding='utf-8', errors='replace') as f:
                    code = f.read()
                
            # Clean up the temporary file
            os.unlink(temp.name)
            
            # Skip empty files
            if not code or len(code.strip()) == 0:
                results.append({
                    'filename': file.filename,
                    'error': 'Empty file'
                })
                continue
            
            # First, detect the language
            language = predict_language(code)
            
            # Then analyze the code
            analysis_result = analyze_code(code)
            
            # Add AST analysis if available
            if AST_PARSING_AVAILABLE:
                try:
                    ast_structure = parse_code_structure(code, language)
                    complexity_metrics = calculate_complexity_metrics(ast_structure)
                    
                    analysis_result['ast_analysis'] = {
                        'structure': {
                            'functions': len(ast_structure.get('ast', {}).get('functions', [])) if 'ast' in ast_structure else 0,
                            'classes': len(ast_structure.get('ast', {}).get('classes', [])) if 'ast' in ast_structure else 0,
                            'imports': len(ast_structure.get('ast', {}).get('imports', [])) if 'ast' in ast_structure else 0
                        },
                        'complexity': complexity_metrics
                    }
                except Exception as e:
                    logger.error(f"Error during AST analysis: {e}")
                    analysis_result['ast_analysis'] = {
                        'error': 'Failed to perform AST analysis',
                        'detail': str(e)
                    }
            
            # Add to results
            results.append({
                'filename': file.filename,
                'language': language,
                'analysis': analysis_result
            })
            
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return jsonify(results)

@app.route('/api/languages', methods=['GET'])
def get_languages():
    """Get supported languages"""
    languages = [
        'Python', 'JavaScript', 'TypeScript', 'HTML', 'CSS', 
        'Java', 'C/C++', 'C#', 'Ruby', 'Go', 'Rust', 'PHP',
        'JSON', 'YAML', 'Shell/Bash', 'SQL', 'Markdown', 'XML'
    ]
    return jsonify(languages)

@app.route('/api/train', methods=['POST'])
def train():
    """Train ML models endpoint"""
    force_retrain = request.json.get('force', False) if request.json else False
    train_models(force=force_retrain)
    return jsonify({'status': 'success', 'message': 'Models trained successfully'})

@app.route('/api/rebuild', methods=['POST'])
def rebuild():
    """Rebuild ML models endpoint"""
    rebuild_models()
    return jsonify({'status': 'success', 'message': 'Models rebuilt successfully'})

@app.route('/', methods=['GET'])
def home():
    """Home page"""
    return jsonify({
        'name': 'ScriptSage API',
        'version': '1.0.0',
        'description': 'API for the ScriptSage language detection and analysis engine',
        'endpoints': [
            '/api/analyze',
            '/api/languages',
            '/api/train',
            '/api/rebuild'
        ]
    })

if __name__ == '__main__':
    app.run(debug=True) 