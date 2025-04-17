import os
import logging
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Any, Tuple, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flag for Tree-sitter availability - Set to False to disable tree-sitter completely
TREE_SITTER_AVAILABLE = False
AST_PARSING_AVAILABLE = True

# We'll skip the tree-sitter import and just use Python's built-in parsers
# Don't log this as a warning since it's intentional
logger.info("Using Python's built-in AST parser")

# Use only the built-in ast module
import ast

# Dictionary to store loaded parsers
parsers = {}

# Initialize with a minimal set of functionality - we'll focus on Python AST first
def initialize_tree_sitter():
    """Initialize Tree-sitter languages and parsers if available"""
    # Tree-sitter is disabled
    return False

def parse_code_with_tree_sitter(code: str, language: str) -> Optional[Dict]:
    """Parse code using Tree-sitter and extract structural information"""
    # Tree-sitter is disabled
    return None

def parse_code_with_ast(code: str) -> Optional[Dict]:
    """Parse Python code using the built-in ast module
    
    Args:
        code: Python code string
    
    Returns:
        Dictionary with parsed AST information or None if parsing failed
    """
    try:
        tree = ast.parse(code)
        
        # Extract basic information
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'start_line': node.lineno,
                    'end_line': getattr(node, 'end_lineno', node.lineno)
                })
        
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        methods.append(child.name)
                
                classes.append({
                    'name': node.name,
                    'methods': methods,
                    'start_line': node.lineno,
                    'end_line': getattr(node, 'end_lineno', node.lineno)
                })
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append({
                        'name': name.name,
                        'alias': name.asname,
                        'line': node.lineno
                    })
            elif isinstance(node, ast.ImportFrom):
                for name in node.names:
                    imports.append({
                        'name': f"{node.module}.{name.name}" if node.module else name.name,
                        'alias': name.asname,
                        'line': node.lineno
                    })
        
        # Count control structures for complexity metrics
        control_structures = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                control_structures += 1
        
        # Extract variables and assignments
        variables = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.append({
                            'name': target.id,
                            'line': node.lineno
                        })
        
        return {
            'functions': functions,
            'classes': classes,
            'imports': imports,
            'variables': variables,
            'control_structures': control_structures,
            'node_count': sum(1 for _ in ast.walk(tree)),
            'line_count': max([getattr(node, 'end_lineno', 0) for node in ast.walk(tree)] or [0])
        }
    except SyntaxError as e:
        logger.error(f"Syntax error parsing Python code: {e}")
        return None
    except Exception as e:
        logger.error(f"Error parsing Python code with ast: {e}")
        return None

# Main function for code parsing using AST
def parse_code_structure(code: str, language: str) -> Dict:
    """Parse code structure using the most appropriate AST parser for the language
    
    Args:
        code: The code string to parse
        language: The programming language of the code
    
    Returns:
        Dictionary with parsed structure information
    """
    results = {}
    
    # For Python code, use built-in parsers
    if language == 'Python':
        try:
            ast_results = parse_code_with_ast(code)
            if ast_results:
                results["ast"] = ast_results
        except Exception:
            logger.warning("Failed to parse with ast module", exc_info=True)
    else:
        # For non-Python languages, provide a basic fallback
        results["basic"] = {
            "language": language,
            "line_count": len(code.splitlines()),
            "character_count": len(code),
            "note": "Detailed parsing not available for this language yet"
        }
    
    return results

# Function to extract structural complexity metrics
def calculate_complexity_metrics(code_structure: Dict) -> Dict:
    """Calculate complexity metrics based on code structure
    
    Args:
        code_structure: Dictionary with parsed code structure
    
    Returns:
        Dictionary with complexity metrics
    """
    metrics = {
        'node_count': 0,
        'max_depth': 0,
        'function_count': 0,
        'class_count': 0,
        'import_count': 0,
        'cyclomatic_complexity': 1  # Base complexity
    }
    
    if not code_structure:
        return metrics
    
    # Extract metrics from AST parsing
    if 'ast' in code_structure:
        ast_results = code_structure.get('ast', {})
        metrics['function_count'] = len(ast_results.get('functions', []))
        metrics['class_count'] = len(ast_results.get('classes', []))
        metrics['import_count'] = len(ast_results.get('imports', []))
        metrics['node_count'] = ast_results.get('node_count', 0)
        
        # Calculate cyclomatic complexity
        metrics['cyclomatic_complexity'] += ast_results.get('control_structures', 0)
    
    # If we don't have AST metrics but have basic metrics
    if 'basic' in code_structure and 'function_count' not in metrics:
        # Provide some basic metrics
        basic = code_structure.get('basic', {})
        metrics['node_count'] = basic.get('character_count', 0)
        metrics['max_depth'] = 0
        metrics['cyclomatic_complexity'] = 1
    
    return metrics 