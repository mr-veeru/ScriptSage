#!/usr/bin/env python
"""
ScriptSage Prediction Script
---------------------------
This script provides a simple interface for running predictions on code samples.
"""

import os
import sys
import argparse
import logging
import time

# Import prediction functions
from language_analyzer import (
    analyze_code
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_from_file(file_path):
    """Analyze a code file and print the results"""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        return predict_from_code(code, file_path)
    
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return False

def predict_from_code(code, source=None):
    """Analyze a code string and print the results"""
    logger.info(f"Analyzing {'code from ' + source if source else 'provided code'}...")
    
    try:
        # Get full analysis
        analysis = analyze_code(code)
        
        if not analysis:
            logger.error("Failed to analyze code")
            return False
        
        # Print the results
        print("\n" + "="*50)
        print("ScriptSage Code Analysis Report")
        print("="*50)
        
        print(f"\nLanguage: {analysis.get('language', 'Unknown')}")
        print(f"Purpose: {analysis.get('purpose', 'Unknown')}")
        print(f"\nComplexity Metrics:")
        
        complexity = analysis.get('complexity', {})
        for metric, value in complexity.items():
            print(f"  - {metric}: {value}")
        
        print("\nSummary:")
        print(analysis.get('summary', 'No summary available'))
        
        print("\n" + "="*50)
        
        return analysis
    
    except Exception as e:
        logger.error(f"Error analyzing code: {e}")
        return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="ScriptSage Code Prediction Tool")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', '-f', help='Path to a code file to analyze')
    group.add_argument('--code', '-c', help='Code string to analyze')
    
    parser.add_argument('--output', '-o', help='Path to save the analysis results as JSON')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    start_time = time.time()
    
    if args.file:
        result = predict_from_file(args.file)
    else:
        result = predict_from_code(args.code)
    
    elapsed = time.time() - start_time
    logger.info(f"Analysis completed in {elapsed:.2f} seconds")
    
    # Save results if output path is specified
    if result and args.output:
        import json
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Analysis results saved to {args.output}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main()) 