#!/usr/bin/env python
"""
Main entry point for the ScriptSage application.
This script starts the Flask API server.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'logs', 'scriptsage.log'), 'a')
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs(os.path.join(os.path.dirname(__file__), 'logs'), exist_ok=True)

# Ensure model and data directories exist in the backend root directory
os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'data', 'augmented'), exist_ok=True)

def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask server"""
    from api.app import app
    
    logger.info("Starting ScriptSage API server...")
    logger.info(f"Server running at http://{host}:{port}")
    
    try:
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return False
    
    return True

def initialize():
    """Initialize the application"""
    # Load environment variables
    load_dotenv()
    
    # Set environment-specific configurations
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    host = os.environ.get('HOST', '0.0.0.0')
    
    # Check if the port is specified and valid
    try:
        port = int(os.environ.get('PORT', 5000))
    except ValueError:
        logger.warning("Invalid PORT environment variable, using default port 5000")
        port = 5000
    
    return host, port, debug

def main():
    """Main entry point for the application"""
    logger.info("Initializing ScriptSage application...")
    
    try:
        host, port, debug = initialize()
        success = run_server(host, port, debug)
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main()) 