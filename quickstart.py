#!/usr/bin/env python
"""
ScriptSage Quickstart Script
----------------------------
This script automates the setup process for ScriptSage by:
1. Checking prerequisites
2. Installing dependencies
3. Setting up pre-trained models
4. Configuring the environment
5. Starting the backend server (optional)
6. Setting up the frontend (optional)
"""

import os
import sys
import subprocess
import platform
import argparse
import shutil
import json
from pathlib import Path
import time
import urllib.request
import zipfile
import tempfile

# Parse arguments at the beginning
parser = argparse.ArgumentParser(description="ScriptSage Quickstart Script")
parser.add_argument("--with-frontend", action="store_true", help="Set up the frontend along with the backend")
parser.add_argument("--start-server", action="store_true", help="Start the servers after setup")
parser.add_argument("--offline", action="store_true", help="Use bundled models without downloading")
args = parser.parse_args()

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_step(step_num, total_steps, message):
    """Print a formatted step message"""
    print(f"\n{Colors.BLUE}[{step_num}/{total_steps}] {Colors.BOLD}{message}{Colors.ENDC}")

def print_success(message):
    """Print a success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")

def print_warning(message):
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")

def print_error(message):
    """Print an error message"""
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")

def check_prerequisites():
    """Check if all prerequisites are installed"""
    total_steps = 5
    print_step(1, total_steps, "Checking prerequisites...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print_error(f"Python 3.8+ required. Found: {python_version.major}.{python_version.minor}")
        return False
    print_success(f"Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    
    # Check pip
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, stdout=subprocess.PIPE)
        print_success("pip is installed")
    except subprocess.CalledProcessError:
        print_error("pip is not installed")
        return False
    
    # Check Node.js if frontend setup is requested
    if args.with_frontend:
        try:
            node_version = subprocess.run(["node", "--version"], check=True, stdout=subprocess.PIPE)
            node_version = node_version.stdout.decode().strip()
            print_success(f"Node.js {node_version} detected")
            
            npm_version = subprocess.run(["npm", "--version"], check=True, stdout=subprocess.PIPE)
            npm_version = npm_version.stdout.decode().strip()
            print_success(f"npm {npm_version} detected")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print_warning("Node.js or npm not found. Frontend setup will be skipped.")
            args.with_frontend = False
    
    return True

def get_python_executable():
    """Get the path to the Python executable"""
    return sys.executable

def install_dependencies():
    """Install Python dependencies"""
    total_steps = 5
    print_step(2, total_steps, "Installing dependencies...")
    
    python_exe = get_python_executable()
    requirements_path = os.path.join(os.getcwd(), "backend", "requirements.txt")
    
    try:
        subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        print_success("pip upgraded")
        
        subprocess.run([python_exe, "-m", "pip", "install", "-r", requirements_path], check=True)
        print_success("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False

def download_pretrained_models():
    """Download and set up pre-trained models"""
    total_steps = 5
    print_step(3, total_steps, "Setting up pre-trained models...")
    
    # Use the correct models directory - inside backend/core/models
    models_dir = os.path.join(os.getcwd(), "backend", "core", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if models are already downloaded
    if os.path.exists(os.path.join(models_dir, "language_classifier.joblib")):
        print_warning("Pre-trained models already exist")
        return True
    
    # In a real scenario, this would download from a GitHub release or similar
    # For this example, we'll simulate the process
    try:
        print("Downloading pre-trained models... (simulated)")
        time.sleep(2)  # Simulate download time
        
        # Create dummy model files for demonstration
        with open(os.path.join(models_dir, "language_classifier.joblib"), "w") as f:
            f.write("# This is a placeholder for the actual model")
        
        with open(os.path.join(models_dir, "purpose_classifier.joblib"), "w") as f:
            f.write("# This is a placeholder for the actual model")
        
        print_success("Pre-trained models installed successfully")
        return True
    except Exception as e:
        print_error(f"Failed to download models: {e}")
        return False

def configure_environment():
    """Configure environment variables"""
    total_steps = 5
    print_step(4, total_steps, "Configuring environment...")
    
    env_path = os.path.join(os.getcwd(), "backend", ".env")
    
    # Check if .env already exists
    if os.path.exists(env_path):
        print_warning(".env file already exists")
        return True
    
    try:
        with open(env_path, "w") as f:
            f.write("DEBUG=True\n")
            f.write("HOST=127.0.0.1\n")
            f.write("PORT=5000\n")
            f.write("USE_BUNDLED_MODELS=True\n")
            # Add other environment variables as needed
        
        print_success(".env file created successfully")
        return True
    except Exception as e:
        print_error(f"Failed to create .env file: {e}")
        return False

def start_backend():
    """Start the backend server"""
    if not args.start_server:
        return True
    
    total_steps = 5
    print_step(5, total_steps, "Starting backend server...")
    
    python_exe = get_python_executable()
    backend_script = os.path.join(os.getcwd(), "backend", "run.py")
    
    try:
        # Use subprocess.Popen to start the server in the background
        process = subprocess.Popen([python_exe, backend_script])
        
        # Wait a moment to see if the process stays running
        time.sleep(3)
        
        if process.poll() is None:
            print_success("Backend server started successfully")
            print(f"API is available at http://127.0.0.1:5000")
            return True
        else:
            print_error("Backend server failed to start")
            return False
    except Exception as e:
        print_error(f"Failed to start backend server: {e}")
        return False

def setup_frontend():
    """Set up the React frontend"""
    if not args.with_frontend:
        return True
    
    # Use the same step number as the backend server
    total_steps = 5
    print_step(5, total_steps, "Setting up frontend...")
    
    frontend_dir = os.path.join(os.getcwd(), "frontend")
    
    try:
        os.chdir(frontend_dir)
        
        # Install npm dependencies
        subprocess.run(["npm", "install"], check=True)
        print_success("Frontend dependencies installed successfully")
        
        if args.start_server:
            # Start the frontend development server
            subprocess.Popen(["npm", "start"])
            print_success("Frontend development server started")
            print(f"Frontend is available at http://localhost:3000")
        
        # Return to the original directory
        os.chdir(os.path.dirname(frontend_dir))
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to set up frontend: {e}")
        return False
    except Exception as e:
        print_error(f"Error during frontend setup: {e}")
        return False

def main():
    """Main execution flow"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}🧙‍♂️ ScriptSage Quickstart{Colors.ENDC}")
    print("============================\n")
    
    # Run all setup steps
    steps = [
        check_prerequisites,
        install_dependencies,
        download_pretrained_models,
        configure_environment
    ]
    
    if args.start_server:
        steps.append(start_backend)
        if args.with_frontend:
            steps.append(setup_frontend)
    
    for step_func in steps:
        if not step_func():
            print_error("Setup failed. Please fix the issues and try again.")
            return 1
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ScriptSage setup completed successfully!{Colors.ENDC}")
    
    if not args.start_server:
        print("\nTo start the backend server:")
        print("  cd backend")
        print("  python run.py")
    
    if args.with_frontend and not args.start_server:
        print("\nTo start the frontend development server:")
        print("  cd frontend")
        print("  npm start")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 