#!/usr/bin/env python
"""
ScriptSage Model Updater
-----------------------
A utility for updating ML models with new data without full retraining.
This script supports:
1. Incremental model updates (partial training)
2. Model version management
3. Scheduled updates
4. Performance benchmarking

Usage:
    python model_updater.py --mode=incremental
    python model_updater.py --mode=full
    python model_updater.py --schedule=daily
"""

import os
import sys
import argparse
import logging
import time
import datetime
import json
import joblib
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'logs', 
                'model_updater.log'
            ), 
            'a'
        )
    ]
)
logger = logging.getLogger(__name__)

# Make sure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from core
try:
    from core.model_trainer import train_language_model, train_purpose_model
except ImportError:
    logger.error("Failed to import training modules. Check your directory structure.")
    sys.exit(1)

class ModelUpdater:
    """Class for handling model updates and version control"""
    
    def __init__(self, models_dir=None, data_dir=None):
        """Initialize the model updater"""
        # Set default paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = models_dir or os.path.join(self.base_dir, 'models')
        self.data_dir = data_dir or os.path.join(self.base_dir, 'data')
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'new_samples'), exist_ok=True)
        
        # Track model versions
        self.versions_file = os.path.join(self.models_dir, 'versions.json')
        self.load_versions()
    
    def load_versions(self):
        """Load model version information"""
        if os.path.exists(self.versions_file):
            try:
                with open(self.versions_file, 'r') as f:
                    self.versions = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Invalid versions file. Creating new one.")
                self.versions = self._create_default_versions()
        else:
            logger.info("No versions file found. Creating new one.")
            self.versions = self._create_default_versions()
            self._save_versions()
    
    def _create_default_versions(self):
        """Create default version information"""
        return {
            "language_detector": {
                "version": "0.1.0",
                "last_updated": datetime.datetime.now().isoformat(),
                "samples_trained_on": 0,
                "performance": {}
            },
            "purpose_classifier": {
                "version": "0.1.0",
                "last_updated": datetime.datetime.now().isoformat(),
                "samples_trained_on": 0,
                "performance": {}
            }
        }
    
    def _save_versions(self):
        """Save version information to file"""
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def has_new_data(self):
        """Check if there's new data for training"""
        new_samples_dir = os.path.join(self.data_dir, 'new_samples')
        return len(os.listdir(new_samples_dir)) > 0
    
    def backup_models(self):
        """Create backups of current models"""
        backup_dir = os.path.join(self.models_dir, 'backups', 
                                 datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(backup_dir, exist_ok=True)
        
        for model_name in ['language_detector', 'purpose_classifier']:
            model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
            if os.path.exists(model_path):
                backup_path = os.path.join(backup_dir, f"{model_name}.joblib")
                shutil.copy2(model_path, backup_path)
                logger.info(f"Backed up {model_name} to {backup_path}")
    
    def update_incrementally(self):
        """Update models incrementally with new data"""
        logger.info("Starting incremental model update")
        
        # Backup current models
        self.backup_models()
        
        # New samples directory
        new_samples_dir = os.path.join(self.data_dir, 'new_samples')
        if not os.listdir(new_samples_dir):
            logger.info("No new samples found for incremental training")
            return False
        
        # Language detection model update
        lang_model_path = os.path.join(self.models_dir, "language_detector.joblib")
        if os.path.exists(lang_model_path):
            try:
                # Load existing model
                model = joblib.load(lang_model_path)
                
                # Update with new data (implementation would be in model_trainer.py)
                success = train_language_model(
                    model=model,
                    new_data_dir=new_samples_dir,
                    incremental=True
                )
                
                if success:
                    # Update version information
                    new_version = self._increment_version(self.versions["language_detector"]["version"])
                    self.versions["language_detector"]["version"] = new_version
                    self.versions["language_detector"]["last_updated"] = datetime.datetime.now().isoformat()
                    # Count new samples
                    new_samples = len(os.listdir(new_samples_dir))
                    self.versions["language_detector"]["samples_trained_on"] += new_samples
                    logger.info(f"Language model updated to version {new_version}")
            except Exception as e:
                logger.error(f"Error updating language model: {e}")
                return False
        
        # Purpose classification model update
        # Similar logic to language model update
        
        # Save updated version information
        self._save_versions()
        
        # Archive new samples
        self._archive_new_samples()
        
        logger.info("Incremental model update completed successfully")
        return True
    
    def _increment_version(self, version_str):
        """Increment the patch version number"""
        major, minor, patch = version_str.split('.')
        return f"{major}.{minor}.{int(patch) + 1}"
    
    def _archive_new_samples(self):
        """Move new samples to archive after training"""
        new_samples_dir = os.path.join(self.data_dir, 'new_samples')
        archive_dir = os.path.join(
            self.data_dir, 
            'archive', 
            datetime.datetime.now().strftime('%Y%m%d')
        )
        os.makedirs(archive_dir, exist_ok=True)
        
        # Move all files
        for filename in os.listdir(new_samples_dir):
            source = os.path.join(new_samples_dir, filename)
            destination = os.path.join(archive_dir, filename)
            shutil.move(source, destination)
        
        logger.info(f"Archived new samples to {archive_dir}")
    
    def full_retrain(self):
        """Perform a full model retraining"""
        logger.info("Starting full model retraining")
        
        # Backup current models
        self.backup_models()
        
        # Train language model from scratch
        try:
            success = train_language_model(incremental=False)
            if success:
                # Update version information - increment minor version
                major, minor, patch = self.versions["language_detector"]["version"].split('.')
                new_version = f"{major}.{int(minor) + 1}.0"
                self.versions["language_detector"]["version"] = new_version
                self.versions["language_detector"]["last_updated"] = datetime.datetime.now().isoformat()
                logger.info(f"Language model retrained to version {new_version}")
        except Exception as e:
            logger.error(f"Error retraining language model: {e}")
            return False
        
        # Train purpose model from scratch
        # Similar logic to language model retraining
        
        # Save updated version information
        self._save_versions()
        
        logger.info("Full model retraining completed successfully")
        return True

def setup_scheduler(mode, interval):
    """Set up a scheduled job for model updates"""
    if platform.system() == "Windows":
        logger.info("Scheduled jobs on Windows require Task Scheduler. Please set up manually.")
        print("To create a scheduled task on Windows:")
        print(f"1. Open Task Scheduler")
        print(f"2. Create a Basic Task named 'ScriptSage Model Update'")
        print(f"3. Set trigger to {interval}")
        print(f"4. Set action to run: {sys.executable}")
        print(f"5. Add arguments: {os.path.abspath(__file__)} --mode={mode}")
    else:
        # Create a crontab entry
        command = f"{sys.executable} {os.path.abspath(__file__)} --mode={mode}"
        
        if interval == "daily":
            cron_time = "0 0 * * *"  # Midnight every day
        elif interval == "weekly":
            cron_time = "0 0 * * 0"  # Midnight every Sunday
        elif interval == "monthly":
            cron_time = "0 0 1 * *"  # Midnight on the first of each month
        
        crontab_line = f"{cron_time} {command} >> {os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'scheduled_update.log')} 2>&1"
        
        print("\nTo add this to your crontab, run:")
        print("crontab -e")
        print("And add the following line:")
        print(crontab_line)

def main():
    """Main entry point for the model updater"""
    parser = argparse.ArgumentParser(description="ScriptSage Model Updater")
    parser.add_argument("--mode", choices=["incremental", "full"], default="incremental",
                        help="Update mode: incremental (default) or full retraining")
    parser.add_argument("--schedule", choices=["daily", "weekly", "monthly"], 
                        help="Set up scheduled updates")
    args = parser.parse_args()
    
    # Set up scheduler if requested
    if args.schedule:
        import platform
        setup_scheduler(args.mode, args.schedule)
        return 0
    
    # Run model update
    updater = ModelUpdater()
    
    if args.mode == "incremental":
        if not updater.has_new_data():
            logger.info("No new data available for incremental update")
            return 0
        success = updater.update_incrementally()
    else:
        success = updater.full_retrain()
    
    return 0 if success else 1

if __name__ == "__main__":
    # Import modules that are only needed in main
    import shutil
    import platform
    
    sys.exit(main()) 