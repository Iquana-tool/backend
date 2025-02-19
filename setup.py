import os
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_project():
    # Get the project root directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Define required directories
    directories = {
        "database": os.path.join(BASE_DIR, "database"),
        "uploads": os.path.join(BASE_DIR, "uploads"),
        "output_masks": os.path.join(BASE_DIR, "output_masks"),
        "selected_masks": os.path.join(BASE_DIR, "selected_masks"),
        "polyps_masks": os.path.join(BASE_DIR, "polyps_masks"),
        "fine_tune_masks": os.path.join(BASE_DIR, "fine_tune_masks")
    }
    
    # Create directories
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
        
        # Ensure directory has correct permissions
        os.chmod(dir_path, 0o755)
        logger.info(f"Set permissions for: {dir_path}")
    
    logger.info("Project setup complete!")

if __name__ == "__main__":
    setup_project()