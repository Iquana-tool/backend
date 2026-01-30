import os
from dotenv import load_dotenv

load_dotenv()

LOGS_DIR = os.getenv("LOGS_DIR", "./logs")

DATA_DIR = os.getenv("DATA_DIR", "./data")
DATABASE_FILE = os.getenv("DATABASE_FILE", "./data/database.db")
DATASETS_DIR = os.getenv("DATASETS_DIR", os.path.join(DATA_DIR, "datasets"))
THUMBNAILS_DIR = os.getenv("THUMBNAILS_DIR", os.path.join(DATA_DIR, "thumbnails"))

SEMANTIC_SEGMENTATION_BACKEND_URL = os.environ.get("SEMANTIC_SEGMENTATION_BACKEND_URL")
PROMPTED_SEGMENTATION_BACKEND_URL = os.environ.get("PROMPTED_SEGMENTATION_BACKEND_URL")
COMPLETION_SEGMENTATION_BACKEND_URL = os.environ.get("COMPLETION_SEGMENTATION_BACKEND_URL")
SECRET_KEY = os.environ.get("SECRET_KEY", "supersecretkey")
