import os
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.getenv("LOGS_DIR", os.path.join(ROOT_DIR, "logs"))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(ROOT_DIR, "data"))
DATABASE_FILE = os.getenv("DATABASE_FILE", os.path.join(DATA_DIR, "database.db"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DATASETS_DIR = os.getenv("DATASETS_DIR", os.path.join(DATA_DIR, "datasets"))
THUMBNAILS_DIR = os.getenv("THUMBNAILS_DIR", os.path.join(DATA_DIR, "thumbnails"))
SEMANTIC_SEGMENTATION_BACKEND_URL = os.environ.get("SEMANTIC_SEGMENTATION_BACKEND_URL")
PROMPTED_SEGMENTATION_BACKEND_URL = os.environ.get("PROMPTED_SEGMENTATION_BACKEND_URL")
COMPLETION_SEGMENTATION_BACKEND_URL = os.environ.get("COMPLETION_SEGMENTATION_BACKEND_URL")
SECRET_KEY = os.environ.get("SECRET_KEY", "supersecretkey")
