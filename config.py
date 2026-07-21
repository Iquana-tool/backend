import os
from dotenv import load_dotenv

load_dotenv()

# Directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.getenv("LOGS_DIR", os.path.join(ROOT_DIR, "logs"))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(ROOT_DIR, "data"))
DATASETS_DIR = os.getenv("DATASETS_DIR", os.path.join(DATA_DIR, "datasets"))
THUMBNAILS_DIR = os.getenv("THUMBNAILS_DIR", os.path.join(DATA_DIR, "thumbnails"))

# URLS <- probably should be replaced with editable YAML
DATABASE_URL = os.getenv("DATABASE_FILE", "sqlite:///" + os.path.join(DATA_DIR, "database.db"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")
SEMANTIC_SEGMENTATION_BACKEND_URL = os.environ.get("SEMANTIC_SEGMENTATION_BACKEND_URL")
PROMPTED_SEGMENTATION_BACKEND_URL = os.environ.get("PROMPTED_SEGMENTATION_BACKEND_URL")
SUGGESTION_SEGMENTATION_BACKEND_URL = os.environ.get("SUGGESTION_SEGMENTATION_BACKEND_URL")
INSTANCE_SEGMENTATION_BACKEND_URL = os.environ.get("INSTANCE_SEGMENTATION_BACKEND_URL")
SECRET_KEY = os.environ.get("SECRET_KEY", "supersecretkey")

# LLM-assisted label-space generation (provider-agnostic via LiteLLM).
# LABEL_SPACE_LLM_MODEL uses LiteLLM's "<provider>/<model>" naming, e.g.
# "anthropic/claude-opus-4-8", "openai/gpt-4o", "gemini/gemini-1.5-pro", "ollama/llama3".
# Generation is disabled until LABEL_SPACE_LLM_API_KEY is set.
LABEL_SPACE_LLM_MODEL = os.environ.get("LABEL_SPACE_LLM_MODEL", "anthropic/claude-opus-4-8")
LABEL_SPACE_LLM_API_KEY = os.environ.get("LABEL_SPACE_LLM_API_KEY")
LABEL_SPACE_LLM_API_BASE = os.environ.get("LABEL_SPACE_LLM_API_BASE")  # optional: self-hosted / Azure / Ollama
