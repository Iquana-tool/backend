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
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///" + os.path.join(DATA_DIR, "database.db"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")
# AI services: all tasks are served by the unified ai-service, one surface per
# task under a URL prefix (/prompted-segmentation, /instance-suggestion,
# /instance-segmentation). Each task client points at "<AI_SERVICE_URL>/<task>",
# so its existing relative calls (/inference, /annotation_session/run, /health,
# /models, preload) resolve against the right surface unchanged.
AI_SERVICE_URL = os.environ.get("AI_SERVICE_URL", "http://localhost:8004")
# Per-task overrides still win when set explicitly, so a single task can be routed
# to a separate (satellite) service -- e.g. a model whose dependencies cannot
# share the unified service's environment.
PROMPTED_SEGMENTATION_BACKEND_URL = os.environ.get(
    "PROMPTED_SEGMENTATION_BACKEND_URL", f"{AI_SERVICE_URL}/prompted-segmentation")
SUGGESTION_SEGMENTATION_BACKEND_URL = os.environ.get(
    "SUGGESTION_SEGMENTATION_BACKEND_URL", f"{AI_SERVICE_URL}/instance-suggestion")
INSTANCE_SEGMENTATION_BACKEND_URL = os.environ.get(
    "INSTANCE_SEGMENTATION_BACKEND_URL", f"{AI_SERVICE_URL}/instance-segmentation")
# Retired (semantic-seg-service is no longer part of the tool); kept only so any
# lingering import does not break. Do not wire new code to it.
SEMANTIC_SEGMENTATION_BACKEND_URL = os.environ.get("SEMANTIC_SEGMENTATION_BACKEND_URL")
# Embedding (DINOv3) surface for cross-image exemplar retrieval.
EMBED_BACKEND_URL = os.environ.get("EMBED_BACKEND_URL", f"{AI_SERVICE_URL}/embed")
# The registered embedder model (ai-service registry key) that precomputes embeddings.
EMBEDDING_MODEL_KEY = os.environ.get("EMBEDDING_MODEL_KEY", "dinov3")
# On-write embedding is opt-in: when False, image/contour writes never enqueue embedding
# work, and the store is populated only by scripts/backfill_embeddings.py. Turn on once the
# ai-service embed surface and a Celery worker are reachable and validated.
EMBEDDING_LIFECYCLE_ENABLED = os.environ.get(
    "EMBEDDING_LIFECYCLE_ENABLED", "false"
).lower() in ("1", "true", "yes", "on")
# The concrete backbone id the store's embeddings were computed with. Retrieval filters the
# store by this, so it MUST match the model_id the embedder returns (DINOv3 ViT-B/16 default).
EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "facebook/dinov3-vitb16-pretrain-lvd1689m")
# Cross-image concept suggestion surface + the model (ai-service registry key) that serves it.
CROSS_IMAGE_BACKEND_URL = os.environ.get("CROSS_IMAGE_BACKEND_URL", f"{AI_SERVICE_URL}/cross-image-suggestion")
CROSS_IMAGE_MODEL_KEY = os.environ.get("CROSS_IMAGE_MODEL_KEY", "sam3")
SECRET_KEY = os.environ.get("SECRET_KEY", "supersecretkey")
# How long a login stays valid. There is no refresh flow -- the frontend drops the
# token and logs the user out on the first 401 -- so this is the whole session, and
# a short value interrupts annotation work mid-task. Defaults to a working day.
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))

# LLM-assisted label-space generation (provider-agnostic via LiteLLM).
# LABEL_SPACE_LLM_MODEL uses LiteLLM's "<provider>/<model>" naming, e.g.
# "anthropic/claude-opus-4-8", "openai/gpt-4o", "gemini/gemini-1.5-pro", "ollama/llama3".
# Generation is disabled until LABEL_SPACE_LLM_API_KEY is set.
LABEL_SPACE_LLM_MODEL = os.environ.get("LABEL_SPACE_LLM_MODEL", "anthropic/claude-opus-4-8")
LABEL_SPACE_LLM_API_KEY = os.environ.get("LABEL_SPACE_LLM_API_KEY")
LABEL_SPACE_LLM_API_BASE = os.environ.get("LABEL_SPACE_LLM_API_BASE")  # optional: self-hosted / Azure / Ollama
