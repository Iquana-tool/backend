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
# Upgrade a PostgreSQL database to the newest Alembic revision on boot. Turn off to
# run `alembic upgrade head` by hand instead (e.g. to take a backup first); the
# backend then only warns when the schema is behind the code.
AUTO_MIGRATE = os.getenv("IQUANA_AUTO_MIGRATE", "true").lower() in ("1", "true", "yes", "on")
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

# --- Dataset ZIP Archive Limits (Issue #94 - Provisional Defaults) ------------
# Provisional defaults pending representative user-study dataset evidence.
# The approval gate for production limits remains open.
#
# Note on upload exhaustion: HTTP multipart parsers spool incoming request
# bodies before route handlers run. DATASET_ARCHIVE_MAX_COMPRESSED_BYTES is a
# secondary application-level check on the staged archive; frontline upload
# exhaustion protection MUST be configured at the reverse proxy / ASGI layer
# (e.g. Nginx client_max_body_size).
# Maximum compressed upload bytes (provisional default: 10 GiB).
DATASET_ARCHIVE_MAX_COMPRESSED_BYTES = int(
    os.environ.get("DATASET_ARCHIVE_MAX_COMPRESSED_BYTES", str(10 * 1024 * 1024 * 1024))
)
# Maximum total declared uncompressed bytes across all members (provisional default: 20 GiB).
DATASET_ARCHIVE_MAX_UNCOMPRESSED_BYTES = int(
    os.environ.get("DATASET_ARCHIVE_MAX_UNCOMPRESSED_BYTES", str(20 * 1024 * 1024 * 1024))
)
# Maximum number of entries/members in a dataset ZIP archive (provisional default: 50,000).
DATASET_ARCHIVE_MAX_MEMBERS = int(
    os.environ.get("DATASET_ARCHIVE_MAX_MEMBERS", "50000")
)
# Maximum uncompressed bytes for annotations.json or config.json (provisional default: 64 MiB).
DATASET_ARCHIVE_MAX_CONTROL_JSON_BYTES = int(
    os.environ.get("DATASET_ARCHIVE_MAX_CONTROL_JSON_BYTES", str(64 * 1024 * 1024))
)
# Maximum uncompressed size for any individual ZIP member (provisional default: 2 GiB).
DATASET_ARCHIVE_MAX_MEMBER_BYTES = int(
    os.environ.get("DATASET_ARCHIVE_MAX_MEMBER_BYTES", str(2 * 1024 * 1024 * 1024))
)

def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# -- User event capture (user studies) ------------------------------------
# Two levels of the same gate, not two kinds of data: nothing is recorded unless
# both are true.
#
# USER_EVENTS_ENABLED is the deployment-level lock, read once at boot: when it is
# false no middleware is installed, the /activity-log routes are not mounted, and
# nothing -- including an admin account -- can switch capture on at runtime.
# Changing it needs a restart. The USER_EVENTS_* values below are only *boot
# defaults*: once the lock is open an admin can change them live over
# PUT /activity-log/config, and those choices persist in `activity_log_settings`.
USER_EVENTS_ENABLED = _flag("USER_EVENTS_ENABLED", False)
# Whether capture is actually running. Separate from the lock so that starting and
# stopping a study is a routine admin action, while deciding a deployment may
# collect at all stays with whoever controls the deploy config.
USER_EVENTS_CAPTURE = _flag("USER_EVENTS_CAPTURE", False)
#: Individually switchable capture components. See app/services/activity_log/config.py.
USER_EVENTS_COMPONENTS = os.environ.get(
    "USER_EVENTS_COMPONENTS", "annotation,ai,navigation,api")
# Client batching hints, served to the frontend by GET /activity-log/config.
USER_EVENTS_FLUSH_INTERVAL_MS = int(os.environ.get("USER_EVENTS_FLUSH_INTERVAL_MS", "5000"))
USER_EVENTS_BATCH_SIZE = int(os.environ.get("USER_EVENTS_BATCH_SIZE", "50"))
# Hard caps applied server-side regardless of what a client sends.
USER_EVENTS_MAX_PAYLOAD_BYTES = int(os.environ.get("USER_EVENTS_MAX_PAYLOAD_BYTES", "4096"))
USER_EVENTS_MAX_BATCH = int(os.environ.get("USER_EVENTS_MAX_BATCH", "200"))
# How many events the in-process queue holds before it starts dropping. Activity log
# must never be the reason an annotation request blocks, so the queue is bounded
# and overflow is counted rather than awaited.
USER_EVENTS_QUEUE_SIZE = int(os.environ.get("USER_EVENTS_QUEUE_SIZE", "10000"))
# No pointer, key or wheel input on an open image for this long counts as idle.
# Boot default only; an admin can change it live over PUT /activity-log/config.
USER_EVENTS_IDLE_THRESHOLD_MS = int(os.environ.get("USER_EVENTS_IDLE_THRESHOLD_MS", "60000"))
