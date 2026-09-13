# IQUANA backend

The REST + WebSocket API behind [IQUANA](https://github.com/Iquana-tool/iquana-tool) —
**I**ntelligent **QU**antification, **AN**notation and **A**nalysis, a tool for AI-assisted
segmentation, annotation and quantification of scientific image datasets, built at
[DFKI](https://www.dfki.de/).

This service owns the database, the files on disk and the permission model. It runs **no
models itself** — every AI task is delegated over HTTP to the separate
[ai-service](https://github.com/Iquana-tool/ai-service).

- **User documentation:** https://iquana-tool.github.io/docs/
- **Installing the whole tool:** do not clone this repo by hand — run the
  [installer](https://github.com/Iquana-tool/iquana-tool), which sets up every component and
  wires their configuration together.
- **Issues:** all IQUANA bug reports and feature requests go to
  [iquana-tool/issues](https://github.com/Iquana-tool/iquana-tool/issues/new/choose).

---

## What it does

| Area | Summary |
|---|---|
| **Datasets** | Upload, thumbnails, per-image metadata keys, annotation progress, COCO and CSV/JSON export |
| **Label space** | Arbitrarily deep label hierarchy, built manually or drafted by an LLM from a prose description |
| **Annotation** | Live annotation sessions over a WebSocket; contours, nesting, edit history, undo |
| **AI orchestration** | Prompted segmentation, instance suggestion, instance segmentation, cross-image suggestion — each proxied to the ai-service |
| **Batch inference** | Dataset-wide runs planned per label and executed by a Celery worker |
| **Review and correction** | Review verdicts, review/correction queues with pluggable sort strategies, rejection reasons |
| **Calibration** | Pixel scale and colour calibration (line-drawn or greyscale card), stored per image with dataset defaults |
| **Quantification** | Tiered metric registry (geometry, appearance, relational, contextual), quantification profiles, per-contour and per-image tables |
| **Access control** | JWT auth, global roles, per-dataset roles, invites, per-member permission overrides |
| **Embeddings** | pgvector store of DINOv3 descriptors for cross-image exemplar retrieval |

---

## Project structure

```
app/
├── __init__.py            # FastAPI app factory: middleware + router registration
├── database/              # SQLAlchemy models — one module per table group
├── schemas/               # Pydantic request/response models local to this service
│                          #   (shared ones live in iquana-toolbox)
├── routes/
│   ├── general/           # CRUD and domain endpoints (datasets, images, contours,
│   │                      #   labels, masks, reviews, queues, calibration, auth,
│   │                      #   admin, members, instance, status, ...)
│   ├── services/          # Thin proxies to the ai-service, one router per task
│   └── websockets/        # Live annotation session (image_annotation_session.py)
└── services/              # Business logic, disconnected from routing
    ├── ai_services/       # HTTP clients for each ai-service task surface
    ├── annotation_session/# Session state machine and operations
    ├── calibration/       # Calibration strategies, cards, registry, store
    ├── database_access/   # Query and persistence helpers per entity
    ├── inference/         # Batch-inference planning, contracts, execution, Celery tasks
    └── *.py               # quantification, permissions, review/annotation queues,
                           #   mlflow, model registry, embeddings, redis, ...
config.py                  # Environment-driven settings (paths, URLs, secrets)
main.py                    # Entry point — builds the app and configures logging
scripts/                   # One-off migrations and backfills
tests/                     # pytest suite
data/                      # Runtime data (gitignored)
├── datasets/<name>/       #   uploaded images and masks per dataset
└── thumbnails/            #   low-resolution copies for fast gallery loading
```

### Request flow

A **route** validates its input with a **schema**, then calls a **service**. The service
talks to the **database** (SQLAlchemy) and/or the **ai-service** (HTTP), and its result is
serialised back through a schema. Routes contain no business logic; services never touch
FastAPI.

---

## Setup

Dependencies are managed with **[uv](https://docs.astral.sh/uv/)** — `pyproject.toml` and
`uv.lock` are the source of truth, and the virtualenv has no `pip`. Add packages with
`uv add <pkg>`, not `pip install`. (`requirements.txt` still exists but is legacy and is
not what installs the environment.)

### Local development

```bash
uv sync
cp env.example .env
```

The database and MLflow run as containers:

```bash
docker compose up -d postgres mlflow
```

Then start the API:

```bash
uv run fastapi dev main.py
```

> API at http://127.0.0.1:8000, interactive OpenAPI docs at http://127.0.0.1:8000/docs

Batch inference additionally needs the Celery worker, on its **own** queue. This must not
be Celery's default queue — the backend and the ai-service are two apps sharing one Redis
broker, and the ai-service worker consumes the default queue as a fallback, so tasks
published there can be picked up by a worker that has them unregistered and silently
discards them (see the comment in `app/services/celery_app.py`):

```bash
uv run celery -A app.services.celery_app worker -Q backend.jobs --loglevel=info
```

On Windows the prefork pool does not work at all; add `--pool=solo`.

### Docker Compose

`docker-compose.yml` brings up Postgres (`pgvector/pgvector:pg16`), MLflow and the backend:

```bash
docker compose up --build
```

Note that the `iquana-backend` service bind-mounts `./data` at an absolute *host* path, so
that the file paths stored in the database resolve both inside the container and from the
ai-service, which runs natively on the host. Adjust that path before using the backend
service on your machine.

---

## Configuration

Everything is read from the environment (via `.env`). `env.example` carries the annotated
list and `config.py` the defaults. The ones that matter most:

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | sqlite in `data/` | Postgres in any real deployment: `postgresql+psycopg://...` |
| `REDIS_URL` | `redis://localhost:6379` | Celery broker and result backend |
| `MLFLOW_URL` | `http://localhost:5000` | Model and experiment tracking |
| `AI_SERVICE_URL` | `http://localhost:8004` | The unified ai-service. Per-task URLs are derived as `<AI_SERVICE_URL>/<task>` |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | CORS allow-list |
| `SECRET_KEY` | — | JWT signing key. **Set this.** |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `480` | There is no refresh flow, so this is the whole session |
| `INSTANCE_NAME` / `_ORG` / `_CONTACT` / `_NOTICE` | empty | Shown on the sign-in page, served from `GET /instance/` |
| `INSTANCE_ALLOW_REGISTRATION` | `false` | Self-registration, enforced by the API rather than merely hidden in the UI |
| `LABEL_SPACE_LLM_MODEL` / `_API_KEY` | — | LiteLLM `<provider>/<model>`; the label-space assistant stays disabled until the key is set |
| `EMBEDDING_LIFECYCLE_ENABLED` | `false` | Opt-in on-write embedding; otherwise the store is filled by `scripts/backfill_embeddings.py` |

Per-task overrides (`PROMPTED_SEGMENTATION_BACKEND_URL`, `SUGGESTION_SEGMENTATION_BACKEND_URL`,
`INSTANCE_SEGMENTATION_BACKEND_URL`) exist so a single task can be routed to a satellite
service whose dependencies cannot share the unified environment.

### Instance identity lives here, not in the frontend

The sign-in page reads the instance name, organisation, contact and registration policy
from `GET /instance/` at runtime. These are backend variables on purpose: the registration
policy has to be enforced by the API — a sign-in page that merely hides the link is not a
closed door — and Vite would otherwise bake the strings into the bundle at build time, so
fixing a typo would mean a rebuild. The *first* account can always be created regardless,
so a fresh installation is never locked out of itself.

---

## Access control

Two independent levels, defined in `app/schemas/permissions.py`:

- **Global role** (on the user account) — `admin`, `member`, `guest`. Answers *what may this
  account do to the platform?*
- **Dataset role** (on the membership row) — `viewer` < `annotator` < `reviewer` < `curator`
  < `owner`. Datasets are IQUANA's unit of governance, so nearly every permission lives here.

Roles are named bundles of `Permission` values. A membership row can also carry
`extra_permissions` / `denied_permissions`, so a single collaborator can be given or refused
one capability without inventing a new role.

Routes enforce this with the `require()` / `require_global()` dependencies, which take the
name of the id parameter to walk up from (contour → mask → image → dataset). When the
dataset id is only known after the body is parsed, handlers call `ensure_permission()`
instead.

The matrix deliberately lives here rather than in `iquana-toolbox`: the toolbox is consumed
as a git-pinned dependency, so keeping it local means a permission change does not require a
toolbox release and re-pin.

---

## Tests

```bash
uv run pytest tests/ -q
```

`app/database/__init__.py` calls `create_all` **at import time**, so with Postgres down the
suite fails during *collection* with connection errors, which looks like a code break but is
not. Run against sqlite instead, using an absolute forward-slash path:

```bash
DATABASE_URL="sqlite:////absolute/path/to/backend/data/test_scratch.db" uv run pytest tests/ -q
```

`.env` also sets `DATABASE_URL`, but `load_dotenv()` does not override a real environment
variable, so the shell value wins. Delete the scratch database afterwards.

---

## Scripts

| Script | Purpose |
|---|---|
| `backfill_contour_metrics.py` | Populate the tall `contour_metrics` table for existing contours |
| `backfill_embeddings.py` | Fill the pgvector store from the ai-service embed surface |
| `clear_stuck_inference_jobs.py` | Reset inference jobs left marked `running` after a crash |
| `copy_sqlite_to_postgres.py` | Migrate a local sqlite database into Postgres |
| `migrate_calibrations.py`, `migrate_response_calibration.py` | Move older scale data onto the calibration model |
| `migrate_roles.py` | Backfill the role columns for pre-RBAC installations |

---

## Related repositories

| Repo | Role |
|---|---|
| [iquana-tool](https://github.com/Iquana-tool/iquana-tool) | Installer, launcher and the issue tracker for all of IQUANA |
| [frontend-react](https://github.com/Iquana-tool/frontend-react) | The web UI |
| [ai-service](https://github.com/Iquana-tool/ai-service) | Model inference and training |
| [iquana-toolbox](https://github.com/Iquana-tool/iquana-toolbox) | Shared Pydantic schemas, metric registry and MLflow helpers |

---

## License

AGPL-3.0 — see [LICENSE](LICENSE).
