"""Runtime configuration an operator can change without a redeploy.

The values here were previously environment-only: correcting a mistyped contact
address or rotating an API key meant editing ``.env`` and restarting the process.
That is fine for a value chosen once at install time and wrong for a credential,
which is rotated on somebody else's schedule.

The resolution order is **database row, then environment, then the declared
default**. Environment stays the default rather than the authority, so a
deployment that never opens the admin page behaves exactly as it did before this
module existed, and an override is always visibly an override.

Two properties are deliberate:

* **Only registered keys are writable.** :data:`SETTINGS` is the whole editable
  surface; an unknown key is refused rather than stored, so this cannot grow into
  a second, unreviewed configuration channel next to ``config.py``.
* **Secrets are never read back.** A secret setting reports only whether it is
  set and the last four characters, which is enough to tell two keys apart and
  not enough to use one.

Reads go to the database on every call rather than through a cache. The read is a
single indexed query over a table with a handful of rows, and the alternative --
a process-local cache -- would go stale the moment the deployment runs more than
one worker, which is a worse failure than a cheap query.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from logging import getLogger

from sqlalchemy.orm import Session

from app.database import get_context_session
from app.database.instance_settings import InstanceSettings

logger = getLogger(__name__)

#: Settings that live on this backend and take effect on the next call.
SCOPE_BACKEND = "backend"
#: Settings the AI service consumes. Stored here because this is where the admin
#: surface and the database are, and pushed across on save -- see
#: ``app.services.ai_services.ai_config``.
SCOPE_AI_SERVICE = "ai-service"

KIND_TEXT = "text"
KIND_SECRET = "secret"
KIND_BOOL = "bool"


@dataclass(frozen=True)
class SettingSpec:
    """One editable setting: where it comes from, and how to show it."""

    #: Stable identifier used by the API and the ``instance_settings`` row.
    key: str
    #: The environment variable holding the default (and, for AI-service scoped
    #: settings, the variable that is set on the far side when pushed).
    env_var: str
    #: Grouping for the admin page.
    group: str
    label: str
    description: str
    kind: str = KIND_TEXT
    #: Used when neither a row nor the environment supplies a value.
    default: str = ""
    scope: str = SCOPE_BACKEND
    #: Shown under the field as an example, never as a value.
    placeholder: str = ""


#: The complete editable surface. Adding a setting here is what makes it
#: reachable from the admin page; nothing else needs to change.
SETTINGS: tuple[SettingSpec, ...] = (
    # -- AI credentials ----------------------------------------------------
    SettingSpec(
        key="hf_token",
        env_var="HF_ACCESS_TOKEN",
        group="ai",
        label="Hugging Face access token",
        description=(
            "Used by the AI service to download gated model weights (SAM, DINOv3). "
            "Leave empty for anonymous downloads, which works for ungated models only."
        ),
        kind=KIND_SECRET,
        scope=SCOPE_AI_SERVICE,
        placeholder="hf_...",
    ),
    SettingSpec(
        key="llm_api_key",
        env_var="LABEL_SPACE_LLM_API_KEY",
        group="ai",
        label="LLM API key",
        description=(
            "Key for the provider named by the model id below. The label-space "
            "assistant stays disabled -- and label editing stays manual -- until this is set."
        ),
        kind=KIND_SECRET,
        placeholder="sk-...",
    ),
    SettingSpec(
        key="llm_model",
        env_var="LABEL_SPACE_LLM_MODEL",
        group="ai",
        label="LLM model",
        description=(
            "LiteLLM model id in \"<provider>/<model>\" form. The provider prefix decides "
            "which key is expected above."
        ),
        default="anthropic/claude-opus-4-8",
        placeholder="anthropic/claude-opus-4-8",
    ),
    SettingSpec(
        key="llm_api_base",
        env_var="LABEL_SPACE_LLM_API_BASE",
        group="ai",
        label="LLM API base URL",
        description="Only for self-hosted, Azure or Ollama deployments. Leave empty otherwise.",
        placeholder="http://localhost:11434",
    ),
    # -- This instance -----------------------------------------------------
    SettingSpec(
        key="instance_name",
        env_var="INSTANCE_NAME",
        group="instance",
        label="Instance name",
        description="Shown on the sign-in page as \"Welcome to ...\".",
        placeholder="HIFMB Reef Lab",
    ),
    SettingSpec(
        key="instance_org",
        env_var="INSTANCE_ORG",
        group="instance",
        label="Organisation",
        description=(
            "Shown as \"... hosted by ...\", so phrase it to follow that -- including the article."
        ),
        placeholder="the Helmholtz Institute for Functional Marine Biodiversity",
    ),
    SettingSpec(
        key="instance_contact",
        env_var="INSTANCE_CONTACT",
        group="instance",
        label="Contact",
        description=(
            "Where people without an account should ask for one. Rendered as a mailto: "
            "link when it contains an \"@\"."
        ),
        placeholder="coral-admin@example.org",
    ),
    SettingSpec(
        key="instance_notice",
        env_var="INSTANCE_NOTICE",
        group="instance",
        label="Sign-in notice",
        description="Optional one-line notice under the sign-in form, e.g. usage terms.",
    ),
    SettingSpec(
        key="allow_registration",
        env_var="INSTANCE_ALLOW_REGISTRATION",
        group="instance",
        label="Allow self-registration",
        description=(
            "Whether visitors may create their own account. With it off, /auth/register is "
            "refused by the API, not merely hidden. The first account is always allowed, so "
            "a fresh install is never locked out."
        ),
        kind=KIND_BOOL,
        default="false",
    ),
)

BY_KEY: dict[str, SettingSpec] = {spec.key: spec for spec in SETTINGS}

#: Human-readable group headings, in the order the admin page renders them.
GROUPS: tuple[tuple[str, str, str], ...] = (
    ("ai", "AI and credentials",
     "Tokens the tool uses to reach model weights and language models."),
    ("instance", "This instance",
     "How this deployment describes itself, and whether it hands out accounts."),
)


def _clean(value: str | None) -> str | None:
    """Treat blank and whitespace-only as unset.

    The installer writes every key it knows about and leaves the skipped ones
    empty, so "present but empty" is the normal way to say "not set" rather than
    an anomaly worth distinguishing.
    """
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


#: The deployment's own value for each environment variable, as it stood before
#: this module wrote anything into the process environment.
_ENV_SNAPSHOT: dict[str, str | None] = {}


def _env_default(spec: SettingSpec) -> str | None:
    """The deployment's configured value for this setting, ignoring overrides.

    Recorded on first use rather than at import: ``load_dotenv()`` runs while the
    app is being built, so an import-time capture could see the process
    environment before ``.env`` had been applied. Once recorded it is never
    re-read, because :func:`apply` writes the *override* into ``os.environ`` for
    the libraries that read it directly -- so after one save the live environment
    is no longer the deployment's answer.
    """
    if spec.env_var not in _ENV_SNAPSHOT:
        _ENV_SNAPSHOT[spec.env_var] = os.getenv(spec.env_var)
    return _ENV_SNAPSHOT[spec.env_var]


def _overrides(db: Session) -> dict[str, str | None]:
    """Every stored override, as ``{key: value}``."""
    return {row.key: row.value for row in db.query(InstanceSettings).all()}


def _resolve(spec: SettingSpec, overrides: dict[str, str | None]) -> str | None:
    """Resolve one setting: stored row, then environment, then default."""
    if spec.key in overrides:
        # A stored row wins even when it is empty: clearing a value from the
        # admin page has to be able to shadow a non-empty environment default,
        # otherwise "delete this key" would silently do nothing.
        return _clean(overrides[spec.key])
    return _clean(_env_default(spec)) or _clean(spec.default)


def get(key: str, db: Session | None = None) -> str | None:
    """The effective value of one setting, or ``None`` when unset.

    ``db`` lets a request reuse its own session; callers outside a request (and
    the sign-in page, which has none) leave it out and get a short-lived one.
    """
    return get_many(key, db=db)[key]


def get_bool(key: str, db: Session | None = None) -> bool:
    """The effective value of a boolean setting.

    Opt-in rather than opt-out: anything other than an explicit "true" reads as
    false, so an instance holding real research data does not change behaviour
    because a value was misspelled.
    """
    return (get(key, db) or "").strip().lower() == "true"


def get_many(*keys: str, db: Session | None = None) -> dict[str, str | None]:
    """Resolve several settings in one query."""
    if db is not None:
        overrides = _overrides(db)
        return {key: _resolve(BY_KEY[key], overrides) for key in keys}
    with get_context_session() as own:
        overrides = _overrides(own)
        return {key: _resolve(BY_KEY[key], overrides) for key in keys}


def _mask(value: str | None) -> str | None:
    """A hint at a secret: enough to tell two keys apart, not enough to use one."""
    if not value:
        return None
    return f"…{value[-4:]}" if len(value) > 4 else "…"


def describe(db: Session) -> list[dict]:
    """Describe every setting for the admin page.

    Secrets report only whether they are set and their last four characters --
    the actual value never leaves the server once written.
    """
    overrides = _overrides(db)
    rows = {row.key: row for row in db.query(InstanceSettings).all()}
    described = []
    for spec in SETTINGS:
        value = _resolve(spec, overrides)
        row = rows.get(spec.key)
        described.append({
            "key": spec.key,
            "label": spec.label,
            "description": spec.description,
            "group": spec.group,
            "kind": spec.kind,
            "scope": spec.scope,
            "env_var": spec.env_var,
            "placeholder": spec.placeholder,
            # Secrets are write-only; everything else round-trips so the field can
            # be edited rather than retyped.
            "value": None if spec.kind == KIND_SECRET else (value or ""),
            "is_set": bool(value),
            "hint": _mask(value) if spec.kind == KIND_SECRET else None,
            # Lets the page say "overriding the environment" rather than leaving
            # an operator to wonder why .env no longer matches what they see.
            "overridden": spec.key in overrides,
            "updated_at": row.updated_at.isoformat() if row and isinstance(row.updated_at, datetime) else None,
            "updated_by": row.updated_by if row else None,
        })
    return described


def apply(db: Session, updates: dict[str, str | None], username: str) -> list[SettingSpec]:
    """Store overrides for the given settings and return the specs that changed.

    Mirrored into ``os.environ`` as well as the database, because the libraries
    these values feed (LiteLLM, huggingface_hub) also read the environment
    directly for cases this code does not route by hand.

    Does not commit -- the caller owns the transaction, so a failed push to the
    AI service can roll the whole change back rather than leaving the database
    claiming something the service never received.
    """
    changed: list[SettingSpec] = []
    for key, raw in updates.items():
        spec = BY_KEY.get(key)
        if spec is None:
            # Refused rather than ignored: silently dropping an unknown key would
            # let a typo look like a saved setting.
            raise KeyError(key)

        value = raw.strip() if isinstance(raw, str) else raw
        if spec.kind == KIND_SECRET and value == "":
            # An empty secret field means "leave it alone", not "clear it" --
            # the field renders blank because the value is never sent back.
            # Clearing is a separate, explicit action.
            continue
        if spec.kind == KIND_BOOL:
            value = "true" if str(value).strip().lower() in ("true", "1", "yes", "on") else "false"

        row = db.get(InstanceSettings, spec.key)
        if row is None:
            row = InstanceSettings(key=spec.key)
            db.add(row)
        row.value = value
        row.updated_by = username
        row.updated_at = datetime.now()

        _env_default(spec)  # record the deployment's own value before shadowing it
        os.environ[spec.env_var] = value or ""
        changed.append(spec)

    if changed:
        logger.info("Settings changed by %r: %s",
                    username, ", ".join(spec.key for spec in changed))
    return changed


def clear(db: Session, key: str, username: str) -> SettingSpec:
    """Drop the override for one setting, falling back to the environment again.

    The row is deleted rather than blanked, so the setting reads as "not
    overridden" and the admin page stops claiming an override that no longer
    differs from the deployment's own configuration.
    """
    spec = BY_KEY[key]
    row = db.get(InstanceSettings, spec.key)
    if row is not None:
        db.delete(row)
    # Put the deployment's own value back: the process environment still holds
    # whatever an earlier save wrote there, which would otherwise keep shadowing
    # the .env the operator is falling back to.
    original = _env_default(spec)
    if original is None:
        os.environ.pop(spec.env_var, None)
    else:
        os.environ[spec.env_var] = original
    logger.info("Setting %r cleared by %r; falling back to %s.",
                spec.key, username, spec.env_var)
    return spec
