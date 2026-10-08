"""Which API key a piece of work uses, and managing personal and organisation keys.

For an LLM call, the first of these that exists:

1. the caller's **personal** key -- unless the organisation the work belongs to has
   switched personal keys off (``organizations.allow_personal_keys``);
2. that **organisation's** key;
3. the **instance's** key (Admin -> Settings, or the environment).

The organisation the work belongs to is the dataset's, when the work is for a
dataset. Otherwise it is the one a new dataset of the caller's would land in
(`organization_for_new_dataset`): label spaces are mostly drafted while a dataset
is being created.

Team keys are deliberately absent. An account can be in several teams, and a
dataset can be shared with several, so "which team's key pays" has no natural
answer -- where an organisation's key has exactly one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.database.credentials import Credentials
from app.database.datasets import Datasets
from app.database.organizations import Organizations
from app.database.users import utc_now
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.credentials import CredentialKind, LlmCredentialSet
from app.services import secrets
from app.services import settings as settings_service
from app.services.database_access.organizations import organization_for_new_dataset

SOURCE_PERSONAL = "personal"
SOURCE_ORGANIZATION = "organization"
SOURCE_INSTANCE = "instance"


@dataclass(frozen=True)
class LlmConfig:
    """Everything one LLM call needs, and where it came from."""

    model: str
    api_key: str = field(repr=False)
    api_base: str | None
    #: "personal", "organization" or "instance".
    source: str
    #: The ``credentials`` row, for personal and organisation keys.
    credential_id: int | None = None


def _llm_from(credential: Credentials | None, source: str) -> LlmConfig | None:
    if credential is None:
        return None
    api_key = secrets.decrypt(credential.secret)
    if not api_key:
        # Unreadable (the encryption key changed): fall through to the next source
        # rather than failing the call. `secrets.decrypt` has logged it.
        return None
    return LlmConfig(model=credential.model or "", api_key=api_key, api_base=credential.api_base,
                     source=source, credential_id=credential.id)


def work_organization(user: AuthenticatedUser, dataset_id: int | None, db: Session) -> int | None:
    """The organisation a piece of work belongs to, for choosing its key."""
    if dataset_id is not None:
        return db.query(Datasets.organization_id).filter_by(id=dataset_id).scalar()
    return organization_for_new_dataset(user, None, db)


def resolve_llm(user: AuthenticatedUser, db: Session, dataset_id: int | None = None) -> LlmConfig | None:
    """The LLM key and model to use for this caller's work, or None if there is none."""
    organization_id = work_organization(user, dataset_id, db)
    organization = db.get(Organizations, organization_id) if organization_id is not None else None

    if organization is None or organization.allow_personal_keys:
        personal = _llm_from(
            db.query(Credentials).filter_by(username=user.username, kind=CredentialKind.LLM.value).first(),
            SOURCE_PERSONAL)
        if personal is not None:
            return personal
    if organization is not None:
        shared = _llm_from(
            db.query(Credentials).filter_by(organization_id=organization.id, kind=CredentialKind.LLM.value).first(),
            SOURCE_ORGANIZATION)
        if shared is not None:
            return shared

    instance = settings_service.get_many("llm_model", "llm_api_key", "llm_api_base", db=db)
    if instance["llm_api_key"]:
        return LlmConfig(model=instance["llm_model"] or "", api_key=instance["llm_api_key"],
                         api_base=instance["llm_api_base"], source=SOURCE_INSTANCE)
    return None


def mark_used(config: LlmConfig, db: Session) -> None:
    """Record that a personal or organisation key was just used."""
    if config.credential_id is None:
        return
    db.query(Credentials).filter_by(id=config.credential_id).update({Credentials.last_used_at: utc_now()})
    db.commit()


# -- Managing keys -----------------------------------------------------------------

def _iso(value) -> str | None:
    return value.isoformat() if value is not None else None


def describe(credential: Credentials) -> dict:
    """A key as the API shows it: everything but the key itself."""
    return {
        "kind": credential.kind,
        "model": credential.model,
        "api_base": credential.api_base,
        "hint": credential.hint,
        "created_by": credential.created_by,
        "created_at": _iso(credential.created_at),
        "updated_at": _iso(credential.updated_at),
        "last_used_at": _iso(credential.last_used_at),
    }


def _owned(db: Session, username: str | None, organization_id: int | None):
    return db.query(Credentials).filter_by(username=username, organization_id=organization_id)


def list_credentials(db: Session, *, username: str | None = None,
                     organization_id: int | None = None) -> list[dict]:
    return [describe(row) for row in _owned(db, username, organization_id).order_by(Credentials.kind)]


def set_llm_credential(db: Session, body: LlmCredentialSet, actor: str, *,
                       username: str | None = None, organization_id: int | None = None) -> Credentials:
    """Create or change the LLM key of one account or one organisation.

    A new base URL needs the key sent again. The key is write-only, but whoever
    may change the base URL could otherwise point it at a server of their own and
    receive the stored key with the next call.
    """
    row = _owned(db, username, organization_id).filter_by(kind=CredentialKind.LLM.value).first()
    if row is None:
        if not body.api_key:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail="An API key is needed to set up a new key.")
        row = Credentials(kind=CredentialKind.LLM.value, username=username,
                          organization_id=organization_id, created_by=actor)
        db.add(row)
    elif body.api_base != row.api_base and not body.api_key:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Changing the base URL needs the API key to be entered again.")

    row.model = body.model
    row.api_base = body.api_base
    if body.api_key:
        row.secret = secrets.encrypt(body.api_key)
        row.hint = secrets.hint(body.api_key)
    db.commit()
    return row


def delete_credential(db: Session, kind: CredentialKind, *, username: str | None = None,
                      organization_id: int | None = None) -> bool:
    row = _owned(db, username, organization_id).filter_by(kind=kind.value).first()
    if row is None:
        return False
    db.delete(row)
    db.commit()
    return True
