"""Who is running *this* deployment.

IQUANA is self-hosted: every research group runs its own instance against its own
datasets and its own user list. Two surfaces need to know which instance this is
— the sign-in page, which greets a visitor who was handed a bare hostname and
tells them who to ask for an account, and the registration endpoint, which has to
know whether this instance hands out accounts at all.

The values live on the backend rather than in the frontend's build for two
reasons. The first is enforcement: whether self-registration is allowed is a
policy, and a policy the frontend merely *renders* is not enforced — anyone can
POST to ``/auth/register`` regardless of what the sign-in page chose to show. The
second is operational: Vite substitutes ``import.meta.env`` at build time, so
frontend-held branding would be baked into the bundle and could not be corrected
without a rebuild.

Every value is optional. With nothing configured the instance simply describes
itself as IQUANA, which is what a single-group local install should say.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _read(key: str) -> str | None:
    """An environment value, treating blank and whitespace-only as unset.

    The installer writes every key it knows about, leaving the ones the operator
    skipped empty, so "present but empty" is the normal way to say "not set" here
    rather than an anomaly worth distinguishing.
    """
    value = os.getenv(key)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


@dataclass(frozen=True)
class InstanceConfig:
    """Branding and account policy for this deployment."""

    #: Display name, e.g. "HIFMB Reef Lab". ``None`` when unbranded.
    name: str | None
    #: Who hosts it, e.g. "the Helmholtz Institute for Functional Marine Biodiversity".
    organisation: str | None
    #: Where to ask for an account — usually an email address.
    contact: str | None
    #: Optional free-text notice shown under the sign-in form.
    notice: str | None
    #: Whether visitors may create their own account.
    allow_registration: bool


def get_instance_config() -> InstanceConfig:
    """Read the instance configuration from the environment.

    Read per call rather than cached at import: ``load_dotenv()`` runs inside
    ``create_app``, so import-time capture would see the process environment
    before the ``.env`` file had been applied.
    """
    return InstanceConfig(
        name=_read("INSTANCE_NAME"),
        organisation=_read("INSTANCE_ORG"),
        contact=_read("INSTANCE_CONTACT"),
        notice=_read("INSTANCE_NOTICE"),
        # Opt-in rather than opt-out: an instance holding real research data
        # should not start accepting strangers because a variable was misspelled.
        allow_registration=(os.getenv("INSTANCE_ALLOW_REGISTRATION", "").strip().lower() == "true"),
    )
