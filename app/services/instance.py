"""Who is running *this* deployment.

IQUANA is self-hosted: every research group runs its own instance against its own
datasets and its own user list. Two surfaces need to know which instance this is
— the sign-in page, which greets a visitor who was handed a bare hostname and
tells them who to ask for an account. There is no self-registration: the installer
creates the first admin (``scripts/create_admin.py --bootstrap``), and the admin
creates every other account from the admin page.

The values live on the backend rather than in the frontend's build because Vite
substitutes ``import.meta.env`` at build time, so frontend-held branding would be
baked into the bundle and could not be corrected without a rebuild.

Every value is optional. With nothing configured the instance simply describes
itself as IQUANA, which is what a single-group local install should say.

Since the admin page landed these are read through :mod:`app.services.settings`,
which resolves a stored override before the environment. The environment
variables documented in ``env.example`` still supply the defaults, so a
deployment that never opens the admin page is unaffected — but fixing a typo in
the contact address no longer means editing ``.env`` and restarting.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.services import settings as settings_service


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


def get_instance_config() -> InstanceConfig:
    """Read the instance configuration: stored overrides first, environment second.

    Read per call rather than cached at import, for two reasons that both still
    hold: ``load_dotenv()`` runs inside ``create_app``, so an import-time capture
    would see the process environment before the ``.env`` file had been applied —
    and an admin editing these values expects the sign-in page to change without
    a restart.
    """
    values = settings_service.get_many(
        "instance_name", "instance_org", "instance_contact", "instance_notice",
    )
    return InstanceConfig(
        name=values["instance_name"],
        organisation=values["instance_org"],
        contact=values["instance_contact"],
        notice=values["instance_notice"],
    )
