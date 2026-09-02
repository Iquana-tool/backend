from sqlalchemy import Column, DateTime, String, Text, func

from app.database import database


class InstanceSettings(database):
    """Runtime overrides for the deployment's own configuration.

    One row per setting, keyed by the identifier in
    ``app.services.settings.SETTINGS``. The environment stays the *default* and
    the row is the *override*, so a deployment that sets nothing here behaves
    exactly as it did before this table existed, and an operator can correct a
    value from the admin page instead of editing ``.env`` and restarting.

    Only settings the admin surface is allowed to edit are ever written here;
    an unknown key is refused rather than stored, so this cannot become a
    parallel, unreviewed configuration channel.
    """

    __tablename__ = "instance_settings"

    #: Identifier from the settings registry, e.g. ``llm_api_key``. Deliberately
    #: not the environment variable name: the registry may re-point a setting at
    #: a different variable without orphaning the stored value.
    key = Column(String(64), primary_key=True)
    #: Stored as text and parsed per the setting's declared kind. ``None`` means
    #: "explicitly cleared", which still shadows the environment default.
    value = Column(Text, nullable=True)
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    #: Who last changed it. Not a foreign key -- the value should survive the
    #: account that set it being removed.
    updated_by = Column(String, nullable=True)
