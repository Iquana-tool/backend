"""API keys held by one account or one organisation.

The instance's own keys stay in ``instance_settings`` (the admin page). A row here
is a key of one *kind* -- so far only ``llm`` -- that belongs either to an account
(a personal key) or to an organisation, never both; each owner holds at most one
key per kind. Which key a request uses is decided in ``app.services.credentials``.

An LLM key is only meaningful together with the model id whose provider issued it
(and, for self-hosted endpoints, the base URL), so those are stored on the same row.

The key itself is encrypted (see ``app.services.secrets``); ``hint`` keeps the last
characters in the clear, so a listing can tell keys apart without decrypting any.
"""
from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint

from app.database import database
from app.database.users import utc_now


class Credentials(database):
    __tablename__ = "credentials"
    __table_args__ = (
        CheckConstraint("(username IS NULL) <> (organization_id IS NULL)",
                        name="ck_credentials_one_owner"),
        # NULLs never collide in a UNIQUE, so each constraint only binds the rows
        # of its own kind of owner.
        UniqueConstraint("username", "kind", name="uq_credentials_username_kind"),
        UniqueConstraint("organization_id", "kind", name="uq_credentials_organization_kind"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    #: What the key is for; one of ``app.schemas.credentials.CredentialKind``.
    kind = Column(String(16), nullable=False)
    username = Column(String, ForeignKey("users.username", ondelete="CASCADE", onupdate="CASCADE"),
                      nullable=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    #: LiteLLM model id, ``<provider>/<model>``.
    model = Column(String(200), nullable=True)
    api_base = Column(String(255), nullable=True)
    #: The key, encrypted.
    secret = Column(Text, nullable=False)
    hint = Column(String(16), nullable=False)
    created_by = Column(String, ForeignKey("users.username", ondelete="SET NULL", onupdate="CASCADE"),
                        nullable=True)
    created_at = Column(DateTime, nullable=False, default=utc_now)
    updated_at = Column(DateTime, nullable=False, default=utc_now, onupdate=utc_now)
    last_used_at = Column(DateTime, nullable=True)
