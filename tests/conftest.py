"""Setup shared by every test."""
import pytest
from cryptography.fernet import Fernet

from app.services import secrets


@pytest.fixture(autouse=True)
def _throwaway_secrets_key(monkeypatch):
    """Encrypt with a key of the test's own.

    Without it, the first test that stores a secret would read -- or create -- the
    deployment's key file in ``data/``.
    """
    monkeypatch.setenv("IQUANA_SECRETS_KEY", Fernet.generate_key().decode())
    secrets.reset_key_cache()
    yield
    secrets.reset_key_cache()
