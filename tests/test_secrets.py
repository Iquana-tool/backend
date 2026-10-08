"""Checks for the encryption of stored API keys, and for the instance's secrets using it."""
from contextlib import contextmanager

from cryptography.fernet import Fernet
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database, import_models
from app.database.instance_settings import InstanceSettings
from app.services import secrets
from app.services import settings as settings_service

import_models()


def _use_keys(monkeypatch, *keys: bytes) -> None:
    monkeypatch.setenv("IQUANA_SECRETS_KEY", ",".join(key.decode() for key in keys))
    secrets.reset_key_cache()


def test_a_secret_round_trips_and_is_not_stored_readable():
    stored = secrets.encrypt("sk-live-1234")

    assert stored.startswith(secrets.PREFIX)
    assert "sk-live-1234" not in stored
    assert secrets.decrypt(stored) == "sk-live-1234"


def test_plaintext_from_before_encryption_is_still_read():
    assert secrets.decrypt("sk-legacy") == "sk-legacy"
    assert not secrets.is_encrypted("sk-legacy")


def test_keys_rotate_the_first_encrypts_and_all_decrypt(monkeypatch):
    old, new = Fernet.generate_key(), Fernet.generate_key()
    _use_keys(monkeypatch, old)
    written_before = secrets.encrypt("sk-before")

    _use_keys(monkeypatch, new, old)
    written_after = secrets.encrypt("sk-after")
    assert secrets.decrypt(written_before) == "sk-before"

    _use_keys(monkeypatch, new)
    assert secrets.decrypt(written_after) == "sk-after"


def test_a_secret_under_a_lost_key_reads_as_unset(monkeypatch):
    stored = secrets.encrypt("sk-orphaned")
    _use_keys(monkeypatch, Fernet.generate_key())

    assert secrets.decrypt(stored) is None


def test_without_a_configured_key_one_is_generated_once_and_kept(monkeypatch, tmp_path):
    key_file = tmp_path / "nested" / "secrets.key"
    monkeypatch.delenv("IQUANA_SECRETS_KEY")
    monkeypatch.setenv("IQUANA_SECRETS_KEY_FILE", str(key_file))
    secrets.reset_key_cache()

    stored = secrets.encrypt("sk-filed")
    assert key_file.exists()
    first_key = key_file.read_bytes()

    # A restart reads the same file instead of generating another key.
    secrets.reset_key_cache()
    assert secrets.decrypt(stored) == "sk-filed"
    assert key_file.read_bytes() == first_key


def test_hints_show_only_the_end():
    assert secrets.hint("sk-abcdefgh1234") == "…1234"
    assert secrets.hint("abc") == "…"
    assert secrets.hint(None) is None


# -- The instance's secrets ----------------------------------------------------

def _settings_db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'settings.db'}")
    database.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def test_an_instance_secret_is_stored_encrypted_and_read_back_plain(tmp_path, monkeypatch):
    Session = _settings_db(tmp_path)
    monkeypatch.setattr(settings_service, "_ENV_SNAPSHOT", {})
    monkeypatch.setenv("LABEL_SPACE_LLM_API_KEY", "")  # apply() writes it; restore afterwards
    db = Session()

    settings_service.apply(db, {"llm_api_key": "sk-instance-9876"}, "root")
    db.commit()

    assert secrets.is_encrypted(db.get(InstanceSettings, "llm_api_key").value)
    assert settings_service.get("llm_api_key", db) == "sk-instance-9876"
    db.close()


def test_plaintext_instance_secrets_are_encrypted_at_startup(tmp_path, monkeypatch):
    Session = _settings_db(tmp_path)
    db = Session()
    db.add_all([
        InstanceSettings(key="llm_api_key", value="sk-plain"),
        InstanceSettings(key="instance_name", value="Reef Lab"),  # not a secret: left alone
    ])
    db.commit()
    db.close()

    @contextmanager
    def test_session():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr(settings_service, "get_context_session", test_session)

    assert settings_service.encrypt_stored_secrets() == 1
    assert settings_service.encrypt_stored_secrets() == 0

    db = Session()
    stored = db.get(InstanceSettings, "llm_api_key").value
    assert secrets.is_encrypted(stored) and secrets.decrypt(stored) == "sk-plain"
    assert db.get(InstanceSettings, "instance_name").value == "Reef Lab"
    db.close()
