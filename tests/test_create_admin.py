"""The installer's first-admin bootstrap (scripts/create_admin.py --bootstrap)."""
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database, import_models
from app.database.users import Users
from app.schemas.permissions import GlobalRole
from app.services.auth import verify_password

import_models()

_SPEC = importlib.util.spec_from_file_location(
    "create_admin", Path(__file__).resolve().parents[1] / "scripts" / "create_admin.py")
create_admin = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(create_admin)


@pytest.fixture
def factory(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'admin.db'}")
    database.metadata.create_all(engine)
    make = sessionmaker(bind=engine)

    @contextmanager
    def session():
        db = make()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr(create_admin, "get_context_session", session)
    monkeypatch.setattr(create_admin, "init_db", lambda: None)
    monkeypatch.setattr(create_admin, "join_default_organization", lambda username, db: None)
    return make


def _run(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["create_admin.py", *argv])
    code = create_admin.main()
    return code, capsys.readouterr()


def test_bootstrap_creates_one_admin_with_a_generated_password(factory, monkeypatch, capsys):
    code, out = _run(monkeypatch, capsys, "--bootstrap", "admin")
    assert code == 0
    password = out.out.strip()
    assert len(password) >= 12 and "\n" not in password
    db = factory()
    admin = db.query(Users).filter_by(username="admin").one()
    assert admin.global_role == GlobalRole.ADMIN.value
    assert admin.is_active and admin.must_change_password
    assert verify_password(password, admin.hashed_password)


def test_bootstrap_does_nothing_once_an_admin_exists(factory, monkeypatch, capsys):
    _run(monkeypatch, capsys, "--bootstrap", "admin")
    before = factory().query(Users).filter_by(username="admin").one().hashed_password
    code, out = _run(monkeypatch, capsys, "--bootstrap", "someone-else")
    assert code == 0 and out.out == ""
    db = factory()
    assert db.query(Users).filter_by(username="admin").one().hashed_password == before
    assert db.query(Users).filter_by(username="someone-else").first() is None


def test_bootstrap_refuses_a_password_on_the_command_line(factory, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _run(monkeypatch, capsys, "--bootstrap", "admin", "hunter2")
