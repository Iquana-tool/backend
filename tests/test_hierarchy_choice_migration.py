"""Exercise the additive policy migration without requiring a PostgreSQL server."""
import importlib.util
from pathlib import Path
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text, inspect


def test_hierarchy_policy_migration_preserves_existing_rows(tmp_path):
    path = Path(__file__).parents[1] / 'migrations/versions/2026_11_07_0017_contour_hierarchy_choice.py'
    spec = importlib.util.spec_from_file_location('choice_migration', path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    assert migration.revision == '0017'
    assert migration.down_revision == '0011'
    engine = create_engine(f'sqlite:///{tmp_path / "migration.db"}')
    with engine.begin() as connection:
        connection.execute(text('CREATE TABLE contours (id INTEGER PRIMARY KEY, parent_id INTEGER)'))
        connection.execute(text('INSERT INTO contours (id, parent_id) VALUES (1, NULL), (2, 1)'))
        migration.op = Operations(MigrationContext.configure(connection))
        migration.upgrade()
        assert connection.execute(text('SELECT id, parent_id, hierarchy_enabled FROM contours ORDER BY id')).all() == [(1, None, 1), (2, 1, 1)]
        connection.execute(text('INSERT INTO contours (id, hierarchy_enabled) VALUES (3, 0)'))
        assert connection.execute(text('SELECT hierarchy_enabled FROM contours WHERE id=3')).scalar() == 0
        migration.downgrade()
        assert 'hierarchy_enabled' not in {column['name'] for column in inspect(connection).get_columns('contours')}
        assert connection.execute(text('SELECT COUNT(*) FROM contours')).scalar() == 3
    engine.dispose()


def test_sqlite_startup_patch_is_idempotent(tmp_path):
    from app.database import _ensure_columns
    engine = create_engine(f'sqlite:///{tmp_path / "legacy.db"}')
    with engine.begin() as connection:
        connection.execute(text('CREATE TABLE contours (id INTEGER PRIMARY KEY)'))
        connection.execute(text('INSERT INTO contours (id) VALUES (1)'))
    _ensure_columns(engine)
    with engine.begin() as connection:
        assert connection.execute(text('SELECT hierarchy_enabled FROM contours WHERE id=1')).scalar() == 1
        connection.execute(text('UPDATE contours SET hierarchy_enabled=0 WHERE id=1'))
    _ensure_columns(engine)
    with engine.connect() as connection:
        assert connection.execute(text('SELECT hierarchy_enabled FROM contours WHERE id=1')).scalar() == 0
    engine.dispose()
