import pytest
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.exc import IntegrityError

from app.database import _ensure_columns, build_schema_from_models, database, import_models


def _legacy_database(path):
    """Create the pre-profile/organization/stack SQLite shape with retained rows."""
    engine = create_engine(f"sqlite:///{path}")
    import_models()
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE users (username VARCHAR PRIMARY KEY NOT NULL, "
            "hashed_password VARCHAR NOT NULL, global_role VARCHAR(20) NOT NULL, "
            "is_active BOOLEAN NOT NULL DEFAULT 1)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE datasets (id INTEGER PRIMARY KEY, name VARCHAR(50) NOT NULL, "
            "description VARCHAR(255), dataset_type VARCHAR(20) NOT NULL, "
            "folder_path VARCHAR(255) NOT NULL, created_by VARCHAR NOT NULL, "
            "require_independent_review BOOLEAN NOT NULL DEFAULT 0, "
            "disabled_ai_tools VARCHAR(255))"
        )
        connection.exec_driver_sql(
            "CREATE TABLE images (id INTEGER PRIMARY KEY, dataset_id INTEGER NOT NULL, "
            "file_name VARCHAR NOT NULL, file_path VARCHAR NOT NULL, "
            "thumbnail_file_path VARCHAR NOT NULL, description VARCHAR, "
            "width INTEGER NOT NULL, height INTEGER NOT NULL, color_mode VARCHAR NOT NULL, "
            "scale_x FLOAT NOT NULL, scale_y FLOAT NOT NULL, unit VARCHAR(10))"
        )
        connection.exec_driver_sql(
            "CREATE TABLE masks (id INTEGER PRIMARY KEY, image_id INTEGER NOT NULL, "
            "fully_annotated BOOLEAN NOT NULL DEFAULT 0, file_path VARCHAR NOT NULL)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE contours (id INTEGER PRIMARY KEY, mask_id INTEGER NOT NULL, "
            "parent_id INTEGER, temporary BOOLEAN NOT NULL DEFAULT 0, "
            "added_by VARCHAR(255) NOT NULL, origin VARCHAR(32), suggestion_id INTEGER, "
            "geometry_edited_at DATETIME, author_username VARCHAR, created_at DATETIME NOT NULL, "
            "confidence_score FLOAT NOT NULL, label_id INTEGER, area FLOAT NOT NULL, "
            "perimeter FLOAT NOT NULL, circularity FLOAT NOT NULL, diameter FLOAT NOT NULL, "
            "x JSON NOT NULL, y JSON NOT NULL)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE image_metadata (id INTEGER PRIMARY KEY, image_id INTEGER NOT NULL, "
            "key VARCHAR(100) NOT NULL, value TEXT NOT NULL, created_by VARCHAR, "
            "created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE ai_suggestions (id INTEGER PRIMARY KEY, run_id VARCHAR(64) NOT NULL, "
            "source VARCHAR(32) NOT NULL, model_key VARCHAR(255), username VARCHAR, "
            "dataset_id INTEGER, image_id INTEGER, mask_id INTEGER, contour_id INTEGER, "
            "label_id INTEGER, confidence FLOAT, x JSON NOT NULL, y JSON NOT NULL, "
            "created_at DATETIME NOT NULL, edited_at DATETIME, deleted_at DATETIME, "
            "reviewed_at DATETIME, superseded_at DATETIME)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE reviewer_contour_association (reviewer_id VARCHAR NOT NULL, "
            "contour_id INTEGER NOT NULL, PRIMARY KEY (reviewer_id, contour_id))"
        )
        connection.exec_driver_sql(
            "INSERT INTO users (username, hashed_password, global_role, is_active) "
            "VALUES ('legacy', 'opaque-hash', 'admin', 1)"
        )
        connection.exec_driver_sql(
            "INSERT INTO datasets (id, name, dataset_type, folder_path, created_by) "
            "VALUES (41, 'Existing dataset', 'image', '/existing', 'legacy')"
        )
        connection.exec_driver_sql(
            "INSERT INTO images (id, dataset_id, file_name, file_path, thumbnail_file_path, "
            "width, height, color_mode, scale_x, scale_y) "
            "VALUES (7, 41, 'image.png', '/existing/image.png', '/existing/thumb.png', "
            "20, 10, 'RGB', 1, 1)"
        )
        connection.exec_driver_sql(
            "INSERT INTO masks (id, image_id, file_path) VALUES (9, 7, '/existing/mask.png')"
        )
        connection.exec_driver_sql(
            "INSERT INTO contours (id, mask_id, added_by, created_at, confidence_score, "
            "area, perimeter, circularity, diameter, x, y) VALUES "
            "(12, 9, 'human', '2026-10-01 00:00:00', 1, 2, 3, 0.5, 2, '[0.1,0.2]', '[0.3,0.4]')"
        )
        connection.exec_driver_sql(
            "INSERT INTO image_metadata (id, image_id, key, value, created_at, updated_at) "
            "VALUES (15, 7, 'kind', 'specimen', '2026-10-01', '2026-10-01')"
        )
        connection.exec_driver_sql(
            "INSERT INTO ai_suggestions (id, run_id, source, x, y, created_at) "
            "VALUES (17, 'run', 'model', '[]', '[]', '2026-10-01')"
        )
    return engine


def test_legacy_sqlite_schema_upgrade_preserves_rows_and_is_idempotent(tmp_path):
    engine = _legacy_database(tmp_path / "legacy.db")

    build_schema_from_models(engine)
    build_schema_from_models(engine)

    with engine.connect() as connection:
        user = connection.execute(text(
            "SELECT hashed_password, preferences, must_change_password, created_at "
            "FROM users WHERE username = 'legacy'"
        )).one()
        assert user == ("opaque-hash", "{}", 0, None)

        organization_id = connection.execute(text(
            "SELECT id FROM organizations WHERE is_default = 1"
        )).scalar_one()
        assert connection.execute(text(
            "SELECT role FROM organization_members WHERE organization_id = :id "
            "AND username = 'legacy'"
        ), {"id": organization_id}).scalar_one() == "admin"
        assert connection.execute(text(
            "SELECT organization_id FROM datasets WHERE id = 41"
        )).scalar_one() == organization_id
        assert connection.execute(text("SELECT count(*) FROM organization_members")).scalar_one() == 1
        assert connection.execute(text("SELECT count(*) FROM datasets")).scalar_one() == 1
        assert connection.execute(text("SELECT count(*) FROM contours")).scalar_one() == 1
        assert connection.execute(text("SELECT count(*) FROM images")).scalar_one() == 1
        assert connection.execute(text("SELECT kind FROM images WHERE id = 7")).scalar_one() == "image"
        assert connection.execute(text(
            "SELECT x, y FROM contours WHERE id = 12"
        )).one() == ("[0.1,0.2]", "[0.3,0.4]")
        assert connection.execute(text(
            "SELECT stack_id FROM image_metadata WHERE id = 15"
        )).scalar_one() is None
        assert connection.execute(text(
            "SELECT final_iou FROM ai_suggestions WHERE id = 17"
        )).scalar_one() is None

        # SQLite UNIQUE indexes permit any number of NULL emails, as intended.
        connection.execute(text(
            "INSERT INTO users (username, hashed_password, global_role) "
            "VALUES ('second', 'opaque-hash', 'member')"
        ))
        connection.execute(text(
            "INSERT INTO users (username, hashed_password, global_role) "
            "VALUES ('third', 'opaque-hash', 'member')"
        ))
        connection.commit()

    indexes = {
        table: {index["name"] for index in inspect(engine).get_indexes(table)}
        for table in (
            "reviewer_contour_association", "contours", "users", "datasets",
            "images", "masks", "image_metadata",
        )
    }
    assert "ix_reviewer_contour_association_contour_id" in indexes["reviewer_contour_association"]
    assert {"ix_contours_author_username", "ix_contours_label_id", "ix_contours_mask_id",
            "ix_contours_parent_id", "ix_contours_suggestion_id"} <= indexes["contours"]
    assert "users_email_key" in indexes["users"]
    assert "ix_datasets_organization_id" in indexes["datasets"]
    assert {"ix_images_dataset_id", "ix_images_kind", "ix_images_stack_id",
            "uq_images_stack_frame_index"} <= indexes["images"]
    assert "ix_masks_image_id" in indexes["masks"]
    assert {"ix_image_metadata_stack_id", "uq_image_metadata_stack_key"} <= indexes["image_metadata"]

    with engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO users (username, hashed_password, global_role, email) "
            "VALUES ('duplicate-email', 'opaque-hash', 'member', 'same@example.test')"
        ))
    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(text(
                "INSERT INTO users (username, hashed_password, global_role, email) "
                "VALUES ('duplicate-email-2', 'opaque-hash', 'member', 'same@example.test')"
            ))

    engine.dispose()


def test_sqlite_schema_upgrade_rolls_back_columns_indexes_and_backfill_on_error(tmp_path):
    engine = _legacy_database(tmp_path / "legacy-rollback.db")
    database.metadata.create_all(engine)

    def fail_before_email_index(_conn, _cursor, statement, _parameters, _context, _many):
        if "CREATE UNIQUE INDEX IF NOT EXISTS users_email_key" in statement:
            raise RuntimeError("injected schema upgrade failure")

    event.listen(engine, "before_cursor_execute", fail_before_email_index)
    with pytest.raises(RuntimeError, match="injected schema upgrade failure"):
        _ensure_columns(engine)
    event.remove(engine, "before_cursor_execute", fail_before_email_index)

    inspector = inspect(engine)
    assert "email" not in {column["name"] for column in inspector.get_columns("users")}
    assert "organization_id" not in {
        column["name"] for column in inspector.get_columns("datasets")
    }
    with engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM organizations")).scalar_one() == 0
        assert connection.execute(text("SELECT count(*) FROM organization_members")).scalar_one() == 0

    _ensure_columns(engine)
    assert "organization_id" in {
        column["name"] for column in inspect(engine).get_columns("datasets")
    }
    with engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM organizations")).scalar_one() == 1

    engine.dispose()
