"""Tests for the role/permission model.

Covers, against a temp-file SQLite database (same pattern as the other tests):
  * the role -> permission matrix and its monotonic escalation,
  * `AuthenticatedUser` resolution from membership rows, including per-member
    `extra_permissions` / `denied_permissions` overrides,
  * the admin bypass and the guest/member split on `dataset.create`,
  * dataset-id resolution from contour / mask / image / label ids,
  * membership grant / revoke / ownership transfer rules,
  * invite links: redemption, expiry, use limits, revocation and no-downgrade,
  * rejections and the reset of the annotate/review phases they produce,
  * separation of duties when a dataset opts into independent review.
"""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.database import database
import app.database.datasets  # noqa: F401
import app.database.images  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.users  # noqa: F401
import app.database.contours  # noqa: F401
import app.database.dataset_members  # noqa: F401
import app.database.rejections  # noqa: F401
from app.database.contours import Contours
from app.database.dataset_members import DatasetInvites, DatasetMembers
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import (
    DATASET_ROLE_PERMISSIONS,
    DatasetRole,
    GLOBAL_PERMISSIONS,
    GlobalRole,
    Permission,
    is_at_least,
)
from app.schemas.review import InviteCreate, RejectionCreate, RejectionReason
from app.services.database_access import contours as contours_db
from app.services.database_access import members as members_db
from app.services.database_access import rejections as rejections_db
from app.services.permissions import (
    dataset_id_for_contour,
    dataset_id_for_image,
    dataset_id_for_label,
    dataset_id_for_mask,
    resolve_dataset_id,
)

from fastapi import HTTPException

WIDTH, HEIGHT = 100, 100


@event.listens_for(Engine, "connect")
def _fk_pragma(dbapi_connection, connection_record):
    import sqlite3
    if isinstance(dbapi_connection, sqlite3.Connection):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    database.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    try:
        yield s
    finally:
        s.close()
        engine.dispose()


def _user(session, username, global_role=GlobalRole.MEMBER):
    user = Users(username=username, hashed_password="x", global_role=global_role.value)
    session.add(user)
    session.flush()
    return user


def _dataset(session, owner="owner", name="ds"):
    ds = Datasets(name=name, description="", dataset_type="image",
                  folder_path=f"/tmp/{name}", created_by=owner)
    session.add(ds)
    session.flush()
    members_db.ensure_owner_membership(ds.id, owner, session)
    return ds


def _grant(session, dataset_id, username, role, extra=None, denied=None):
    session.add(DatasetMembers(
        dataset_id=dataset_id,
        username=username,
        role=role.value,
        extra_permissions=[p.value for p in (extra or [])],
        denied_permissions=[p.value for p in (denied or [])],
    ))
    session.commit()


def _seed(session):
    """owner + one dataset + image -> mask -> contour, plus a label."""
    _user(session, "owner")
    ds = _dataset(session, owner="owner")

    img = Images(dataset_id=ds.id, file_name="a.png", file_path="/tmp/a.png",
                 thumbnail_file_path="/tmp/t.png", width=WIDTH, height=HEIGHT,
                 color_mode="RGB")
    session.add(img)
    session.flush()

    mask = Masks(image_id=img.id, fully_annotated=False, file_path="/tmp/m.png")
    session.add(mask)
    session.flush()

    label = Labels(dataset_id=ds.id, parent_id=None, name="cell", value=1)
    session.add(label)
    session.flush()

    contour = Contours(mask_id=mask.id, added_by="manual", author_username="owner",
                       confidence_score=1.0, label_id=label.id, area=1.0, perimeter=1.0,
                       circularity=1.0, diameter=1.0, x=[0.1, 0.2, 0.3], y=[0.1, 0.2, 0.3])
    session.add(contour)
    session.commit()
    return ds, img, mask, label, contour


def _auth(session, username) -> AuthenticatedUser:
    return AuthenticatedUser.from_query(session.query(Users).filter_by(username=username).one())


# -- The matrix ------------------------------------------------------------

def test_roles_escalate_monotonically():
    """Each role is a superset of the one below it, so no role is a sideways move."""
    order = [DatasetRole.VIEWER, DatasetRole.ANNOTATOR, DatasetRole.REVIEWER,
             DatasetRole.CURATOR, DatasetRole.OWNER]
    for lower, higher in zip(order, order[1:]):
        assert DATASET_ROLE_PERMISSIONS[lower] < DATASET_ROLE_PERMISSIONS[higher], (
            f"{higher.value} should strictly contain {lower.value}"
        )


def test_no_dataset_role_grants_a_global_permission():
    for role, permissions in DATASET_ROLE_PERMISSIONS.items():
        assert not (permissions & GLOBAL_PERMISSIONS), f"{role.value} leaks a global permission"


def test_key_permissions_sit_at_the_intended_tier():
    viewer = DATASET_ROLE_PERMISSIONS[DatasetRole.VIEWER]
    annotator = DATASET_ROLE_PERMISSIONS[DatasetRole.ANNOTATOR]
    reviewer = DATASET_ROLE_PERMISSIONS[DatasetRole.REVIEWER]
    curator = DATASET_ROLE_PERMISSIONS[DatasetRole.CURATOR]

    assert Permission.EXPORT_ANNOTATIONS not in viewer
    assert Permission.EXPORT_ANNOTATIONS not in annotator
    # Annotators must not be able to approve their way to "finished".
    assert Permission.REVIEW_APPROVE not in annotator
    assert Permission.REVIEW_APPROVE in reviewer
    # Raw imagery is a separate grant from the annotations.
    assert Permission.EXPORT_IMAGES not in reviewer
    assert Permission.EXPORT_IMAGES in curator
    # Destructive label-space edits and training are curator-and-up.
    assert Permission.LABEL_MANAGE not in reviewer
    assert Permission.AI_TRAIN not in reviewer
    assert Permission.MEMBER_GRANT not in curator


def test_is_at_least():
    assert is_at_least(DatasetRole.CURATOR, DatasetRole.ANNOTATOR)
    assert not is_at_least(DatasetRole.ANNOTATOR, DatasetRole.CURATOR)
    assert not is_at_least("nonsense", DatasetRole.VIEWER)


# -- User resolution -------------------------------------------------------

def test_owner_gets_every_dataset_permission(session):
    ds, *_ = _seed(session)
    owner = _auth(session, "owner")
    assert owner.role_for(ds.id) is DatasetRole.OWNER
    assert owner.has_permission(ds.id, Permission.DATASET_DELETE)
    assert owner.has_permission(ds.id, Permission.EXPORT_IMAGES)


def test_non_member_has_nothing(session):
    ds, *_ = _seed(session)
    _user(session, "stranger")
    session.commit()
    stranger = _auth(session, "stranger")
    assert stranger.role_for(ds.id) is None
    assert not stranger.has_permission(ds.id, Permission.DATASET_READ)
    assert not stranger.has_permission(ds.id, Permission.ANNOTATION_READ)


def test_annotator_can_annotate_but_not_review_or_export(session):
    ds, *_ = _seed(session)
    _user(session, "ann")
    _grant(session, ds.id, "ann", DatasetRole.ANNOTATOR)

    ann = _auth(session, "ann")
    assert ann.has_permission(ds.id, Permission.ANNOTATION_CREATE)
    assert ann.has_permission(ds.id, Permission.MASK_SUBMIT)
    assert not ann.has_permission(ds.id, Permission.REVIEW_APPROVE)
    assert not ann.has_permission(ds.id, Permission.MASK_REOPEN)
    assert not ann.has_permission(ds.id, Permission.EXPORT_ANNOTATIONS)


def test_extra_permission_grants_one_capability_without_a_new_role(session):
    """The escape hatch: an annotator who may also pull the measurements."""
    ds, *_ = _seed(session)
    _user(session, "ann")
    _grant(session, ds.id, "ann", DatasetRole.ANNOTATOR,
           extra=[Permission.EXPORT_QUANTIFICATION])

    ann = _auth(session, "ann")
    assert ann.has_permission(ds.id, Permission.EXPORT_QUANTIFICATION)
    # ...but nothing else came with it.
    assert not ann.has_permission(ds.id, Permission.EXPORT_IMAGES)
    assert not ann.has_permission(ds.id, Permission.REVIEW_APPROVE)


def test_denied_permission_beats_the_role_and_any_extra(session):
    ds, *_ = _seed(session)
    _user(session, "cur")
    _grant(session, ds.id, "cur", DatasetRole.CURATOR,
           extra=[Permission.EXPORT_IMAGES],
           denied=[Permission.EXPORT_IMAGES, Permission.AI_TRAIN])

    cur = _auth(session, "cur")
    assert not cur.has_permission(ds.id, Permission.EXPORT_IMAGES)
    assert not cur.has_permission(ds.id, Permission.AI_TRAIN)
    assert cur.has_permission(ds.id, Permission.IMAGE_UPLOAD)


def test_admin_bypasses_dataset_membership(session):
    ds, *_ = _seed(session)
    _user(session, "root", GlobalRole.ADMIN)
    session.commit()

    root = _auth(session, "root")
    assert root.role_for(ds.id) is None  # not a member...
    assert root.has_permission(ds.id, Permission.DATASET_DELETE)  # ...but may act anyway
    assert root.has_global_permission(Permission.USER_MANAGE)


def test_guest_cannot_create_datasets_but_member_can(session):
    _user(session, "guest", GlobalRole.GUEST)
    _user(session, "member", GlobalRole.MEMBER)
    session.commit()

    assert not _auth(session, "guest").has_global_permission(Permission.DATASET_CREATE)
    assert _auth(session, "member").has_global_permission(Permission.DATASET_CREATE)
    assert not _auth(session, "member").has_global_permission(Permission.USER_MANAGE)


def test_deactivated_account_loses_everything(session):
    ds, *_ = _seed(session)
    session.query(Users).filter_by(username="owner").one().is_active = False
    session.commit()

    owner = _auth(session, "owner")
    assert not owner.has_permission(ds.id, Permission.DATASET_READ)
    assert not owner.has_global_permission(Permission.DATASET_CREATE)


def test_legacy_creator_without_membership_row_is_still_owner(session):
    """Datasets created before memberships existed must not lock their creator out."""
    _user(session, "legacy")
    ds = Datasets(name="old", description="", dataset_type="image",
                  folder_path="/tmp/old", created_by="legacy")
    session.add(ds)
    session.commit()
    assert session.query(DatasetMembers).filter_by(dataset_id=ds.id).count() == 0

    legacy = _auth(session, "legacy")
    assert legacy.role_for(ds.id) is DatasetRole.OWNER
    assert legacy.has_permission(ds.id, Permission.DATASET_DELETE)


def test_is_admin_stays_in_sync_with_global_role(session):
    """`is_admin` is a derived view now; setting either way round must agree."""
    user = _user(session, "a", GlobalRole.ADMIN)
    session.commit()
    assert user.is_admin is True

    user.is_admin = False
    session.commit()
    assert user.global_role == GlobalRole.MEMBER.value
    assert session.query(Users).filter(Users.is_admin.is_(True)).count() == 0


# -- Dataset resolution ----------------------------------------------------

def test_dataset_resolution_from_every_entity_type(session):
    ds, img, mask, label, contour = _seed(session)
    assert dataset_id_for_image(img.id, session) == ds.id
    assert dataset_id_for_mask(mask.id, session) == ds.id
    assert dataset_id_for_contour(contour.id, session) == ds.id
    assert dataset_id_for_label(label.id, session) == ds.id
    assert resolve_dataset_id("dataset_id", ds.id, session) == ds.id


def test_dataset_resolution_returns_none_for_unknown_ids(session):
    _seed(session)
    assert dataset_id_for_mask(9999, session) is None
    assert dataset_id_for_contour(9999, session) is None
    assert resolve_dataset_id("dataset_id", 9999, session) is None


# -- Membership management -------------------------------------------------

def test_grant_and_revoke_membership(session):
    ds, *_ = _seed(session)
    _user(session, "ann")
    session.commit()

    members_db.grant_role(ds.id, "ann", DatasetRole.ANNOTATOR, granted_by="owner", db=session)
    assert _auth(session, "ann").role_for(ds.id) is DatasetRole.ANNOTATOR

    # Re-granting changes the role in place rather than duplicating the row.
    members_db.grant_role(ds.id, "ann", DatasetRole.REVIEWER, granted_by="owner", db=session)
    assert session.query(DatasetMembers).filter_by(dataset_id=ds.id, username="ann").count() == 1
    assert _auth(session, "ann").role_for(ds.id) is DatasetRole.REVIEWER

    assert members_db.revoke_membership(ds.id, "ann", session) is True
    assert _auth(session, "ann").role_for(ds.id) is None


def test_grant_cannot_mint_a_second_owner(session):
    ds, *_ = _seed(session)
    _user(session, "ann")
    session.commit()
    with pytest.raises(HTTPException) as exc:
        members_db.grant_role(ds.id, "ann", DatasetRole.OWNER, granted_by="owner", db=session)
    assert exc.value.status_code == 400


def test_owner_cannot_be_revoked_or_demoted(session):
    ds, *_ = _seed(session)
    with pytest.raises(HTTPException):
        members_db.revoke_membership(ds.id, "owner", session)
    with pytest.raises(HTTPException):
        members_db.grant_role(ds.id, "owner", DatasetRole.VIEWER, granted_by="owner", db=session)


def test_transfer_ownership_demotes_the_previous_owner(session):
    ds, *_ = _seed(session)
    _user(session, "successor")
    session.commit()

    members_db.transfer_ownership(ds.id, "successor", current_owner="owner", db=session)

    assert _auth(session, "successor").role_for(ds.id) is DatasetRole.OWNER
    assert _auth(session, "owner").role_for(ds.id) is DatasetRole.CURATOR
    # created_by is provenance and must survive the transfer untouched.
    assert session.query(Datasets).filter_by(id=ds.id).one().created_by == "owner"


# -- Invite links ----------------------------------------------------------

def test_invite_round_trip(session):
    ds, *_ = _seed(session)
    _user(session, "invitee")
    session.commit()

    invite, token = members_db.create_invite(
        ds.id, InviteCreate(role=DatasetRole.ANNOTATOR), created_by="owner", db=session)

    # Only the hash is stored, so the raw token cannot be recovered from the DB.
    assert token not in (invite.token_hash or "")
    assert len(invite.token_hash) == 64

    preview = members_db.preview_invite(token, "invitee", session)
    assert preview.dataset_id == ds.id
    assert preview.role is DatasetRole.ANNOTATOR
    assert preview.already_member is False

    dataset_id, role = members_db.accept_invite(token, "invitee", session)
    assert dataset_id == ds.id and role is DatasetRole.ANNOTATOR
    assert _auth(session, "invitee").has_permission(ds.id, Permission.ANNOTATION_CREATE)


def test_invite_never_downgrades_an_existing_role(session):
    ds, *_ = _seed(session)
    _user(session, "cur")
    _grant(session, ds.id, "cur", DatasetRole.CURATOR)

    _, token = members_db.create_invite(
        ds.id, InviteCreate(role=DatasetRole.VIEWER), created_by="owner", db=session)
    _, role = members_db.accept_invite(token, "cur", session)

    assert role is DatasetRole.CURATOR
    assert _auth(session, "cur").role_for(ds.id) is DatasetRole.CURATOR


def test_invite_respects_use_limit(session):
    ds, *_ = _seed(session)
    _user(session, "a")
    _user(session, "b")
    session.commit()

    _, token = members_db.create_invite(
        ds.id, InviteCreate(role=DatasetRole.VIEWER, max_uses=1), created_by="owner", db=session)
    members_db.accept_invite(token, "a", session)

    with pytest.raises(HTTPException) as exc:
        members_db.accept_invite(token, "b", session)
    assert exc.value.status_code == 410


def test_expired_invite_is_refused(session):
    ds, *_ = _seed(session)
    _user(session, "late")
    session.commit()

    invite, token = members_db.create_invite(
        ds.id, InviteCreate(role=DatasetRole.VIEWER), created_by="owner", db=session)
    invite.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
    session.commit()

    with pytest.raises(HTTPException) as exc:
        members_db.accept_invite(token, "late", session)
    assert exc.value.status_code == 410


def test_revoked_invite_is_refused(session):
    ds, *_ = _seed(session)
    _user(session, "late")
    session.commit()

    invite, token = members_db.create_invite(
        ds.id, InviteCreate(role=DatasetRole.VIEWER), created_by="owner", db=session)
    members_db.revoke_invite(invite.id, ds.id, session)

    with pytest.raises(HTTPException) as exc:
        members_db.accept_invite(token, "late", session)
    assert exc.value.status_code == 410


def test_invite_cannot_grant_ownership():
    with pytest.raises(ValueError):
        InviteCreate(role=DatasetRole.OWNER)


def test_unknown_token_is_a_404(session):
    _seed(session)
    _user(session, "x")
    session.commit()
    with pytest.raises(HTTPException) as exc:
        members_db.accept_invite("not-a-real-token", "x", session)
    assert exc.value.status_code == 404


# -- Review, rejections and separation of duties ---------------------------

def test_rejection_requires_a_note_only_for_other():
    RejectionCreate(reason=RejectionReason.BAD_OUTLINE)
    RejectionCreate(reason=RejectionReason.OTHER, note="the scale bar is wrong")
    with pytest.raises(ValueError):
        RejectionCreate(reason=RejectionReason.OTHER)
    with pytest.raises(ValueError):
        RejectionCreate(reason=RejectionReason.OTHER, note="   ")


def test_open_rejection_resets_the_annotate_and_review_phases(session):
    ds, img, mask, label, contour = _seed(session)
    mask.fully_annotated = True
    session.commit()
    # Submitted but nothing approved yet: annotating is done, reviewing has not begun.
    assert mask.annotate_status == "finished"
    assert mask.review_status == "not_started"

    rejection = asyncio.run(rejections_db.reject(
        mask.id, RejectionCreate(reason=RejectionReason.BAD_OUTLINE, contour_id=contour.id),
        username="owner", db=session))
    session.refresh(mask)

    # Sending work back reopens both mask phases.
    assert mask.annotate_status == "in_progress"
    assert mask.review_status == "in_progress"
    # Rejecting sends the mask back out of the review queue.
    assert mask.fully_annotated is False

    asyncio.run(rejections_db.resolve(rejection.id, username="owner", db=session))
    session.refresh(mask)
    assert mask.annotate_status == "in_progress"
    assert mask.review_status == "not_started"


def test_review_finishes_only_when_every_contour_is_approved_and_submitted(session):
    ds, img, mask, label, contour = _seed(session)
    owner = _auth(session, "owner")

    asyncio.run(contours_db.review_contour(contour.id, owner, session))
    session.refresh(mask)
    # Approved, but the mask was never submitted, so more objects may still appear.
    assert mask.annotate_status == "in_progress"
    assert mask.review_status == "in_progress"

    mask.fully_annotated = True
    session.commit()
    session.refresh(mask)
    assert mask.annotate_status == "finished"
    assert mask.review_status == "finished"


def test_phase_statuses_are_visible_to_sql_filters(session):
    """The hybrid_properties and their SQL expressions must agree."""
    ds, img, mask, label, contour = _seed(session)
    asyncio.run(rejections_db.reject(
        mask.id, RejectionCreate(reason=RejectionReason.MISSING_OBJECTS),
        username="owner", db=session))

    for column in (Masks.annotate_status, Masks.review_status):
        rows = session.query(Masks).filter(column == "in_progress").all()
        assert [row.id for row in rows] == [mask.id]


def test_rejection_must_belong_to_the_mask(session):
    ds, img, mask, label, contour = _seed(session)
    other_mask = Masks(image_id=img.id, fully_annotated=False, file_path="/tmp/m2.png")
    session.add(other_mask)
    session.commit()

    with pytest.raises(HTTPException) as exc:
        asyncio.run(rejections_db.reject(
            other_mask.id,
            RejectionCreate(reason=RejectionReason.BAD_OUTLINE, contour_id=contour.id),
            username="owner", db=session))
    assert exc.value.status_code == 400


def test_self_review_is_allowed_by_default(session):
    """A solo owner must still be able to finish their own dataset."""
    ds, img, mask, label, contour = _seed(session)
    owner = _auth(session, "owner")

    assert asyncio.run(contours_db.review_contour(contour.id, owner, session)) is True
    session.refresh(contour)
    assert [u.username for u in contour.reviewed_by] == ["owner"]


def test_independent_review_blocks_the_author(session):
    ds, img, mask, label, contour = _seed(session)
    ds.require_independent_review = True
    session.commit()
    owner = _auth(session, "owner")

    with pytest.raises(PermissionError):
        asyncio.run(contours_db.review_contour(contour.id, owner, session, strict=True))

    # Non-strict is the label-edit path: it declines quietly instead of failing.
    assert asyncio.run(contours_db.review_contour(contour.id, owner, session, strict=False)) is False
    session.refresh(contour)
    assert contour.reviewed_by == []


def test_independent_review_allows_a_different_reviewer(session):
    ds, img, mask, label, contour = _seed(session)
    ds.require_independent_review = True
    _user(session, "reviewer")
    _grant(session, ds.id, "reviewer", DatasetRole.REVIEWER)

    reviewer = _auth(session, "reviewer")
    assert asyncio.run(contours_db.review_contour(contour.id, reviewer, session)) is True
    session.refresh(contour)
    assert [u.username for u in contour.reviewed_by] == ["reviewer"]


def test_review_is_idempotent_and_removable(session):
    """Both used to be broken: a Pydantic user was compared against ORM rows."""
    ds, img, mask, label, contour = _seed(session)
    owner = _auth(session, "owner")

    asyncio.run(contours_db.review_contour(contour.id, owner, session))
    asyncio.run(contours_db.review_contour(contour.id, owner, session))
    session.refresh(contour)
    assert len(contour.reviewed_by) == 1, "reviewing twice must not duplicate the reviewer"

    asyncio.run(contours_db.remove_review(contour.id, owner, session))
    session.refresh(contour)
    assert contour.reviewed_by == [], "a review must actually be removable"
