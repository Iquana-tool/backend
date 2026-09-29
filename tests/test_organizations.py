"""Checks for organisations, teams, and datasets shared with a team.

The fixture is one small organisation, "Lab", as it would look in use:

* ann -- organisation admin, owner of the dataset "Reef";
* dan -- organisation admin with no role on any dataset;
* bob -- member, in team "Corals", which sits inside the department "Biology";
* cid -- member, in team "Imaging", a top-level team;
* eve -- member of a different organisation, "Other".

The caller is resolved with ``AuthenticatedUser.from_query`` against the real rows,
so every check below goes through the same permission merging a request does.
"""
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database, get_session, import_models
from app.database.dataset_members import DatasetMembers
from app.database.datasets import Datasets
from app.database.organizations import (
    DatasetTeamGrants,
    OrganizationMembers,
    Organizations,
    TeamMembers,
    Teams,
)
from app.database.users import Users
from app.routes.general.members import router as members_router
from app.routes.general.organizations import router as organizations_router
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import DatasetRole, GlobalRole, Permission
from app.services.auth import get_current_user
from app.services.database_access.organizations import (
    join_default_organization,
    organization_for_new_dataset,
)

import_models()


@pytest.fixture
def ctx(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'organizations.db'}")
    database.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    db = Session()
    for username in ("root", "ann", "dan", "bob", "cid", "eve", "new"):
        db.add(Users(username=username, hashed_password="x",
                     global_role=GlobalRole.ADMIN.value if username == "root" else GlobalRole.MEMBER.value))
    lab = Organizations(name="Lab", is_default=True)
    other = Organizations(name="Other")
    db.add_all([lab, other])
    db.flush()
    db.add_all([
        OrganizationMembers(organization_id=lab.id, username="ann", role="admin"),
        OrganizationMembers(organization_id=lab.id, username="dan", role="admin"),
        OrganizationMembers(organization_id=lab.id, username="bob", role="member"),
        OrganizationMembers(organization_id=lab.id, username="cid", role="member"),
        OrganizationMembers(organization_id=other.id, username="eve", role="admin"),
    ])
    biology = Teams(organization_id=lab.id, name="Biology")
    imaging = Teams(organization_id=lab.id, name="Imaging")
    elsewhere = Teams(organization_id=other.id, name="Elsewhere")
    db.add_all([biology, imaging, elsewhere])
    db.flush()
    corals = Teams(organization_id=lab.id, name="Corals", parent_team_id=biology.id)
    db.add(corals)
    db.flush()
    db.add_all([
        TeamMembers(team_id=corals.id, username="bob", role="member"),
        TeamMembers(team_id=imaging.id, username="cid", role="member"),
    ])
    reef = Datasets(name="Reef", dataset_type="image", folder_path="/tmp/reef",
                    created_by="ann", organization_id=lab.id)
    db.add(reef)
    db.flush()
    db.add(DatasetMembers(dataset_id=reef.id, username="ann", role=DatasetRole.OWNER.value,
                          extra_permissions=[], denied_permissions=[], granted_by="ann"))
    db.commit()
    ids = {"lab": lab.id, "other": other.id, "biology": biology.id, "corals": corals.id,
           "imaging": imaging.id, "elsewhere": elsewhere.id, "reef": reef.id}
    db.close()

    app = FastAPI()
    app.include_router(organizations_router)
    app.include_router(members_router)
    caller = {"username": "ann"}

    def override_session():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    def override_user(session=Depends(override_session)):
        return AuthenticatedUser.from_query(session.query(Users).filter_by(username=caller["username"]).one())

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_current_user] = override_user

    def user(username: str) -> AuthenticatedUser:
        session = Session()
        try:
            return AuthenticatedUser.from_query(session.query(Users).filter_by(username=username).one())
        finally:
            session.close()

    def as_caller(username: str) -> TestClient:
        caller["username"] = username
        return client

    client = TestClient(app)
    yield {"as": as_caller, "user": user, "ids": ids, "Session": Session}
    engine.dispose()


def _grant(ctx, team: str, role: str, **extra):
    return ctx["as"]("ann").put(f"/datasets/{ctx['ids']['reef']}/teams/{ctx['ids'][team]}",
                                json={"role": role, **extra})


# -- How team grants reach people ----------------------------------------------

def test_a_department_grant_reaches_the_teams_below_it(ctx):
    assert _grant(ctx, "biology", "viewer").status_code == 200
    reef = ctx["ids"]["reef"]

    bob = ctx["user"]("bob")
    assert bob.role_for(reef) is DatasetRole.VIEWER
    assert bob.memberships[reef].via_teams == [ctx["ids"]["biology"]]
    assert reef in bob.available_datasets
    # cid's team is not under Biology.
    assert ctx["user"]("cid").role_for(reef) is None


def test_grants_add_up_and_a_denial_stays_with_its_grant(ctx):
    reef = ctx["ids"]["reef"]
    session = ctx["Session"]()
    session.add(DatasetMembers(dataset_id=reef, username="bob", role=DatasetRole.REVIEWER.value,
                               extra_permissions=[], granted_by="ann",
                               denied_permissions=[Permission.ANNOTATION_CREATE.value]))
    session.commit()
    session.close()
    _grant(ctx, "corals", "annotator")

    bob = ctx["user"]("bob")
    # The higher role is reported...
    assert bob.role_for(reef) is DatasetRole.REVIEWER
    # ...and the team's annotator grant gives back what the direct grant withheld.
    assert bob.has_permission(reef, Permission.ANNOTATION_CREATE)
    assert bob.has_permission(reef, Permission.REVIEW_APPROVE)


def test_a_team_grant_stops_at_curator(ctx):
    assert _grant(ctx, "corals", "owner").status_code == 422


def test_a_team_of_another_organisation_cannot_be_granted(ctx):
    response = _grant(ctx, "elsewhere", "viewer")
    assert response.status_code == 400
    assert "own organisation" in response.json()["detail"]


def test_granting_a_team_needs_the_member_grant_permission(ctx):
    _grant(ctx, "corals", "curator")  # curator lacks member.grant
    response = ctx["as"]("bob").put(f"/datasets/{ctx['ids']['reef']}/teams/{ctx['ids']['imaging']}",
                                    json={"role": "viewer"})
    assert response.status_code == 403


def test_revoking_a_team_grant_takes_its_access_away(ctx):
    _grant(ctx, "corals", "annotator")
    response = ctx["as"]("ann").delete(f"/datasets/{ctx['ids']['reef']}/teams/{ctx['ids']['corals']}")
    assert response.status_code == 200
    assert ctx["user"]("bob").role_for(ctx["ids"]["reef"]) is None


def test_team_grants_are_listed_with_their_team_names(ctx):
    _grant(ctx, "corals", "annotator")
    response = ctx["as"]("ann").get(f"/datasets/{ctx['ids']['reef']}/teams")
    assert [(t["team_name"], t["role"]) for t in response.json()["teams"]] == [("Corals", "annotator")]


# -- Running an organisation ---------------------------------------------------

def test_organisation_admins_create_teams_and_members_cannot(ctx):
    lab = ctx["ids"]["lab"]
    created = ctx["as"]("ann").post(f"/organizations/{lab}/teams",
                                    json={"name": "Microscopy", "parent_team_id": ctx["ids"]["biology"]})
    assert created.status_code == 201
    assert created.json()["team"]["parent_team_id"] == ctx["ids"]["biology"]

    assert ctx["as"]("bob").post(f"/organizations/{lab}/teams", json={"name": "Mine"}).status_code == 403


def test_team_names_are_unique_within_an_organisation(ctx):
    response = ctx["as"]("ann").post(f"/organizations/{ctx['ids']['lab']}/teams", json={"name": "Corals"})
    assert response.status_code == 409


def test_a_maintainer_runs_their_own_team_only(ctx):
    ids = ctx["ids"]
    ctx["as"]("ann").put(f"/teams/{ids['corals']}/members/bob", json={"role": "maintainer"})
    bob = ctx["as"]("bob")

    assert bob.put(f"/teams/{ids['corals']}/members/cid", json={"role": "member"}).status_code == 200
    assert bob.put(f"/teams/{ids['imaging']}/members/bob", json={"role": "member"}).status_code == 403
    # Only the organisation's members can join its teams.
    assert bob.put(f"/teams/{ids['corals']}/members/eve", json={"role": "member"}).status_code == 400


def test_a_team_cannot_sit_inside_its_own_sub_team(ctx):
    ids = ctx["ids"]
    response = ctx["as"]("ann").patch(f"/teams/{ids['biology']}", json={"parent_team_id": ids["corals"]})
    assert response.status_code == 400


def test_a_parent_team_must_be_in_the_same_organisation(ctx):
    ids = ctx["ids"]
    response = ctx["as"]("ann").patch(f"/teams/{ids['corals']}", json={"parent_team_id": ids["elsewhere"]})
    assert response.status_code == 400


def test_deleting_a_department_leaves_its_teams_at_the_top(ctx):
    ids = ctx["ids"]
    assert ctx["as"]("ann").delete(f"/teams/{ids['biology']}").status_code == 200

    session = ctx["Session"]()
    corals = session.get(Teams, ids["corals"])
    assert corals is not None and corals.parent_team_id is None
    session.close()


def test_leaving_the_organisation_leaves_its_teams_and_their_access(ctx):
    _grant(ctx, "corals", "annotator")
    lab = ctx["ids"]["lab"]
    assert ctx["as"]("ann").delete(f"/organizations/{lab}/members/bob").status_code == 200

    bob = ctx["user"]("bob")
    assert lab not in bob.organizations
    assert bob.role_for(ctx["ids"]["reef"]) is None


def test_the_last_admin_can_neither_leave_nor_be_demoted(ctx):
    lab = ctx["ids"]["lab"]
    ann = ctx["as"]("ann")
    assert ann.delete(f"/organizations/{lab}/members/dan").status_code == 200

    assert ann.delete(f"/organizations/{lab}/members/ann").status_code == 400
    assert ann.put(f"/organizations/{lab}/members/ann", json={"role": "member"}).status_code == 400


def test_outsiders_cannot_see_an_organisations_people(ctx):
    lab = ctx["ids"]["lab"]
    assert ctx["as"]("eve").get(f"/organizations/{lab}/members").status_code == 403
    assert ctx["as"]("bob").get(f"/organizations/{lab}/members").status_code == 200


def test_organisations_are_listed_for_their_members_and_all_for_platform_admins(ctx):
    assert [o["name"] for o in ctx["as"]("bob").get("/organizations").json()["organizations"]] == ["Lab"]
    assert [o["name"] for o in ctx["as"]("root").get("/organizations").json()["organizations"]] == ["Lab", "Other"]


def test_only_platform_admins_create_organisations(ctx):
    assert ctx["as"]("ann").post("/organizations", json={"name": "New"}).status_code == 403
    assert ctx["as"]("root").post("/organizations", json={"name": "New"}).status_code == 201


def test_there_is_only_ever_one_default_organisation(ctx):
    ids = ctx["ids"]
    assert ctx["as"]("ann").patch(f"/organizations/{ids['lab']}", json={"is_default": False}).status_code == 403
    assert ctx["as"]("root").patch(f"/organizations/{ids['other']}", json={"is_default": True}).status_code == 200

    session = ctx["Session"]()
    assert [o.name for o in session.query(Organizations).filter_by(is_default=True)] == ["Other"]
    session.close()


def test_an_organisation_with_datasets_cannot_be_deleted(ctx):
    assert ctx["as"]("root").delete(f"/organizations/{ctx['ids']['lab']}").status_code == 409
    assert ctx["as"]("root").delete(f"/organizations/{ctx['ids']['other']}").status_code == 200


# -- Organisation admins and datasets ------------------------------------------

def test_an_organisation_admin_can_reassign_an_owner_without_seeing_the_data(ctx):
    ids = ctx["ids"]
    dan = ctx["as"]("dan")
    listed = dan.get(f"/organizations/{ids['lab']}/datasets").json()["datasets"]
    assert listed == [{"id": ids["reef"], "name": "Reef", "owners": ["ann"]}]
    # Being an admin of the organisation gives no role on its datasets.
    assert ctx["user"]("dan").permissions_for(ids["reef"]) == frozenset()

    response = dan.post(f"/organizations/{ids['lab']}/datasets/{ids['reef']}/transfer_ownership",
                        params={"new_owner": "bob"})
    assert response.status_code == 200
    assert ctx["user"]("bob").role_for(ids["reef"]) is DatasetRole.OWNER
    assert ctx["user"]("ann").role_for(ids["reef"]) is DatasetRole.CURATOR


def test_ownership_only_goes_to_someone_in_the_organisation(ctx):
    ids = ctx["ids"]
    response = ctx["as"]("dan").post(
        f"/organizations/{ids['lab']}/datasets/{ids['reef']}/transfer_ownership", params={"new_owner": "eve"})
    assert response.status_code == 400


def test_moving_a_dataset_out_drops_its_team_grants(ctx):
    _grant(ctx, "imaging", "viewer")
    ids = ctx["ids"]
    response = ctx["as"]("ann").put(f"/datasets/{ids['reef']}/organization", json={"organization_id": None})

    assert response.json()["team_grants_removed"] == 1
    assert ctx["user"]("cid").role_for(ids["reef"]) is None
    session = ctx["Session"]()
    assert session.query(DatasetTeamGrants).count() == 0
    assert session.get(Datasets, ids["reef"]).organization_id is None
    session.close()


def test_a_dataset_only_moves_into_an_organisation_the_caller_is_in(ctx):
    ids = ctx["ids"]
    response = ctx["as"]("ann").put(f"/datasets/{ids['reef']}/organization",
                                    json={"organization_id": ids["other"]})
    assert response.status_code == 403


# -- New accounts and new datasets ---------------------------------------------

def test_a_new_dataset_lands_in_the_callers_organisation(ctx):
    ids = ctx["ids"]
    session = ctx["Session"]()
    assert organization_for_new_dataset(ctx["user"]("bob"), None, session) == ids["lab"]
    # No organisation at all: the dataset is personal.
    assert organization_for_new_dataset(ctx["user"]("new"), None, session) is None
    session.close()


def test_a_new_dataset_can_only_be_put_into_ones_own_organisation(ctx):
    from fastapi import HTTPException

    session = ctx["Session"]()
    with pytest.raises(HTTPException) as refused:
        organization_for_new_dataset(ctx["user"]("eve"), ctx["ids"]["lab"], session)
    assert refused.value.status_code == 403
    session.close()


def test_a_new_account_joins_the_default_organisation(ctx):
    session = ctx["Session"]()
    session.add(Users(username="fay", hashed_password="x"))
    join_default_organization("fay", session)
    session.commit()
    session.close()

    assert ctx["user"]("fay").organizations == {ctx["ids"]["lab"]: "member"}
