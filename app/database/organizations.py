"""Organisations, their teams, and datasets shared with a team.

An organisation groups accounts -- an institute, a lab. Teams sit inside one
organisation and can nest through ``parent_team_id``, which is how a department
holding several teams is modelled: a department is a team with child teams.

A team can be given a role on a dataset (``dataset_team_grants``); every member of
that team, *and of every team below it*, then holds that role on the dataset.
Belonging to an organisation or a team grants nothing by itself.

Organisation and team membership rows cascade with the account, and a username
change follows them (ON UPDATE CASCADE), like every other key into ``users``.
"""
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    false,
    text,
    true,
)
from sqlalchemy.orm import Session

from app.database import database
from app.database.users import utc_now
from app.schemas.permissions import DatasetRole, OrganizationRole, TeamRole


class Organizations(database):
    __tablename__ = "organizations"
    __table_args__ = (
        # At most one organisation is the default, the one new accounts join.
        Index("uq_organizations_default", "is_default", unique=True,
              postgresql_where=text("is_default"), sqlite_where=text("is_default")),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    is_default = Column(Boolean, nullable=False, default=False, server_default=false())
    # Whether members' personal API keys are used for work in this organisation. Off
    # when the organisation needs its data to reach only a provider it chose -- the
    # key decides where prompts are sent.
    allow_personal_keys = Column(Boolean, nullable=False, default=True, server_default=true())
    created_at = Column(DateTime, nullable=False, default=utc_now)


class OrganizationMembers(database):
    __tablename__ = "organization_members"

    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"),
                             primary_key=True)
    # Indexed on its own: the primary key leads with the organisation, and "which
    # organisations is this account in" is asked on every request.
    username = Column(String, ForeignKey("users.username", ondelete="CASCADE", onupdate="CASCADE"),
                      primary_key=True, index=True)
    role = Column(String(20), nullable=False, default=OrganizationRole.MEMBER.value)
    joined_at = Column(DateTime, nullable=False, default=utc_now)


class Teams(database):
    __tablename__ = "teams"
    __table_args__ = (
        UniqueConstraint("organization_id", "name", name="uq_teams_organization_name"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"),
                             nullable=False, index=True)
    # The team this one sits inside, e.g. its department; always in the same
    # organisation. SET NULL: removing a department leaves its teams standing at the
    # top level, rather than deleting them and every grant they hold.
    parent_team_id = Column(Integer, ForeignKey("teams.id", ondelete="SET NULL"),
                            nullable=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(String(255), nullable=True)
    created_at = Column(DateTime, nullable=False, default=utc_now)


class TeamMembers(database):
    __tablename__ = "team_members"

    team_id = Column(Integer, ForeignKey("teams.id", ondelete="CASCADE"), primary_key=True)
    username = Column(String, ForeignKey("users.username", ondelete="CASCADE", onupdate="CASCADE"),
                      primary_key=True, index=True)
    role = Column(String(20), nullable=False, default=TeamRole.MEMBER.value)
    joined_at = Column(DateTime, nullable=False, default=utc_now)


class DatasetTeamGrants(database):
    """A team's role on a dataset; the team-level counterpart of `DatasetMembers`.

    Carries the same ``extra_permissions`` / ``denied_permissions`` escape hatch. The
    role is at most curator (`TEAM_GRANT_MAX_ROLE`), and the team belongs to the
    dataset's organisation.
    """

    __tablename__ = "dataset_team_grants"

    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.id", ondelete="CASCADE"),
                     primary_key=True, index=True)
    role = Column(String(20), nullable=False, default=DatasetRole.VIEWER.value)
    extra_permissions = Column(JSON, nullable=False, default=list)
    denied_permissions = Column(JSON, nullable=False, default=list)
    granted_by = Column(String, ForeignKey("users.username", ondelete="SET NULL", onupdate="CASCADE"),
                        nullable=True)
    granted_at = Column(DateTime, nullable=False, default=utc_now)


def teams_reaching(username: str, db: Session) -> set[int]:
    """Every team whose dataset grants reach this account.

    That is the teams it belongs to and every team above them, so a grant to a
    department reaches the people in the department's teams. Walked in Python over
    the organisations' (id, parent) pairs -- a handful of rows -- rather than with a
    recursive query, and the walk stops at a team it has seen, so a cycle in the
    tree cannot hang a request.
    """
    direct = [team_id for (team_id,) in db.query(TeamMembers.team_id).filter_by(username=username)]
    if not direct:
        return set()
    organizations = db.query(Teams.organization_id).filter(Teams.id.in_(direct)).distinct()
    parent_of = dict(db.query(Teams.id, Teams.parent_team_id)
                     .filter(Teams.organization_id.in_(organizations)))
    reached: set[int] = set()
    for team_id in direct:
        while team_id is not None and team_id not in reached:
            reached.add(team_id)
            team_id = parent_of.get(team_id)
    return reached


def team_grants_for(username: str, db: Session) -> list[DatasetTeamGrants]:
    """The dataset grants this account holds through its teams."""
    teams = teams_reaching(username, db)
    if not teams:
        return []
    return db.query(DatasetTeamGrants).filter(DatasetTeamGrants.team_id.in_(teams)).all()
