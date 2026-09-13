from sqlalchemy import Column, String, Boolean, case
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship

from app.database import database
from app.database.contours import reviewer_contour_association
from app.schemas.permissions import GlobalRole


class Users(database):
    """ Represents our users. """
    __tablename__ = "users"
    username = Column(String, nullable=False, unique=True, primary_key=True)  # Ensure usernames are unique
    hashed_password = Column(String, nullable=False)  # Store hashed passwords only
    # Platform-level role. Dataset-level rights live on `dataset_members.role`.
    global_role = Column(String(20), nullable=False, default=GlobalRole.MEMBER.value)
    # Accounts can be switched off without deleting the annotations they authored.
    is_active = Column(Boolean, nullable=False, default=True)

    owned_datasets = relationship("Datasets",
                                  back_populates="owner")
    memberships = relationship("DatasetMembers",
                               foreign_keys="DatasetMembers.username",
                               back_populates="user",
                               cascade="all, delete-orphan")
    # Read-only convenience view over the membership rows. `dataset_members` has two
    # foreign keys into `users` (the member and whoever granted the role), so the
    # join has to be spelled out.
    accessible_datasets = relationship("Datasets",
                                       secondary="dataset_members",
                                       primaryjoin="Users.username == DatasetMembers.username",
                                       secondaryjoin="Datasets.id == DatasetMembers.dataset_id",
                                       viewonly=True)
    reviewed_objects = relationship("Contours",
                                    secondary=reviewer_contour_association,
                                    back_populates="reviewed_by")

    @hybrid_property
    def is_admin(self) -> bool:
        """Kept so existing call sites and fixtures keep working.

        `global_role` is the single source of truth; this is a derived view of it
        rather than a second column that can drift out of sync.
        """
        return self.global_role == GlobalRole.ADMIN

    @is_admin.setter
    def is_admin(self, value: bool):
        self.global_role = GlobalRole.ADMIN.value if value else GlobalRole.MEMBER.value

    @is_admin.expression
    def is_admin(cls):
        return case((cls.global_role == GlobalRole.ADMIN.value, True), else_=False)
