from pydantic import BaseModel, Field


class User(BaseModel):
    username: str = Field(...)
    is_admin: bool = Field(..., description="User is admin")

    owned_datasets: list[int] = Field(..., description="User's owned datasets")
    accessible_datasets: list[int] = Field(..., description="Datasets shared with user")

    @classmethod
    def from_query(cls, user_db):
        return cls(
            username=user_db.username,
            is_admin=user_db.is_admin,
            owned_datasets=[ds.id for ds in user_db.owned_datasets],
            accessible_datasets=[ds.id for ds in user_db.accessible_datasets],
        )

    @property
    def available_datasets(self) -> list[int]:
        return self.owned_datasets + self.accessible_datasets


class System(User):
    """ System user function for models and system-initiated actions that require user authentication. """
    username: str = Field(...)
    is_admin: bool = False
    owned_datasets: list[int] = []
    accessible_datasets: list[int] = []






