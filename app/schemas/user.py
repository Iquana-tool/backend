from pydantic import BaseModel, Field


class User(BaseModel):
    name: str = Field(..., description="User name")
    is_admin: bool = Field(..., description="User is admin")

    owned_datasets: list[int] = Field(..., description="User's owned datasets")
    accessible_datasets: list[int] = Field(..., description="Datasets shared with user")

    @classmethod
    def from_query(cls, user_db):
        return cls(
            name=user_db.username,
            is_admin=user_db.is_admin,
            owned_datasets=[ds.id for ds in user_db.owned_datasets],
            accessible_datasets=[ds.id for ds in user_db.accessible_datasets],
        )






