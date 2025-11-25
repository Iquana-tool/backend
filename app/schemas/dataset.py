from pydantic import BaseModel, Field

from app.database.users import Users


class Dataset(BaseModel):
    id: int
    name: str
    description: str
    dataset_type: str = Field(...)
    created_by: str = Field(...)

    @classmethod
    def from_query(cls, query):
        return cls(
            id=query.id,
            name=query.name,
            description=query.description,
            dataset_type=query.dataset_type,
            created_by=query.created_by,
        )

