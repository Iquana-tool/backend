from pydantic import BaseModel, ValidationError
from fastapi import HTTPException
from typing import Type


def validate_request(request: dict, schema: Type[BaseModel]) -> BaseModel:
    """ Validate the request against the schema. """
    try:
        return schema(**request)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors())
