"""Requests and responses for mapping a model's label space onto a dataset's labels."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from iquana_toolbox.schemas.model_info import ModelClass

#: What a model class resolves to in a dataset. ``unmapped`` predictions keep no label.
ResolvedAction = Literal["map", "discard", "unmapped"]


class ClassMappingRead(BaseModel):
    """One model class and what it currently means in the dataset."""

    model_class: ModelClass
    action: ResolvedAction
    label_id: Optional[int] = Field(default=None, description="The dataset label, when action is 'map'.")
    attribute_id: Optional[int] = Field(
        default=None, description="An attribute the class also sets on its objects (saved choices only).")
    attribute_value: Optional[str] = Field(default=None, description="The value it sets.")
    source: Literal["saved", "suggested", "none"] = Field(
        ...,
        description="'saved': someone chose this. 'suggested': no choice saved, and a dataset "
                    "label has the class's name, so it is used. 'none': neither.",
    )


class ModelLabelMappingRead(BaseModel):
    dataset_id: int
    model_registry_key: str
    model_name: str
    classes: list[ClassMappingRead]


class ClassMappingWrite(BaseModel):
    """What one model class should mean in the dataset from now on."""

    model_class: str = Field(..., description="Key of the class in the model's label space.")
    action: Literal["map", "discard", "create", "unmapped"] = Field(
        ...,
        description="'map' to an existing label, 'discard' its predictions, 'create' a new label "
                    "for it (nested under its parent class's label, if that is mapped), or "
                    "'unmapped' to drop a saved choice and fall back to the name suggestion.",
    )
    label_id: Optional[int] = Field(default=None, description="Required for 'map'.")
    label_name: Optional[str] = Field(
        default=None, description="Name of the label 'create' makes; the class name if omitted."
    )
    attribute_id: Optional[int] = Field(
        default=None,
        description="With 'map' or 'create': an attribute every object of this class also gets, "
                    "e.g. species for a species detector. Values arrive as suggestions.",
    )
    attribute_value: Optional[str] = Field(
        default=None, description="The value; the class name if omitted."
    )

    @model_validator(mode="after")
    def _label_for_map(self) -> "ClassMappingWrite":
        if self.action == "map" and self.label_id is None:
            raise ValueError(f"Mapping class '{self.model_class}' needs a label_id.")
        if self.attribute_id is not None and self.action not in ("map", "create"):
            raise ValueError(f"Class '{self.model_class}' can only set an attribute when it is "
                             f"mapped to a label.")
        return self


class ModelLabelMappingWrite(BaseModel):
    classes: list[ClassMappingWrite] = Field(
        ..., description="Changes, by class. Classes not listed keep what they have."
    )
