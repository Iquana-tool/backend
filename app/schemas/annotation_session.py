from typing import Union
from pydantic import BaseModel, Field, field_validator
from enum import StrEnum


class ClientMessageType(StrEnum):
    FOCUS_IMAGE = "focus_image"  # Focus on a specific region of the image
    UNFOCUS_IMAGE = "unfocus_image"  # Revert to the original image
    PROMPTED_SELECT_MODEL = "prompted_select_model"  # Select a model for prompted segmentation
    PROMPTED_SEGMENTATION = "prompted_segmentation"  # Perform prompted segmentation
    AUTOMATIC_SELECT_MODEL = "automatic_select_model"  # Select a model for automatic segmentation
    AUTOMATIC_SEGMENTATION = "automatic_segmentation"  # Perform automatic segmentation
    COMPLETION_SELECT_MODEL = "completion_select_model"  # Select a model for mask completion
    COMPLETION_ENABLE = "completion_enable"  # Enable mask completion
    COMPLETION_DISABLE = "completion_disable"  # Disable mask completion
    OBJECT_ADD_MANUAL = "object_add_manual"  # Add a new object to the annotation session, if it was manually created
    OBJECT_FINALISE = "object_finalise"  # Mark a temporary object as not temporary anymore.
    OBJECT_DELETE = "object_delete"  # Delete an object from the annotation session
    OBJECT_MODIFY = "object_modify"  # Modify an existing object
    # These are too ambitious for now. Needs state management
    #UNDO = "undo"  # Undo the last action
    #REDO = "redo"  # Redo the last undone action
    OBJECT_CONFLICT_RESOLUTION = "object_conflict_resolution"  # Resolve conflicts between objects
    FINISH_ANNOTATION = "finish_annotation"  # Mark the mask as finished and save to DB
    ERROR = "error"


class ServerMessageType(StrEnum):
    SESSION_INITIALIZED = "session_initialized"  # Session has been initialized, gives info about running backends
    OBJECT_ADDED = "object_added"  # Send a newly added object
    OBJECT_REMOVED = "object_removed" # Tell which object has been deleted
    OBJECT_MODIFIED = "object_modified" # Tell which object has been modified and what has been modified.
    PROMPTED_SEGMENTATION_RESULT = "prompted_segmentation_result"  # Result
    AUTOMATIC_SEGMENTATION_RESULT = "automatic_segmentation_result"  # Result
    COMPLETION_RESULT = "completion_result"  # Result
    OBJECT_CONFLICT_PROMPT = "object_conflict_prompt"  # Prompt the user on how to resolve an object conflict.
    ERROR = "error"  # An error occurred, gives info about the error


# Precompute set of possible strings to speed up validation
possible_client_msg_types = {e.value for e in ClientMessageType}
possible_server_msg_types = {e.value for e in ServerMessageType}


class Message(BaseModel):
    id: str = Field(..., description="Unique message ID to correlate messages.")
    type: str = Field(..., description="Identifier for what this message should trigger or what is being delivered.")
    message: str | None = Field(default=None, description="Additional human readable message.")
    success: bool = Field(..., description="Whether the message was successful. Can be ignored for requests.")
    data: Union[dict, list, None] = Field(default_factory=dict, description="Data to send.")


class ClientMessage(Message):
    @field_validator("type")
    def validate_type(cls, v):
        ClientMessageType(v)  # Raises error if v is not in the enum
        return v


class ServerMessage(Message):
    @field_validator("type")
    def validate_type(cls, v):
        ServerMessageType(v)  # Raises error if v is not in the enum
        return v
