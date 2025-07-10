import os
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Tuple


class TrainingRequest(BaseModel):
    dataset_id: int = Field(default=1, title="Dataset ID")
    model_identifier: str = Field(default="unet", description="Identifier for the model to be trained. "
                                                      "Should be got by GET /models/get_trainable_models")
    model_id_db: int = Field(description="ID of the model in the database. Must be provided.")
    restart: bool = Field(default=True, description="Whether to restart training from scratch. If set to True, "
                                                     "the model will be trained from scratch, else it will continue "
                                                     "training from the last checkpoint.")
    epochs: int = Field(default=50, description="Number of epochs to train the model.")
    batch_size: int = Field(default=64, description="Batch size to train the model.")
    lr: float = Field(default=0.0001, description="Learning rate to train the model.")
    augment: bool = Field(default=True, description="Whether to augment the dataset. This should be done for small "
                                                    "datasets, but can be left out for bigger datasets.")
    image_size: Optional[Tuple[int, int]] = Field(default=(256, 256), description="Image size to use. Smaller values "
                                                                                  "may lead to faster training, "
                                                                                  "but may also lead to "
                                                                                  "loss of information.")
    num_classes: int = Field(default=2, description="Number of classes that are present on the masks. Corresponds to the "
                                                    "number of labels.")
    in_channels: int = 3
    early_stopping: bool = Field(default=True, description="Whether to use early stopping during training. "
                                                           "This will stop training if the validation loss "
                                                           "does not improve for 5 epochs.")
