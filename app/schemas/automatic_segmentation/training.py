import os
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Tuple, Union


class TrainingRequest(BaseModel):
    dataset_id: int = Field(default=1, title="Dataset ID")
    model_identifier: Union[int, str] = Field(description="Start training from either a base model, if given a model "
                                                    "registry key, or from a trained model checkpoint if given "
                                                   "a model identifier number."
                                                          "Important: Numbers must be given as int and not as string!")
    overwrite: bool = Field(default=False, title="Overwrite the existing model or save as a new model. Note: "
                                                 "Base models cannot be overwritten.")
    epochs: int = Field(default=50, description="Number of epochs to train the model.")
    augment: bool = Field(default=True, description="Whether to augment the dataset. This should be done for small "
                                                    "datasets, but can be left out for bigger datasets.")
    image_size: Optional[Tuple[int, int]] = Field(default=(256, 256), description="Image size to use. Smaller values "
                                                                                  "may lead to faster training, "
                                                                                  "but may also lead to "
                                                                                  "loss of information.")
    early_stopping: bool = Field(default=True, description="Whether to use early stopping during training. "
                                                           "This will stop training if the validation loss "
                                                           "does not improve for 5 epochs.")
