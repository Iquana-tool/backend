from typing import Optional, Tuple, Union

from pydantic import BaseModel, Field


class TrainingRequest(BaseModel):
    dataset_id: int = Field(default=1, title="Dataset ID")
    model_identifier: Union[str] = Field(description="Start training from either a general model, if given a model "
                                                    "registry key, or from a trained model checkpoint if given "
                                                   "a model identifier number."
                                                          "Important: Numbers must be given as int and not as string!")
    epochs: int = Field(default=50, title="The number of epochs the model should be trained for.")
    augment: bool = Field(default=True, description="Whether to augment the dataset. This should be done for small "
                                                    "datasets, but can be left out for bigger datasets.")
    image_size: Optional[Tuple[int, int]] = Field(default=(256, 256), description="Image size to use. Smaller values "
                                                                                  "may lead to faster training, "
                                                                                  "but may also lead to "
                                                                                  "loss of information.")
    early_stopping: bool = Field(default=True, description="Whether to use early stopping during training. "
                                                           "This will stop training if the validation loss "
                                                           "does not improve for 5 epochs.")
