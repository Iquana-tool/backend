from enum import StrEnum
from functools import cached_property
from logging import getLogger

from iquana_toolbox.schemas.database.contour_hierarchy import ContourHierarchy
from pydantic import field_validator, BaseModel, Field, PrivateAttr
from pydantic_core.core_schema import ValidationInfo

from app.database import get_context_session
from app.database.images import Images
from app.database.masks import Masks
from app.services.ai_services.base_service import BaseService

logger = getLogger(__name__)


class Backends(StrEnum):
    PROMPTED_SEGMENTATION = "prompted_segmentation"
    SUGGESTION_SEGMENTATION = "suggestion_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"


class AnnotationSessionState(BaseModel):
    """ A class to track the state of the annotation session.

    A session belongs to a user, not to an image: the client opens one socket and points
    it at whichever image it is showing (see ``switch_to_image``). It may therefore start
    without an image at all, which is why ``image_id`` is optional.
    """
    image_id: int | None = Field(default=None, title="Image ID",
                                 description="The image currently being annotated. None until the "
                                             "client points the session at one.")
    mask_id: int | None = Field(default=None, title="Mask ID",
                                description="The mask id. If None, will be validated and the correct"
                                            "id is fetched from the db.")
    # Always the authenticated username, resolved from the connection's bearer
    # token rather than from the URL. Contours created in this session are
    # attributed to it.
    user_id: str = Field(..., title="User ID")
    dataset_id: int | None = Field(default=None, title="Dataset ID",
                                   description="Dataset the image belongs to, resolved once at "
                                               "connection time for permission checks.")
    contour_hierarchy: ContourHierarchy | None = Field(default=None, title="Contour Hierarchy")
    focussed_contour_id: int | None = Field(default=None, title="Contour ID")
    refinement_contour_id: int | None = Field(default=None, title="Contour ID")
    _running_backends: dict[str, BaseService] = PrivateAttr(
        default_factory=dict  # This should not be serialized
    )
    _failed_backends: dict[str, BaseService] = PrivateAttr(
        default_factory=dict  # This should not be serialized
    )

    @field_validator("image_id", mode="before")
    @classmethod
    def validate_image_id(cls, value):
        if value is None:
            # An imageless session is legitimate: the socket is opened per user and only
            # later told which image to work on.
            return value
        with get_context_session() as session:
            # exists() is faster than fetching the whole object
            exists = session.query(Images.id).filter_by(id=value).scalar() is not None
            if not exists:
                raise ValueError(f"Image ID {value} does not exist.")
        return value

    @field_validator("mask_id", mode="before")
    @classmethod
    def validate_mask_id(cls, value, info: ValidationInfo):
        image_id = info.data.get("image_id")
        if value is None and image_id is None:
            return None

        with get_context_session() as session:
            # If mask_id is missing, try to find it via image_id
            if value is None and image_id:
                mask = session.query(Masks).filter_by(image_id=image_id).first()
                value = mask.id if mask else None

            if value is None:
                raise ValueError("Mask ID could not be determined.")

            # Check if the determined/provided ID actually exists
            exists = session.query(Masks.id).filter_by(id=value).scalar() is not None
            if not exists:
                raise ValueError(f"Mask ID {value} does not exist.")
        return value

    @cached_property
    def image_db(self) -> Images:
        with get_context_session() as session:
            image_db = session.query(Images).filter_by(id=self.image_id).one()
        return image_db

    @cached_property
    def mask_db(self) -> Masks:
        with get_context_session() as session:
            mask_db = session.query(Masks).filter_by(id=self.mask_id).one()
        return mask_db

    def switch_to_image(self, image_id: int, dataset_id: int | None = None) -> None:
        """ Point this session at another image, keeping everything that is not per-image.

            The AI backends stay registered and the models they hold stay loaded, which is
            the reason a switch is a message rather than a reconnect: re-running the health
            checks and the model selection for every image is what made stepping through a
            dataset slow.

            Everything that describes the *previous* image is dropped, including the two
            cached ORM rows -- ``functools.cached_property`` stores its value in the
            instance dict, so removing the key is what forces a re-read.

            :param image_id: The image to annotate from now on. Must exist.
            :param dataset_id: The dataset it belongs to, for the per-message permission
                checks. Resolved by the caller, which needs it for its own check anyway.
            :raises ValueError: If the image has no mask, or does not exist.
        """
        self.__dict__.pop("image_db", None)
        self.__dict__.pop("mask_db", None)

        # Round-trips through the validators so an unknown image or a missing mask is
        # rejected here rather than surfacing later as a confusing handler error.
        validated = AnnotationSessionState(
            image_id=image_id,
            mask_id=None,
            user_id=self.user_id,
            dataset_id=dataset_id,
        )

        self.image_id = validated.image_id
        self.mask_id = validated.mask_id
        self.dataset_id = dataset_id
        self.contour_hierarchy = None
        self.focussed_contour_id = None
        self.refinement_contour_id = None

    async def check_and_register_backend(self, service: BaseService, key):
        if not await service.check_backend():
            logger.error(f"{key} is not reachable. Please make sure it is running.")
            self._failed_backends[key] = service
        else:
            logger.debug(f"{key} is reachable.")
            self._running_backends[key] = service

    async def focus_contour(self, contour_id: int):
        self.focussed_contour_id = contour_id
        return True

    async def unfocus_contour(self):
        self.focussed_contour_id = None
        successful = []
        unsuccessful = []
        return self._running_backends.keys(), []
        for key, service in self._running_backends.items():
            try:
                response = await service.unfocus_crop(self.user_id)
                if not response["success"]:
                    logger.error(f"{key} ran into an error. Unfocussing might not have worked.")
                    unsuccessful.append(key)
                else:
                    successful.append(key)
            except Exception as e:
                unsuccessful.append(key)
        return successful, unsuccessful
