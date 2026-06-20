"""Transport-agnostic AI segmentation operations.

Each ``run_*`` function takes plain primitives plus an AI service instance, performs the
inference, parses the result into domain objects, and returns it alongside the service's
``success`` / ``message``. They never touch a WebSocket, a ``ClientMessage`` or the
database, so they are reusable from HTTP routes, Celery tasks or tests.

Persistence (adding/replacing contours on a mask) and session bookkeeping are the
caller's responsibility -- see ``app.routes.websockets.annotation_handlers`` for the
WebSocket adapter.
"""

from dataclasses import dataclass
from logging import getLogger

from iquana_toolbox.schemas.database.contour_hierarchy import ContourHierarchy
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.prompts import Prompts
from iquana_toolbox.schemas.networking.http.services import (
    CompletionRequest as CompletionServiceRequest,
    PromptedSegmentationRequest,
    SemanticSegmentationRequest,
)

from app.services.ai_services.base_service import BaseService

logger = getLogger(__name__)


@dataclass
class PromptedSegmentationResult:
    contour: Contour
    success: bool
    message: str


@dataclass
class SemanticSegmentationResult:
    hierarchy: ContourHierarchy
    success: bool
    message: str


@dataclass
class CompletionResult:
    contours: list[Contour]
    success: bool
    message: str


async def run_prompted_segmentation(
        *,
        service: BaseService,
        image_url: str,
        image_width: int,
        image_height: int,
        model_key: str,
        prompts: Prompts,
        user_id: str,
        previous_mask=None,
        parent_id: int | None = None,
) -> PromptedSegmentationResult:
    """Run prompted segmentation and build the resulting contour.

    Computes the SVG path, but leaves the add-vs-replace / refinement decision and any
    persistence to the caller.
    """
    request = PromptedSegmentationRequest(
        user_id=str(user_id),
        image_url=image_url,
        model_registry_key=model_key,
        previous_mask=previous_mask,
        prompts=prompts,
    )
    response = await service.inference(request)
    contour = Contour.model_validate(response["result"])
    contour.parent_id = parent_id
    contour.compute_path(image_width=image_width, image_height=image_height)
    return PromptedSegmentationResult(
        contour=contour,
        success=response["success"],
        message=response["message"],
    )


async def run_semantic_segmentation(
        *,
        service: BaseService,
        image_url: str,
        model_registry_key: str,
        user_id: str,
) -> SemanticSegmentationResult:
    """Run semantic segmentation and parse the resulting contour hierarchy."""
    request = SemanticSegmentationRequest(
        model_registry_key=model_registry_key,
        image_url=image_url,
        user_id=user_id,
    )
    response = await service.inference(request)
    hierarchy = ContourHierarchy.model_validate(response["result"])
    return SemanticSegmentationResult(
        hierarchy=hierarchy,
        success=response["success"],
        message=response["message"],
    )


async def run_completion_segmentation(
        *,
        service: BaseService,
        image_url: str,
        model_key: str,
        user_id: str,
        positive_exemplars: list,
        concept=None,
        negative_exemplars: list | None = None,
) -> CompletionResult:
    """Run annotation completion and parse the resulting contours.

    ``positive_exemplars`` (binary mask models) and ``concept`` are resolved by the
    caller, since deriving them from seed contours requires database access.
    """
    request = CompletionServiceRequest(
        image_url=image_url,
        model_registry_key=model_key,
        user_id=user_id,
        positive_exemplars=positive_exemplars,
        negative_exemplars=negative_exemplars,
        concept=concept,
    )
    response = await service.inference(request)
    contours = [Contour.model_validate(contour_json) for contour_json in (response["result"] or [])]
    return CompletionResult(
        contours=contours,
        success=response["success"],
        message=response["message"],
    )
