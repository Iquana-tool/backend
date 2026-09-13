"""Image calibration: typed, per-image transforms that make measurements comparable.

Three layers, imported in this order:

``cards``
    Reference card profiles — how many patches a physical target has and what each
    should read. Data, so a new card is a profile, not a code change.
``strategies``
    Ways of *estimating* a calibration. A two-patch sample and a twenty-patch
    reference card are two estimators of one transform, not two corrections.
``registry``
    What each calibration kind *is* — its parameters, validation, pixel transform,
    pipeline position, which strategies estimate it, and which stored metrics it
    invalidates. Adding a kind means adding an entry here.
``store``
    Row-level persistence for ``image_calibrations``, plus the staleness helper.
    Knows nothing about kinds.
``service``
    The kind-agnostic orchestration the routes and the quantification batch call.

The public surface is re-exported here so callers write
``from app.services.calibration import set_calibration`` without caring about the
split.
"""
from app.services.calibration.cards import (  # noqa: F401
    DEFAULT_CARD,
    ReferenceCard,
    all_cards,
    get_card,
)
from app.services.calibration.strategies import (  # noqa: F401
    DEFAULT_STRATEGY,
    CalibrationStrategy,
    all_strategies,
    get_strategy,
)
from app.services.calibration.registry import (  # noqa: F401
    APPEARANCE_STALE_METRIC_KEYS,
    LEGACY_KINDS,
    SCALE_STALE_METRIC_KEYS,
    CalibrationKind,
    CalibrationSource,
    all_kinds,
    get_kind,
    pixel_stages,
)
from app.services.calibration.service import (  # noqa: F401
    apply_calibration_pipeline,
    apply_to_dataset,
    calibrated_counts,
    clear_calibration,
    get_calibration_state,
    get_dataset_defaults,
    kinds_metadata,
    load_calibrated_image_rgb,
    sample_patch,
    sample_patches,
    set_calibration,
    set_dataset_defaults,
    validate_defaults,
)
