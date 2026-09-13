"""In-process cache for a mask's contour hierarchy.

Rebuilding a hierarchy is the expensive half of switching images: every contour row is
turned into a ``Contour`` model, its SVG path is recomputed from the normalized
coordinates, and the result is serialized. None of that changes between two reads of the
same unmodified mask, so stepping back and forth through a filmstrip pays the same cost
over and over.

The cache is keyed by mask id and holds the hierarchy model, its client payload (the two
are always wanted together) and a fingerprint of the mask's contour rows. Freshness is
guarded from two directions, because neither alone is enough:

* Every write through the database-access layer calls :func:`invalidate`. This catches
  in-place edits -- a label assignment, an approval -- that leave the row set the same
  size. It only reaches the process that performed the write.
* Every read passes a fingerprint of the stored rows (row count and highest id, computed
  by ``masks.contour_fingerprint``) and a mismatch is treated as a miss. This catches
  inserts and deletes made by *another* process, which is how batch inference writes
  contours: its Celery worker has its own cache and cannot invalidate this one.

What is left uncovered is an in-place edit made in a different process than the read --
two API workers with two users editing the same mask's labels at the same time. That is
already outside what an annotation session supports (there is no cross-session broadcast
either), so it is accepted rather than paid for with a schema change.

The module lives directly under ``app.services`` and imports nothing from ``app`` so that
the database-access layer can invalidate from inside it without an import cycle.
"""

from collections import OrderedDict
from logging import getLogger
from threading import Lock
from typing import Any

from iquana_toolbox.schemas.database.contour_hierarchy import ContourHierarchy

logger = getLogger(__name__)

#: How many masks to keep. A user navigating a dataset touches a handful of images at a
#: time, and each entry holds a full contour set, so the cache stays small on purpose.
MAX_ENTRIES = 32

# The entries are handed out to request handlers that may run concurrently, so the
# bookkeeping around the mapping is locked. The entries themselves are treated as
# read-only by callers (see get_cached_contour_hierarchy_of_mask).
_lock = Lock()
_entries: "OrderedDict[int, tuple[Any, ContourHierarchy, dict]]" = OrderedDict()


def get(mask_id: int, fingerprint: Any) -> tuple[ContourHierarchy, dict] | None:
    """Return a mask's cached ``(hierarchy, client_payload)``, or None.

    A miss and a fingerprint mismatch are reported the same way: the caller rebuilds.
    """
    with _lock:
        entry = _entries.get(mask_id)
        if entry is None:
            return None
        cached_fingerprint, hierarchy, payload = entry
        if cached_fingerprint != fingerprint:
            # Someone else's process added or removed contours since this was cached.
            del _entries[mask_id]
            return None
        _entries.move_to_end(mask_id)
        return hierarchy, payload


def put(mask_id: int, fingerprint: Any, hierarchy: ContourHierarchy) -> dict:
    """Cache a mask's hierarchy and return the client payload computed alongside it."""
    payload = hierarchy.dump_for_client()
    with _lock:
        _entries[mask_id] = (fingerprint, hierarchy, payload)
        _entries.move_to_end(mask_id)
        while len(_entries) > MAX_ENTRIES:
            _entries.popitem(last=False)
    return payload


def invalidate(mask_id: int | None) -> None:
    """Drop a mask's cached hierarchy. Safe to call for a mask that was never cached."""
    if mask_id is None:
        return
    with _lock:
        if _entries.pop(mask_id, None) is not None:
            logger.debug("Invalidated cached contour hierarchy for mask %s.", mask_id)


def clear() -> None:
    """Drop every entry. Used by tests and by bulk operations that touch many masks."""
    with _lock:
        _entries.clear()
