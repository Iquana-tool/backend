"""Pushing credential settings across to the AI service.

The Hugging Face token is edited here -- this is where the admin surface and the
database are -- but consumed over there, by the process that downloads model
weights. This module is the bridge.

Deliberately push rather than pull: the AI service has no database and no account
system, so having it fetch its own configuration would mean giving it a
credential to authenticate with, which is the problem it is trying to solve.

The push is best-effort by design. A saved setting is durable in the database
whether or not the AI service happened to be running, and the admin page reports
the far side's live state so an operator can see the drift and re-push, rather
than the save failing because an unrelated service was restarting.
"""
from __future__ import annotations

import os
from logging import getLogger

import httpx

from config import AI_SERVICE_URL

logger = getLogger(__name__)

#: Shared secret for the AI service's config endpoint. Optional: the service
#: refuses the write when *it* has one configured and the header does not match,
#: and accepts it otherwise, which keeps a single-machine install working with no
#: setup while letting a networked deployment lock the endpoint down.
_ADMIN_TOKEN_ENV = "AI_SERVICE_ADMIN_TOKEN"

_TIMEOUT = 10


def _headers() -> dict[str, str]:
    token = (os.getenv(_ADMIN_TOKEN_ENV) or "").strip()
    return {"X-Admin-Token": token} if token else {}


async def read_config() -> dict:
    """Report what the AI service currently holds, for the admin page.

    Returns ``{"reachable": False}`` rather than raising when the service is
    down: "cannot tell" is a legitimate answer for a status panel, and a stopped
    AI service must not make the settings page fail to load.
    """
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.get(f"{AI_SERVICE_URL}/config", headers=_headers())
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:  # noqa: BLE001 - any transport or status error means "unknown"
        logger.info("AI service configuration unavailable: %s", exc)
        return {"reachable": False}
    return {"reachable": True, **payload}


async def push_config(values: dict[str, str | None]) -> dict:
    """Send credential values to the AI service.

    ``values`` is keyed by the environment variable the far side should set, e.g.
    ``{"HF_ACCESS_TOKEN": "hf_..."}``.

    Returns ``{"pushed": bool, "error": str | None}``; never raises, so a save
    completes even when the AI service is unreachable.
    """
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.patch(
                f"{AI_SERVICE_URL}/config",
                json={"values": values},
                headers=_headers(),
            )
            response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - reported to the operator, not raised
        logger.warning("Could not push configuration to the AI service: %s", exc)
        return {"pushed": False, "error": str(exc)}

    logger.info("Pushed %s to the AI service.", ", ".join(sorted(values)))
    return {"pushed": True, "error": None}
