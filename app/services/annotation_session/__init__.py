"""Reusable, transport-agnostic building blocks for image annotation sessions.

This package holds the session state and the AI segmentation operations so that the
core logic can be called outside of a WebSocket (HTTP routes, Celery tasks, tests).
The WebSocket adapter lives in ``app.routes.websockets``.
"""

from app.services.annotation_session.state import AnnotationSessionState, Backends

__all__ = ["AnnotationSessionState", "Backends"]
