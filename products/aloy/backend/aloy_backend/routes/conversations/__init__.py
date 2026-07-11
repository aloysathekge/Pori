"""Conversations API package.

Each submodule owns a prefix-less ``APIRouter`` for one concern (crud,
search, artifacts, messaging, files, live-run control); this package
assembles them under the ``/conversations`` prefix and re-exports ``router``
so ``routes/__init__.py``'s ``from .conversations import router`` keeps
working unchanged.

Include order matters for overlapping paths: the literal ``/search`` and
``/context/search`` routes must be registered before ``/{conversation_id}``,
and ``/{conversation_id}/...`` POST routes before ``/clarify/{id}`` (live),
mirroring the original single-file registration order.
"""

from fastapi import APIRouter

from . import artifacts, crud, files, live, messaging, search

# The prefix rides on each include (not on the router itself): FastAPI
# rejects an empty include-prefix combined with the empty-path routes
# (POST/GET "") that crud.py defines.
_PREFIX = "/conversations"

router = APIRouter(tags=["conversations"])
router.include_router(search.router, prefix=_PREFIX)
router.include_router(crud.router, prefix=_PREFIX)
router.include_router(artifacts.router, prefix=_PREFIX)
router.include_router(messaging.router, prefix=_PREFIX)
router.include_router(files.router, prefix=_PREFIX)
router.include_router(live.router, prefix=_PREFIX)

__all__ = ["router"]
