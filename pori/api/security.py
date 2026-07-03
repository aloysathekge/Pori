import logging
import os
from typing import List, Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_503_SERVICE_UNAVAILABLE

logger = logging.getLogger("pori.api.security")

# auto_error=False so get_api_key handles a missing header uniformly (a clean
# 401) instead of Starlette raising a 403 before our logic runs.
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_keys() -> List[str]:
    """Valid API keys from PORI_API_KEY (comma-separated), read once at import."""
    api_keys_str = os.getenv("PORI_API_KEY", "")
    if not api_keys_str:
        return []
    return [key.strip() for key in api_keys_str.split(",") if key.strip()]


VALID_API_KEYS = get_api_keys()


def _unauthenticated_allowed() -> bool:
    """Explicit opt-in to run without auth (local dev only)."""
    return os.getenv("PORI_API_ALLOW_UNAUTHENTICATED", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def get_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> str:
    """Verify the X-API-Key header.

    Fails **closed**: when no keys are configured the API denies every request
    (503) instead of the old fail-open "allow all" — an unconfigured deployment is
    an unprotected one. Set ``PORI_API_ALLOW_UNAUTHENTICATED`` to explicitly opt in
    to unauthenticated access for local dev.
    """
    if not VALID_API_KEYS:
        if _unauthenticated_allowed():
            return api_key or ""
        logger.error(
            "Refusing request: no API keys configured. Set PORI_API_KEY, or "
            "PORI_API_ALLOW_UNAUTHENTICATED=1 to run unauthenticated (dev only)."
        )
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "API authentication is not configured. Set PORI_API_KEY (or "
                "PORI_API_ALLOW_UNAUTHENTICATED=1 for unauthenticated dev)."
            ),
        )
    if api_key and api_key in VALID_API_KEYS:
        return api_key
    raise HTTPException(
        status_code=HTTP_401_UNAUTHORIZED, detail="Invalid or missing API Key"
    )
