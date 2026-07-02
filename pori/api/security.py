import os
from typing import List

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")


def get_api_keys() -> List[str]:
    """Retrieves the list of valid API keys from the environment."""
    api_keys_str = os.getenv("PORI_API_KEY", "")
    if not api_keys_str:
        return []
    return [key.strip() for key in api_keys_str.split(",")]


VALID_API_KEYS = get_api_keys()


def get_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    Dependency that verifies the API key in the X-API-Key header.
    """
    if not VALID_API_KEYS:
        # If no keys are configured, allow access but log a warning.
        # In a production environment, you might want to deny access by default.
        print("Warning: No API keys configured. Allowing all requests.")
        return api_key

    if api_key in VALID_API_KEYS:
        return api_key
    else:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED, detail="Invalid or missing API Key"
        )
