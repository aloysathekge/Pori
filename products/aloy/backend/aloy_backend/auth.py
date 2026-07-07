from __future__ import annotations

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwk, jwt

from .config import settings

_bearer = HTTPBearer()

# Cache the JWKS keys in memory (refreshed on restart)
_jwks_cache: dict | None = None


async def _get_jwks() -> dict:
    """Fetch and cache Supabase's JWKS public keys."""
    global _jwks_cache
    if _jwks_cache is None:
        async with httpx.AsyncClient() as client:
            resp = await client.get(settings.supabase_jwks_url)
            resp.raise_for_status()
            _jwks_cache = resp.json()
    return _jwks_cache


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    """Verify Supabase JWT and return the user's ID (sub claim)."""
    token = credentials.credentials

    try:
        # Get the key ID from the token header
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")

        # Find the matching key in JWKS
        jwks_data = await _get_jwks()
        signing_key = None
        for key in jwks_data.get("keys", []):
            if key.get("kid") == kid:
                signing_key = key
                break

        if not signing_key:
            # Key not found — maybe rotated. Clear cache and retry once.
            global _jwks_cache
            _jwks_cache = None
            jwks_data = await _get_jwks()
            for key in jwks_data.get("keys", []):
                if key.get("kid") == kid:
                    signing_key = key
                    break

        if not signing_key:
            raise JWTError("No matching signing key found")

        # Build the public key and verify
        public_key = jwk.construct(signing_key)
        payload = jwt.decode(
            token,
            public_key,
            algorithms=[signing_key.get("alg", "ES256")],
            audience="authenticated",
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    user_id: str | None = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing user ID",
        )
    return user_id
