"""Token custody: exchange, upsert, refresh-on-use, revoke.

The single choke point every tool goes through to get a valid access token.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from sqlmodel import select

from ..models import OAuthConnection
from .crypto import decrypt, encrypt
from .providers import ProviderSpec

logger = logging.getLogger("aloy_backend")

# Refresh a token this many seconds before it actually expires.
_REFRESH_SKEW = 120


def _aware(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


async def exchange_code(
    spec: ProviderSpec, code: str, redirect_uri: str, pkce_verifier: str
) -> dict:
    """Authorization code -> tokens."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            spec.token_url,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": spec.client_id,
                "client_secret": spec.client_secret,
                "code_verifier": pkce_verifier,
            },
        )
        resp.raise_for_status()
        return resp.json()


async def fetch_account_email(spec: ProviderSpec, access_token: str) -> Optional[str]:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                spec.userinfo_url,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            resp.raise_for_status()
            return resp.json().get("email")
    except Exception:
        return None


async def upsert_connection(
    session,
    *,
    organization_id: str,
    user_id: str,
    spec: ProviderSpec,
    token_payload: dict,
    account_email: Optional[str],
) -> OAuthConnection:
    existing = (
        (
            await session.execute(
                select(OAuthConnection).where(
                    OAuthConnection.organization_id == organization_id,
                    OAuthConnection.user_id == user_id,
                    OAuthConnection.provider == spec.name,
                )
            )
        )
        .scalars()
        .first()
    )

    expires_in = token_payload.get("expires_in")
    expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
        if expires_in
        else None
    )
    access_enc = encrypt(token_payload["access_token"])
    # Google only returns refresh_token on first consent; keep the old one on
    # re-consent if a new one isn't provided.
    new_refresh = token_payload.get("refresh_token")
    refresh_enc = encrypt(new_refresh) if new_refresh else None
    scopes = (token_payload.get("scope") or "").split() or spec.scopes

    if existing is None:
        conn = OAuthConnection(
            organization_id=organization_id,
            user_id=user_id,
            provider=spec.name,
            access_token_enc=access_enc,
            refresh_token_enc=refresh_enc,
            scopes=scopes,
            account_email=account_email,
            expires_at=expires_at,
            status="active",
        )
        session.add(conn)
    else:
        existing.access_token_enc = access_enc
        if refresh_enc:
            existing.refresh_token_enc = refresh_enc
        existing.scopes = scopes
        existing.account_email = account_email or existing.account_email
        existing.expires_at = expires_at
        existing.status = "active"
        existing.updated_at = datetime.now(timezone.utc)
        session.add(existing)
        conn = existing
    await session.commit()
    await session.refresh(conn)
    return conn


async def _refresh(session, conn: OAuthConnection, spec: ProviderSpec) -> Optional[str]:
    if not conn.refresh_token_enc:
        return None
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                spec.token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": decrypt(conn.refresh_token_enc),
                    "client_id": spec.client_id,
                    "client_secret": spec.client_secret,
                },
            )
            resp.raise_for_status()
            payload = resp.json()
    except Exception:
        logger.warning("Token refresh failed for connection %s", conn.id, exc_info=True)
        conn.status = "error"
        conn.updated_at = datetime.now(timezone.utc)
        session.add(conn)
        await session.commit()
        return None

    access = payload["access_token"]
    conn.access_token_enc = encrypt(access)
    if payload.get("refresh_token"):
        conn.refresh_token_enc = encrypt(payload["refresh_token"])
    if payload.get("expires_in"):
        conn.expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=int(payload["expires_in"])
        )
    conn.status = "active"
    conn.updated_at = datetime.now(timezone.utc)
    session.add(conn)
    await session.commit()
    return access


async def get_access_token(
    session, organization_id: str, user_id: str, spec: ProviderSpec
) -> Optional[str]:
    """A valid access token for the user's connection, refreshing if near expiry.

    Returns None when the user has no active connection or a refresh failed.
    """
    conn = (
        (
            await session.execute(
                select(OAuthConnection).where(
                    OAuthConnection.organization_id == organization_id,
                    OAuthConnection.user_id == user_id,
                    OAuthConnection.provider == spec.name,
                    OAuthConnection.status == "active",
                )
            )
        )
        .scalars()
        .first()
    )
    if conn is None:
        return None
    expires_at = _aware(conn.expires_at)
    if expires_at is not None and expires_at <= datetime.now(timezone.utc) + timedelta(
        seconds=_REFRESH_SKEW
    ):
        return await _refresh(session, conn, spec)
    return decrypt(conn.access_token_enc)


async def resolve_run_connections(session, organization_id: str, user_id: str) -> dict:
    """{provider: {access_token, account_email}} for the user's ACTIVE
    connections, tokens freshly refreshed. Injected into the run's tool context
    so sync tools get a valid token without doing async DB work themselves."""
    from .providers import PROVIDERS

    out: dict = {}
    active = (
        (
            await session.execute(
                select(OAuthConnection).where(
                    OAuthConnection.organization_id == organization_id,
                    OAuthConnection.user_id == user_id,
                    OAuthConnection.status == "active",
                )
            )
        )
        .scalars()
        .all()
    )
    for conn in active:
        spec = PROVIDERS.get(conn.provider)
        if spec is None:
            continue
        token = await get_access_token(session, organization_id, user_id, spec)
        if token:
            out[conn.provider] = {
                "access_token": token,
                "account_email": conn.account_email,
            }
    return out


async def revoke_connection(
    session, organization_id: str, user_id: str, spec: ProviderSpec
) -> bool:
    conn = (
        (
            await session.execute(
                select(OAuthConnection).where(
                    OAuthConnection.organization_id == organization_id,
                    OAuthConnection.user_id == user_id,
                    OAuthConnection.provider == spec.name,
                )
            )
        )
        .scalars()
        .first()
    )
    if conn is None:
        return False
    # Best-effort remote revoke, then delete locally.
    try:
        token = decrypt(conn.access_token_enc)
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(spec.revoke_url, data={"token": token})
    except Exception:
        logger.info("Remote revoke failed for %s (deleting locally anyway)", conn.id)
    await session.delete(conn)
    await session.commit()
    return True
