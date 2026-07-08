"""Account connection endpoints — the native OAuth web flow."""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ..config import settings
from ..connections import available_providers, get_provider
from ..connections.store import (
    exchange_code,
    fetch_account_email,
    revoke_connection,
    upsert_connection,
)
from ..database import get_session
from ..models import OAuthConnection, OAuthFlowState
from ..schemas import ConnectionResponse, ConnectionStartResponse, ProviderInfo
from ..tenancy import OrganizationContext, Permission, require_permission

logger = logging.getLogger("aloy_backend")

router = APIRouter(prefix="/connections", tags=["connections"])


def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .decode()
        .rstrip("=")
    )
    return verifier, challenge


def _callback_uri(provider: str) -> str:
    return f"{settings.backend_base_url}/v1/connections/{provider}/callback"


@router.get("", response_model=list[ProviderInfo])
async def list_providers(
    context: OrganizationContext = Depends(require_permission(Permission.RUN_READ)),
    session: AsyncSession = Depends(get_session),
):
    """Available providers + this user's connection status for each."""
    conns = (
        (
            await session.execute(
                select(OAuthConnection).where(
                    OAuthConnection.organization_id == context.organization_id,
                    OAuthConnection.user_id == context.user_id,
                )
            )
        )
        .scalars()
        .all()
    )
    by_provider = {c.provider: c for c in conns}
    out: list[ProviderInfo] = []
    for spec in available_providers():
        conn = by_provider.get(spec.name)
        out.append(
            ProviderInfo(
                provider=spec.name,
                label=spec.label or spec.name,
                description=spec.description,
                connected=conn is not None and conn.status != "revoked",
                status=conn.status if conn else None,
                account_email=conn.account_email if conn else None,
            )
        )
    return out


@router.post("/{provider}/start", response_model=ConnectionStartResponse)
async def start_connection(
    provider: str,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
):
    spec = get_provider(provider)
    if spec is None or not spec.is_configured():
        raise HTTPException(status_code=404, detail="Provider not available")

    state = secrets.token_urlsafe(32)
    verifier, challenge = _pkce_pair()
    session.add(
        OAuthFlowState(
            state=state,
            organization_id=context.organization_id,
            user_id=context.user_id,
            provider=spec.name,
            pkce_verifier=verifier,
            expires_at=datetime.now(timezone.utc)
            + timedelta(seconds=settings.connection_flow_ttl_seconds),
        )
    )
    await session.commit()

    params = {
        "client_id": spec.client_id,
        "redirect_uri": _callback_uri(spec.name),
        "response_type": "code",
        "scope": " ".join(spec.scopes),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        **spec.extra_authorize_params,
    }
    return ConnectionStartResponse(
        authorize_url=f"{spec.authorize_url}?{urlencode(params)}"
    )


@router.get("/{provider}/callback")
async def oauth_callback(
    provider: str,
    state: str = Query(...),
    code: str | None = Query(default=None),
    error: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
):
    """Provider redirects here (no auth header — identity is carried by `state`)."""
    dest = f"{settings.app_base_url}/connections"
    flow = await session.get(OAuthFlowState, state)
    if flow is None or flow.expires_at.replace(
        tzinfo=flow.expires_at.tzinfo or timezone.utc
    ) < datetime.now(timezone.utc):
        return RedirectResponse(f"{dest}?connected=error&reason=state")
    # One-time use.
    await session.delete(flow)
    await session.commit()

    if error or not code:
        return RedirectResponse(f"{dest}?connected=error&reason={error or 'no_code'}")

    spec = get_provider(provider)
    if spec is None or flow.provider != provider:
        return RedirectResponse(f"{dest}?connected=error&reason=provider")

    try:
        payload = await exchange_code(
            spec, code, _callback_uri(spec.name), flow.pkce_verifier
        )
        email = await fetch_account_email(spec, payload["access_token"])
        await upsert_connection(
            session,
            organization_id=flow.organization_id,
            user_id=flow.user_id,
            spec=spec,
            token_payload=payload,
            account_email=email,
        )
    except Exception:
        logger.exception("OAuth callback failed for provider %s", provider)
        return RedirectResponse(f"{dest}?connected=error&reason=exchange")

    return RedirectResponse(f"{dest}?connected={provider}")


@router.delete("/{provider}", response_model=ConnectionResponse)
async def disconnect(
    provider: str,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
):
    spec = get_provider(provider)
    if spec is None:
        raise HTTPException(status_code=404, detail="Unknown provider")
    removed = await revoke_connection(
        session, context.organization_id, context.user_id, spec
    )
    if not removed:
        raise HTTPException(status_code=404, detail="Not connected")
    return ConnectionResponse(provider=provider, connected=False)
