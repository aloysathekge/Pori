"""Gateway pairing + link management routes."""

from __future__ import annotations

import logging
import secrets
import string
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ..config import settings
from ..database import get_session
from ..models import GatewayLink, GatewayPairingCode
from ..schemas import GatewayLinkResponse, GatewayPairResponse
from ..tenancy import OrganizationContext, Permission, require_permission

logger = logging.getLogger("aloy_backend")

router = APIRouter(prefix="/gateway", tags=["gateway"])

_CODE_ALPHABET = string.ascii_uppercase + string.digits


def _mint_code() -> str:
    return "".join(secrets.choice(_CODE_ALPHABET) for _ in range(8))


@router.post("/pair", response_model=GatewayPairResponse, status_code=201)
async def create_pairing_code(
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
):
    """Mint a one-time code the user sends to the bot to pair a chat."""
    code = GatewayPairingCode(
        code=_mint_code(),
        organization_id=context.organization_id,
        user_id=context.user_id,
        platform="telegram",
        expires_at=datetime.now(timezone.utc)
        + timedelta(seconds=settings.gateway_pairing_ttl_seconds),
    )
    session.add(code)
    await session.commit()
    await session.refresh(code)
    return GatewayPairResponse(
        code=code.code, platform=code.platform, expires_at=code.expires_at
    )


@router.get("/links", response_model=list[GatewayLinkResponse])
async def list_links(
    context: OrganizationContext = Depends(require_permission(Permission.RUN_READ)),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(GatewayLink).where(
            GatewayLink.organization_id == context.organization_id,
            GatewayLink.user_id == context.user_id,
        )
    )
    return result.scalars().all()


@router.delete("/links/{link_id}", status_code=204)
async def delete_link(
    link_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
):
    link = await session.get(GatewayLink, link_id)
    if (
        not link
        or link.organization_id != context.organization_id
        or link.user_id != context.user_id
    ):
        raise HTTPException(status_code=404, detail="Link not found")
    await session.delete(link)
    await session.commit()
    return None
