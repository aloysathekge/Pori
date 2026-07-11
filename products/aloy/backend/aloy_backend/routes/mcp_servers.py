"""MCP server management — personal + org-shared (scope/union like connections)."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from ..connections.crypto import encrypt
from ..database import get_session
from ..models import ORG_CONNECTION_USER, McpServer
from ..schemas import McpServerCreate, McpServerInfo, McpServerUpdate
from ..tenancy import OrganizationContext, Permission, require_permission

router = APIRouter(prefix="/mcp-servers", tags=["mcp"])


def _info(server: McpServer, can_manage_org: bool) -> McpServerInfo:
    managed = server.scope == "user" or can_manage_org
    return McpServerInfo(
        id=server.id,
        name=server.name,
        url=server.url,
        transport=server.transport,
        auth_kind=server.auth_kind,
        scope=server.scope,
        enabled=server.enabled,
        account_managed=managed,
    )


@router.get("", response_model=list[McpServerInfo])
async def list_servers(
    context: OrganizationContext = Depends(require_permission(Permission.RUN_READ)),
    session: AsyncSession = Depends(get_session),
) -> list[McpServerInfo]:
    rows = (
        (
            await session.execute(
                select(McpServer).where(
                    McpServer.organization_id == context.organization_id,
                    or_(
                        col(McpServer.user_id) == context.user_id,
                        col(McpServer.scope) == "org",
                    ),
                )
            )
        )
        .scalars()
        .all()
    )
    can_manage = context.permits(Permission.CONNECTION_MANAGE)
    return [_info(s, can_manage) for s in rows]


@router.post("", response_model=McpServerInfo, status_code=201)
async def create_server(
    req: McpServerCreate,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
) -> McpServerInfo:
    if req.scope not in ("user", "org"):
        raise HTTPException(status_code=422, detail="scope must be user|org")
    if req.transport not in ("http", "sse"):
        raise HTTPException(status_code=422, detail="transport must be http|sse")
    if req.auth_kind not in ("none", "static"):
        raise HTTPException(status_code=422, detail="auth_kind must be none|static")
    if req.scope == "org" and not context.permits(Permission.CONNECTION_MANAGE):
        raise HTTPException(
            status_code=403, detail="Managing organization MCP servers requires admin"
        )
    if req.auth_kind == "static" and not req.static_secret:
        raise HTTPException(status_code=422, detail="static auth needs a secret")

    eff_user = ORG_CONNECTION_USER if req.scope == "org" else context.user_id
    # Name must be unique within the caller's scope.
    dup = (
        (
            await session.execute(
                select(McpServer).where(
                    McpServer.organization_id == context.organization_id,
                    McpServer.user_id == eff_user,
                    McpServer.name == req.name,
                )
            )
        )
        .scalars()
        .first()
    )
    if dup is not None:
        raise HTTPException(status_code=409, detail="A server with that name exists")

    static_secret_enc = None
    if req.auth_kind == "static":
        # Guarded above: static auth always carries a secret.
        assert req.static_secret is not None
        static_secret_enc = encrypt(req.static_secret)

    server = McpServer(
        organization_id=context.organization_id,
        user_id=eff_user,
        scope=req.scope,
        created_by=context.user_id,
        name=req.name,
        transport=req.transport,
        url=req.url,
        auth_kind=req.auth_kind,
        static_secret_enc=static_secret_enc,
        tools_include=req.tools_include,
        tools_exclude=req.tools_exclude,
    )
    session.add(server)
    await session.commit()
    await session.refresh(server)
    return _info(server, context.permits(Permission.CONNECTION_MANAGE))


async def _load_manageable(
    session: AsyncSession, context: OrganizationContext, server_id: str
) -> McpServer:
    server = await session.get(McpServer, server_id)
    if server is None or server.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Server not found")
    if server.scope == "org":
        if not context.permits(Permission.CONNECTION_MANAGE):
            raise HTTPException(status_code=403, detail="Requires admin")
    elif server.user_id != context.user_id:
        raise HTTPException(status_code=404, detail="Server not found")
    return server


@router.patch("/{server_id}", response_model=McpServerInfo)
async def update_server(
    server_id: str,
    req: McpServerUpdate,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
) -> McpServerInfo:
    server = await _load_manageable(session, context, server_id)
    server.enabled = req.enabled
    server.updated_at = datetime.now(timezone.utc)
    session.add(server)
    await session.commit()
    await session.refresh(server)
    return _info(server, context.permits(Permission.CONNECTION_MANAGE))


@router.delete("/{server_id}", status_code=204)
async def delete_server(
    server_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
) -> None:
    server = await _load_manageable(session, context, server_id)
    await session.delete(server)
    await session.commit()
