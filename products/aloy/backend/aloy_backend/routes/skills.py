"""Org-scoped skill registry endpoints: CRUD for ``SkillDefinition`` rows
and grant management (``SkillGrant``) controlling who may use each skill.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from pori import parse_skill_markdown

from ..database import get_session
from ..models import SkillDefinition, SkillGrant
from ..schemas import (
    SkillCreate,
    SkillGrantCreate,
    SkillGrantResponse,
    SkillImportPreview,
    SkillImportRequest,
    SkillResponse,
    SkillUpdate,
)
from ..tenancy import OrganizationContext, Permission, require_permission

router = APIRouter(prefix="/skills", tags=["skills"])


_MAX_SKILL_FETCH_BYTES = 262_144  # 256KB — a SKILL.md, not an archive
_SLUG_RE = re.compile(r"[^a-z0-9-]+")


def _slugify(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.strip().lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)[:64].strip("-")
    # SkillCreate's pattern needs 3+ chars starting/ending alphanumeric.
    return slug if len(slug) >= 3 else f"skill-{slug}".strip("-")


def _github_raw(url: str) -> str:
    """Rewrite a github.com blob URL to its raw content URL (the paste
    people actually have on their clipboard)."""
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/(.+)", url)
    if m:
        return (
            f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}"
        )
    return url


async def _fetch_skill_text(url: str) -> str:
    async with httpx.AsyncClient(
        timeout=15, follow_redirects=True, max_redirects=3
    ) as client:
        resp = await client.get(_github_raw(url))
    if resp.status_code != 200:
        raise HTTPException(
            status_code=422, detail=f"Could not fetch skill ({resp.status_code})"
        )
    if len(resp.content) > _MAX_SKILL_FETCH_BYTES:
        raise HTTPException(status_code=422, detail="Skill file too large (256KB max)")
    return resp.text


@router.post("/preview", response_model=SkillImportPreview)
async def preview_skill_import(
    body: SkillImportRequest,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_WRITE)),
) -> SkillImportPreview:
    """Parse a SKILL.md (by URL or pasted text) into prefilled create-form
    fields. Import-first UX: users paste, review, save — they never hand-type
    slugs or version strings. The format has ONE parser (the kernel's)."""
    if bool(body.url) == bool(body.text):
        raise HTTPException(status_code=422, detail="Provide exactly one of url|text")
    warnings: list[str] = []
    if body.url:
        text = await _fetch_skill_text(body.url.strip())
        fallback_name = (
            body.url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".md") or "skill"
        )
    else:
        text = body.text or ""
        fallback_name = "imported-skill"

    metadata, instructions = parse_skill_markdown(text)
    if not metadata:
        warnings.append(
            "No YAML frontmatter found — name and summary were inferred; review them."
        )
    if not instructions.strip():
        raise HTTPException(status_code=422, detail="The skill has no instructions")

    name = str(metadata.get("name") or fallback_name.replace("-", " ").title())
    summary = str(metadata.get("description") or metadata.get("summary") or "").strip()
    if not summary:
        summary = instructions.strip().splitlines()[0][:500]
        warnings.append("No description in metadata — first line used as summary.")
    raw_tags = metadata.get("tags") or []
    tags = (
        [t.strip() for t in raw_tags.split(",") if t.strip()]
        if isinstance(raw_tags, str)
        else [str(t) for t in raw_tags]
    )
    if len(instructions) > 50_000:
        instructions = instructions[:50_000]
        warnings.append("Instructions truncated to the 50k limit.")

    return SkillImportPreview(
        slug=_slugify(str(metadata.get("slug") or name)),
        version=str(metadata.get("version") or "1"),
        name=name[:120],
        summary=summary[:500],
        instructions=instructions,
        tags=tags,
        category=str(metadata.get("category") or "organization"),
        author=str(metadata.get("author") or ""),
        license=str(metadata.get("license") or ""),
        warnings=warnings,
    )


@router.post("", response_model=SkillResponse, status_code=201)
async def create_skill(
    body: SkillCreate,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_WRITE)),
    session: AsyncSession = Depends(get_session),
) -> SkillDefinition:
    skill = SkillDefinition(
        organization_id=context.organization_id,
        created_by=context.user_id,
        **body.model_dump(),
    )
    session.add(skill)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail="Skill version exists") from exc
    await session.refresh(skill)
    return skill


@router.get("", response_model=list[SkillResponse])
async def list_skills(
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
) -> Sequence[SkillDefinition]:
    result = await session.execute(
        select(SkillDefinition)
        .where(SkillDefinition.organization_id == context.organization_id)
        .order_by(SkillDefinition.slug, SkillDefinition.version)
    )
    return result.scalars().all()


@router.get("/{skill_id}", response_model=SkillResponse)
async def get_skill(
    skill_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
) -> SkillDefinition:
    skill = await session.get(SkillDefinition, skill_id)
    if not skill or skill.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Skill not found")
    return skill


@router.patch("/{skill_id}", response_model=SkillResponse)
async def update_skill(
    skill_id: str,
    body: SkillUpdate,
    context: OrganizationContext = Depends(
        require_permission(Permission.POLICY_MANAGE)
    ),
    session: AsyncSession = Depends(get_session),
) -> SkillDefinition:
    skill = await session.get(SkillDefinition, skill_id)
    if not skill or skill.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Skill not found")
    for key, value in body.model_dump(exclude_unset=True).items():
        setattr(skill, key, value)
    skill.updated_at = datetime.now(timezone.utc)
    session.add(skill)
    await session.commit()
    await session.refresh(skill)
    return skill


@router.post("/{skill_id}/grants", response_model=SkillGrantResponse, status_code=201)
async def create_skill_grant(
    skill_id: str,
    body: SkillGrantCreate,
    context: OrganizationContext = Depends(
        require_permission(Permission.POLICY_MANAGE)
    ),
    session: AsyncSession = Depends(get_session),
) -> SkillGrant:
    skill = await session.get(SkillDefinition, skill_id)
    if not skill or skill.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Skill not found")
    grant = SkillGrant(
        organization_id=context.organization_id,
        skill_id=skill_id,
        principal_type=body.principal_type,
        principal_id=body.principal_id,
        created_by=context.user_id,
    )
    session.add(grant)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail="Skill grant exists") from exc
    await session.refresh(grant)
    return grant


@router.get("/{skill_id}/grants", response_model=list[SkillGrantResponse])
async def list_skill_grants(
    skill_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
) -> Sequence[SkillGrant]:
    skill = await session.get(SkillDefinition, skill_id)
    if not skill or skill.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Skill not found")
    result = await session.execute(
        select(SkillGrant).where(
            SkillGrant.organization_id == context.organization_id,
            SkillGrant.skill_id == skill_id,
        )
    )
    return result.scalars().all()
