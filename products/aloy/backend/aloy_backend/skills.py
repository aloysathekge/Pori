"""Organization skill catalog loading for Cloud runs."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from pori import SkillCatalog, SkillManifest

from .models import SkillDefinition, SkillGrant


async def load_skill_catalog(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    role: str,
) -> SkillCatalog:
    catalog = SkillCatalog()
    result = await session.execute(
        select(SkillDefinition).where(
            SkillDefinition.organization_id == organization_id,
            SkillDefinition.status == "approved",
        )
    )
    skills = result.scalars().all()
    if not skills:
        return catalog
    grants = await session.execute(
        select(SkillGrant).where(
            SkillGrant.organization_id == organization_id,
            col(SkillGrant.skill_id).in_([skill.id for skill in skills]),
        )
    )
    allowed_ids = {
        grant.skill_id
        for grant in grants.scalars().all()
        if (grant.principal_type == "user" and grant.principal_id == user_id)
        or (grant.principal_type == "role" and grant.principal_id == role)
        or (grant.principal_type == "role" and grant.principal_id == "*")
    }
    for skill in skills:
        if skill.id not in allowed_ids:
            continue
        catalog.register(
            SkillManifest(
                slug=skill.slug,
                name=skill.name,
                version=skill.version,
                summary=skill.summary,
                tags=tuple(skill.tags or []),
                category=skill.category,
                author=skill.author,
                license=skill.license,
                commands=tuple(skill.commands or []),
                argument_hint=skill.argument_hint,
                provenance=skill.provenance,
                trust_level=skill.trust_level,
                required_commands=tuple(skill.required_commands or []),
                setup_help=skill.setup_help,
                required_tools=frozenset(skill.required_tools or []),
                required_credentials=tuple(skill.required_credentials or []),
                required_platforms=tuple(skill.required_platforms or []),
                required_model_capabilities=frozenset(
                    skill.required_model_capabilities or []
                ),
                source=f"aloy-backend:{skill.id}",
                source_url=skill.source_url,
                install_command=skill.install_command,
                readiness_warnings=tuple(skill.readiness_warnings or []),
                sensitivity=skill.sensitivity,
            ),
            skill.instructions,
        )
    return catalog


__all__ = ["load_skill_catalog"]
