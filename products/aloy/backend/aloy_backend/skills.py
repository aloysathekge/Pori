"""Organization skill catalog loading for Cloud runs."""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from pori import SkillCatalog, SkillManifest, load_skill_catalog_from_directories

from .models import SkillDefinition, SkillGrant

logger = logging.getLogger("aloy_backend.skills")

_PRODUCT_SKILLS_DIR = Path(__file__).resolve().parent / "product_skills"
SURFACE_BUILDER_SKILL_ID = "surface-builder@1"


def _load_bundled_skill_catalog() -> SkillCatalog:
    """Load Aloy-owned skills before tenant-installed skills."""
    discovered = load_skill_catalog_from_directories([_PRODUCT_SKILLS_DIR])
    catalog = SkillCatalog()
    for manifest in discovered.manifests():
        loaded = discovered.load(manifest.skill_id)
        bundled_manifest = manifest.model_copy(
            update={
                "category": "aloy-product",
                "provenance": "aloy-bundled",
                "trust_level": "product",
                "source": f"aloy-bundled:{manifest.slug}",
                "required_model_capabilities": frozenset({"tools"}),
                "model_invocation_disabled": True,
            }
        )
        catalog.register(
            bundled_manifest,
            loaded.instructions,
            root_path=_PRODUCT_SKILLS_DIR / manifest.slug,
        )
    return catalog


async def load_skill_catalog(
    session: AsyncSession,
    *,
    organization_id: str,
    user_id: str,
    role: str,
) -> SkillCatalog:
    catalog = _load_bundled_skill_catalog()
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
    bundled_ids = {manifest.skill_id for manifest in catalog.manifests()}
    for skill in skills:
        if skill.id not in allowed_ids:
            continue
        manifest = SkillManifest(
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
        )
        if manifest.skill_id in bundled_ids:
            logger.warning(
                "Ignoring organization skill %s because Aloy owns that skill id",
                manifest.skill_id,
            )
            continue
        catalog.register(manifest, skill.instructions)
    return catalog


__all__ = ["SURFACE_BUILDER_SKILL_ID", "load_skill_catalog"]
