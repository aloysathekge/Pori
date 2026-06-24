"""Runtime tools for progressive skill discovery and viewing."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from ...skills import SkillCatalog
from ..registry import ToolRegistry


class SkillsListParams(BaseModel):
    query: str = Field(
        "",
        description="Optional search query. Leave empty to list all available skills.",
    )
    limit: int = Field(10, ge=1, le=50, description="Maximum skills to return.")


class SkillViewParams(BaseModel):
    skill: str = Field(..., description="Skill slug or skill id, e.g. teach or teach@1")
    file_path: Optional[str] = Field(
        None,
        description=(
            "Optional linked file path inside the skill, such as "
            "references/examples.md. Omit to view SKILL.md."
        ),
    )


def _catalog_from_context(context: Dict[str, Any]) -> SkillCatalog | None:
    catalog = context.get("skill_catalog")
    return catalog if isinstance(catalog, SkillCatalog) else None


def _snapshot_from_context(context: Dict[str, Any]):
    snapshot = context.get("capability_snapshot")
    if snapshot is not None:
        return snapshot
    registry = context.get("tools_registry")
    if registry is not None and hasattr(registry, "snapshot"):
        return registry.snapshot()
    return None


def register_skill_tools(registry: ToolRegistry) -> None:
    """Register skill catalog tools on the provided registry."""

    @registry.tool(
        name="skills_list",
        param_model=SkillsListParams,
        description=(
            "List or search available skills by metadata without loading full "
            "instructions."
        ),
    )
    def skills_list_tool(params: SkillsListParams, context: Dict[str, Any]):
        catalog = _catalog_from_context(context)
        snapshot = _snapshot_from_context(context)
        if catalog is None:
            return {
                "available": False,
                "error": "No skill catalog is configured for this run.",
                "skills": [],
            }
        if snapshot is None:
            return {
                "available": False,
                "error": "No capability snapshot is available for skill eligibility.",
                "skills": [],
            }

        if params.query.strip():
            hits = catalog.search(params.query, snapshot, limit=params.limit)
            return {
                "available": True,
                "query": params.query,
                "skills": [
                    {
                        **hit.entry.model_dump(),
                        "score": hit.score,
                        "matched_terms": list(hit.matched_terms),
                    }
                    for hit in hits
                ],
            }

        return {
            "available": True,
            "query": "",
            "skills": [
                item.model_dump() for item in catalog.index(snapshot)[: params.limit]
            ],
        }

    @registry.tool(
        name="skill_view",
        param_model=SkillViewParams,
        description=(
            "Load a skill's full instructions or one linked file on demand. "
            "Use after skills_list identifies a relevant skill."
        ),
    )
    def skill_view_tool(params: SkillViewParams, context: Dict[str, Any]):
        catalog = _catalog_from_context(context)
        if catalog is None:
            return {
                "available": False,
                "error": "No skill catalog is configured for this run.",
            }
        skill_id = catalog.resolve_skill_id(params.skill)
        view = catalog.view_file(skill_id, params.file_path)
        return {
            "available": True,
            "skill_id": view.manifest.skill_id,
            "name": view.manifest.name,
            "path": view.path,
            "content": view.content,
            "linked_files": [linked.model_dump() for linked in view.linked_files],
        }


__all__ = ["SkillViewParams", "SkillsListParams", "register_skill_tools"]
