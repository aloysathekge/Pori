"""Metadata-first, tool-backed progressive skill loading."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .capabilities import SkillEligibility
from .runtime import stable_fingerprint
from .tools.registry import CapabilitySnapshot


class SkillManifest(BaseModel):
    """Cheap metadata available before full skill instructions are loaded."""

    model_config = ConfigDict(frozen=True)

    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
    name: str = Field(min_length=1, max_length=120)
    version: str = Field(min_length=1, max_length=64)
    summary: str = Field(min_length=1, max_length=500)
    tags: Tuple[str, ...] = ()
    required_tools: frozenset[str] = frozenset()
    required_credentials: Tuple[str, ...] = ()
    required_platforms: Tuple[str, ...] = ()
    required_model_capabilities: frozenset[str] = frozenset()
    source: str = "local"
    sensitivity: str = "internal"
    instructions_fingerprint: Optional[str] = None

    @property
    def skill_id(self) -> str:
        return f"{self.slug}@{self.version}"

    def eligibility(self) -> SkillEligibility:
        return SkillEligibility(
            required_tools=self.required_tools,
            required_credentials=self.required_credentials,
            required_platforms=self.required_platforms,
            required_model_capabilities=self.required_model_capabilities,
            version=self.version,
            source=self.source,
            sensitivity=self.sensitivity,
        )


class SkillSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: str
    name: str
    summary: str
    tags: Tuple[str, ...]
    eligible: bool
    reasons: Tuple[str, ...] = ()


class SelectedSkill(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest: SkillManifest
    instructions: str
    fingerprint: str


@dataclass(frozen=True)
class _SkillEntry:
    manifest: SkillManifest
    loader: Callable[[], str]


class SkillCatalog:
    """Versioned skill metadata with explicit, lazy content loading."""

    def __init__(self, *, max_instruction_chars: int = 50_000):
        self.max_instruction_chars = max_instruction_chars
        self._entries: Dict[str, _SkillEntry] = {}

    def register(
        self,
        manifest: SkillManifest,
        instructions: str | Callable[[], str],
    ) -> None:
        if manifest.skill_id in self._entries:
            raise ValueError(f"Skill '{manifest.skill_id}' is already registered")
        loader = instructions if callable(instructions) else lambda: instructions
        self._entries[manifest.skill_id] = _SkillEntry(manifest, loader)

    def manifests(self) -> Tuple[SkillManifest, ...]:
        return tuple(entry.manifest for _, entry in sorted(self._entries.items()))

    def summaries(
        self,
        snapshot: CapabilitySnapshot,
        *,
        model_capabilities: frozenset[str] = frozenset(),
    ) -> Tuple[SkillSummary, ...]:
        summaries = []
        for manifest in self.manifests():
            report = manifest.eligibility().evaluate(
                available_tools=snapshot.tool_names,
                model_capabilities=model_capabilities,
            )
            summaries.append(
                SkillSummary(
                    skill_id=manifest.skill_id,
                    name=manifest.name,
                    summary=manifest.summary,
                    tags=manifest.tags,
                    eligible=report.eligible,
                    reasons=report.reasons,
                )
            )
        return tuple(summaries)

    @staticmethod
    def _terms(value: str) -> frozenset[str]:
        return frozenset(re.findall(r"[a-z0-9]+", value.lower()))

    def select(
        self,
        query: str,
        snapshot: CapabilitySnapshot,
        *,
        model_capabilities: frozenset[str] = frozenset(),
        limit: int = 3,
        explicit_skill_ids: Optional[Iterable[str]] = None,
    ) -> Tuple[SkillSummary, ...]:
        summaries = self.summaries(snapshot, model_capabilities=model_capabilities)
        if explicit_skill_ids is not None:
            requested = set(explicit_skill_ids)
            unknown = requested.difference(self._entries)
            if unknown:
                raise ValueError(f"Unknown skills: {', '.join(sorted(unknown))}")
            selected = [item for item in summaries if item.skill_id in requested]
        else:
            query_terms = self._terms(query)
            ranked = []
            for item in summaries:
                manifest = self._entries[item.skill_id].manifest
                metadata_terms = self._terms(
                    " ".join(
                        (manifest.slug, manifest.name, manifest.summary, *manifest.tags)
                    )
                )
                score = len(query_terms.intersection(metadata_terms))
                if score:
                    ranked.append((score, item.skill_id, item))
            selected = [item for _, _, item in sorted(ranked, reverse=True)]
        ineligible = [item for item in selected if not item.eligible]
        if ineligible:
            details = "; ".join(
                f"{item.skill_id}: {', '.join(item.reasons)}" for item in ineligible
            )
            raise ValueError(f"Selected skills are ineligible: {details}")
        return tuple(selected[: max(0, limit)])

    def load(self, skill_id: str) -> SelectedSkill:
        try:
            entry = self._entries[skill_id]
        except KeyError as exc:
            raise ValueError(f"Unknown skill '{skill_id}'") from exc
        instructions = entry.loader().strip()
        if not instructions:
            raise ValueError(f"Skill '{skill_id}' has empty instructions")
        if len(instructions) > self.max_instruction_chars:
            raise ValueError(
                f"Skill '{skill_id}' exceeds {self.max_instruction_chars} characters"
            )
        fingerprint = stable_fingerprint(instructions)
        expected = entry.manifest.instructions_fingerprint
        if expected and fingerprint != expected:
            raise ValueError(f"Skill '{skill_id}' instructions fingerprint mismatch")
        return SelectedSkill(
            manifest=entry.manifest,
            instructions=instructions,
            fingerprint=fingerprint,
        )

    def load_selected(
        self, summaries: Iterable[SkillSummary]
    ) -> Tuple[SelectedSkill, ...]:
        return tuple(self.load(summary.skill_id) for summary in summaries)


def render_selected_skills(skills: Iterable[SelectedSkill]) -> str:
    sections = []
    for skill in skills:
        sections.append(
            f"## Skill: {skill.manifest.name} ({skill.manifest.skill_id})\n"
            f"{skill.instructions}\n\n"
            "All external effects must use the currently authorized tools."
        )
    return "\n\n".join(sections)


_SKIP_SCAN_DIRS = frozenset(
    {".git", ".hg", ".svn", ".venv", "venv", "__pycache__", "node_modules"}
)


def _normalize_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", value.strip().lower().replace("_", "-"))
    slug = re.sub(r"-+", "-", slug).strip("-")
    if len(slug) < 3:
        slug = f"{slug or 'skill'}-skill"
    return slug[:64].strip("-")


def _as_tuple(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, Iterable):
        items = [str(item).strip() for item in value]
    else:
        items = [str(value).strip()]
    return tuple(item for item in items if item)


def _read_skill_file(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    raw_metadata = yaml.safe_load(parts[1]) or {}
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    return metadata, parts[2].strip()


def _summary_from_body(body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip(" #\t")
        if stripped:
            return stripped[:500]
    return "Local Pori skill"


def _iter_skill_files(directory: Path) -> Iterable[Path]:
    if not directory.exists() or not directory.is_dir():
        return
    for child in sorted(directory.rglob("SKILL.md")):
        if any(part in _SKIP_SCAN_DIRS for part in child.parts):
            continue
        yield child


def load_skill_catalog_from_directories(
    directories: Iterable[str | Path],
    *,
    max_instruction_chars: int = 50_000,
) -> SkillCatalog:
    """Build a skill catalog from local SKILL.md packages."""

    catalog = SkillCatalog(max_instruction_chars=max_instruction_chars)
    for directory in directories:
        base_dir = Path(directory).expanduser()
        if not base_dir.is_absolute():
            base_dir = Path.cwd() / base_dir
        for skill_file in _iter_skill_files(base_dir.resolve()):
            metadata, instructions = _read_skill_file(skill_file)
            skill_dir = skill_file.parent
            nested = (
                metadata.get("pori") if isinstance(metadata.get("pori"), dict) else {}
            )
            slug = _normalize_slug(
                str(metadata.get("slug") or metadata.get("name") or skill_dir.name)
            )
            manifest = SkillManifest(
                slug=slug,
                name=str(metadata.get("name") or slug.replace("-", " ").title()),
                version=str(metadata.get("version") or "1"),
                summary=str(
                    metadata.get("summary")
                    or metadata.get("description")
                    or _summary_from_body(instructions)
                )[:500],
                tags=_as_tuple(metadata.get("tags") or nested.get("tags")),
                required_tools=frozenset(
                    _as_tuple(
                        metadata.get("required_tools") or nested.get("required_tools")
                    )
                ),
                required_credentials=_as_tuple(
                    metadata.get("required_credentials")
                    or metadata.get("required_environment_variables")
                    or nested.get("required_credentials")
                ),
                required_platforms=_as_tuple(
                    metadata.get("required_platforms")
                    or nested.get("required_platforms")
                ),
                required_model_capabilities=frozenset(
                    _as_tuple(
                        metadata.get("required_model_capabilities")
                        or nested.get("required_model_capabilities")
                    )
                ),
                source=f"local:{skill_dir}",
                sensitivity=str(metadata.get("sensitivity") or "internal"),
                instructions_fingerprint=metadata.get("instructions_fingerprint"),
            )
            catalog.register(manifest, instructions)
    return catalog


__all__ = [
    "load_skill_catalog_from_directories",
    "SelectedSkill",
    "SkillCatalog",
    "SkillManifest",
    "SkillSummary",
    "render_selected_skills",
]
