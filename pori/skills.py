"""Metadata-first, tool-backed progressive skill loading."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from .capabilities import EligibilityReport, SkillEligibility
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


__all__ = [
    "SelectedSkill",
    "SkillCatalog",
    "SkillManifest",
    "SkillSummary",
    "render_selected_skills",
]
