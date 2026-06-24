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
    category: str = "local"
    author: str = ""
    license: str = ""
    commands: Tuple[str, ...] = ()
    argument_hint: str = ""
    required_tools: frozenset[str] = frozenset()
    required_credentials: Tuple[str, ...] = ()
    required_platforms: Tuple[str, ...] = ()
    required_model_capabilities: frozenset[str] = frozenset()
    source: str = "local"
    source_url: str = ""
    install_command: str = ""
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


class SkillIndexEntry(BaseModel):
    """Searchable, Hermes-style metadata for a skill without instructions."""

    model_config = ConfigDict(frozen=True)

    skill_id: str
    slug: str
    name: str
    version: str
    summary: str
    tags: Tuple[str, ...]
    category: str
    author: str
    license: str
    commands: Tuple[str, ...]
    argument_hint: str
    source: str
    source_url: str
    install_command: str
    eligible: bool
    reasons: Tuple[str, ...] = ()


class SkillSearchHit(BaseModel):
    """One ranked skill-index match for a user request."""

    model_config = ConfigDict(frozen=True)

    entry: SkillIndexEntry
    score: int
    matched_terms: Tuple[str, ...] = ()


class SkillInvocation(BaseModel):
    """A selected skill plus the user's meaningful invocation text."""

    model_config = ConfigDict(frozen=True)

    skill_id: str
    invocation_text: str
    missing_argument: bool = False
    argument_hint: str = ""


class SkillLinkedFile(BaseModel):
    """A supporting file packaged with a local skill."""

    model_config = ConfigDict(frozen=True)

    path: str
    kind: str = "file"
    size_bytes: int = Field(default=0, ge=0)


class SkillConfigDeclaration(BaseModel):
    """A config value declared by skill frontmatter."""

    model_config = ConfigDict(frozen=True)

    key: str
    description: str
    default: Any = None
    prompt: Optional[str] = None


class SelectedSkill(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest: SkillManifest
    instructions: str
    fingerprint: str


class SkillFileView(BaseModel):
    """Loaded content for a skill's main instructions or linked file."""

    model_config = ConfigDict(frozen=True)

    manifest: SkillManifest
    path: str
    content: str
    linked_files: Tuple[SkillLinkedFile, ...] = ()


@dataclass(frozen=True)
class _SkillEntry:
    manifest: SkillManifest
    loader: Callable[[], str]
    root_path: Optional[Path] = None
    config_declarations: Tuple[SkillConfigDeclaration, ...] = ()


class SkillCatalog:
    """Versioned skill metadata with explicit, lazy content loading."""

    def __init__(self, *, max_instruction_chars: int = 50_000):
        self.max_instruction_chars = max_instruction_chars
        self._entries: Dict[str, _SkillEntry] = {}

    def register(
        self,
        manifest: SkillManifest,
        instructions: str | Callable[[], str],
        *,
        root_path: Optional[Path] = None,
        config_declarations: Iterable[SkillConfigDeclaration] = (),
    ) -> None:
        if manifest.skill_id in self._entries:
            raise ValueError(f"Skill '{manifest.skill_id}' is already registered")
        loader = instructions if callable(instructions) else lambda: instructions
        self._entries[manifest.skill_id] = _SkillEntry(
            manifest,
            loader,
            root_path=root_path,
            config_declarations=tuple(config_declarations),
        )

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

    def index(
        self,
        snapshot: CapabilitySnapshot,
        *,
        model_capabilities: frozenset[str] = frozenset(),
    ) -> Tuple[SkillIndexEntry, ...]:
        """Return searchable metadata for all known skills."""

        entries = []
        for manifest in self.manifests():
            report = manifest.eligibility().evaluate(
                available_tools=snapshot.tool_names,
                model_capabilities=model_capabilities,
            )
            entries.append(
                SkillIndexEntry(
                    skill_id=manifest.skill_id,
                    slug=manifest.slug,
                    name=manifest.name,
                    version=manifest.version,
                    summary=manifest.summary,
                    tags=manifest.tags,
                    category=manifest.category,
                    author=manifest.author,
                    license=manifest.license,
                    commands=manifest.commands,
                    argument_hint=manifest.argument_hint,
                    source=manifest.source,
                    source_url=manifest.source_url,
                    install_command=manifest.install_command,
                    eligible=report.eligible,
                    reasons=report.reasons,
                )
            )
        return tuple(entries)

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
            summary_by_id = {item.skill_id: item for item in summaries}
            selected = [
                summary_by_id[hit.entry.skill_id]
                for hit in self.search(
                    query,
                    snapshot,
                    model_capabilities=model_capabilities,
                    limit=limit,
                )
            ]
        ineligible = [item for item in selected if not item.eligible]
        if ineligible:
            details = "; ".join(
                f"{item.skill_id}: {', '.join(item.reasons)}" for item in ineligible
            )
            raise ValueError(f"Selected skills are ineligible: {details}")
        return tuple(selected[: max(0, limit)])

    def search(
        self,
        query: str,
        snapshot: CapabilitySnapshot,
        *,
        model_capabilities: frozenset[str] = frozenset(),
        limit: int = 10,
        min_score: int = 1,
    ) -> Tuple[SkillSearchHit, ...]:
        """Search the skill index by slug, name, summary, tags, and commands."""

        query_terms = self._terms(query)
        if not query_terms:
            return ()
        hits = []
        for entry in self.index(snapshot, model_capabilities=model_capabilities):
            metadata = " ".join(
                (
                    entry.slug,
                    entry.name,
                    entry.summary,
                    entry.category,
                    *entry.tags,
                    *entry.commands,
                )
            )
            metadata_terms = self._terms(metadata)
            matched = query_terms.intersection(metadata_terms)
            score = len(matched)
            lowered_query = query.casefold()
            if entry.slug.casefold() in lowered_query:
                score += 8
            if entry.name.casefold() in lowered_query:
                score += 6
            if any(tag.casefold() in lowered_query for tag in entry.tags):
                score += 3
            if score >= min_score:
                hits.append(
                    SkillSearchHit(
                        entry=entry,
                        score=score,
                        matched_terms=tuple(sorted(matched)),
                    )
                )
        ranked = sorted(
            hits,
            key=lambda hit: (
                hit.score,
                hit.entry.eligible,
                hit.entry.skill_id,
            ),
            reverse=True,
        )
        return tuple(ranked[: max(0, limit)])

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

    def resolve_skill_id(self, identifier: str) -> str:
        requested = identifier.strip().lower()
        for skill_id, entry in self._entries.items():
            manifest = entry.manifest
            if requested in {skill_id.lower(), manifest.slug.lower()}:
                return skill_id
        raise ValueError(f"Unknown skill '{identifier}'")

    def build_invocation(self, skill_id: str, task: str) -> SkillInvocation:
        entry = self._entry_for(skill_id)
        invocation_text = _skill_invocation_text(entry.manifest, task)
        return SkillInvocation(
            skill_id=skill_id,
            invocation_text=invocation_text,
            missing_argument=bool(
                entry.manifest.argument_hint
                and _missing_invocation_argument(invocation_text)
            ),
            argument_hint=entry.manifest.argument_hint,
        )

    def linked_files(self, skill_id: str) -> Tuple[SkillLinkedFile, ...]:
        entry = self._entry_for(skill_id)
        if entry.root_path is None:
            return ()
        return tuple(_iter_linked_files(entry.root_path))

    def config_declarations(self, skill_id: str) -> Tuple[SkillConfigDeclaration, ...]:
        return self._entry_for(skill_id).config_declarations

    def view_file(
        self, skill_id: str, file_path: Optional[str] = None
    ) -> SkillFileView:
        entry = self._entry_for(skill_id)
        if file_path is None or not file_path.strip():
            selected = self.load(skill_id)
            return SkillFileView(
                manifest=entry.manifest,
                path="SKILL.md",
                content=selected.instructions,
                linked_files=self.linked_files(skill_id),
            )

        if entry.root_path is None:
            raise ValueError(f"Skill '{skill_id}' has no local directory")
        path = _resolve_linked_file(entry.root_path, file_path)
        content = path.read_text(encoding="utf-8")
        if len(content) > self.max_instruction_chars:
            raise ValueError(
                f"Skill file '{file_path}' exceeds {self.max_instruction_chars} characters"
            )
        return SkillFileView(
            manifest=entry.manifest,
            path=_relative_posix(entry.root_path, path),
            content=content,
            linked_files=self.linked_files(skill_id),
        )

    def _entry_for(self, skill_id: str) -> _SkillEntry:
        try:
            return self._entries[skill_id]
        except KeyError as exc:
            raise ValueError(f"Unknown skill '{skill_id}'") from exc


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
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
    }
)
_SKILL_SUPPORT_DIRS = frozenset({"references", "templates", "assets", "scripts"})


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


def _nested_mapping(metadata: dict[str, Any], key: str) -> dict[str, Any]:
    value = metadata.get(key)
    return value if isinstance(value, dict) else {}


def _pori_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    nested = _nested_mapping(metadata, "pori")
    frontmatter_metadata = _nested_mapping(metadata, "metadata")
    nested_metadata = _nested_mapping(frontmatter_metadata, "pori")
    return {**nested_metadata, **nested}


def _extract_config_declarations(
    metadata: dict[str, Any]
) -> Tuple[SkillConfigDeclaration, ...]:
    raw = _pori_metadata(metadata).get("config")
    if raw is None:
        return ()
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return ()
    declarations = []
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip()
        description = str(item.get("description", "")).strip()
        if not key or not description or key in seen:
            continue
        declarations.append(
            SkillConfigDeclaration(
                key=key,
                description=description,
                default=item.get("default"),
                prompt=item.get("prompt"),
            )
        )
        seen.add(key)
    return tuple(declarations)


def _resolve_config_value(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _render_skill_config(
    declarations: Iterable[SkillConfigDeclaration],
    config_values: dict[str, Any],
) -> str:
    lines = []
    for declaration in declarations:
        key = declaration.key
        value = _resolve_config_value(config_values, key)
        if value is None or (isinstance(value, str) and not value.strip()):
            value = declaration.default or ""
        lines.append(f"  {key} = {value if value != '' else '(not set)'}")
    if not lines:
        return ""
    return "\n\n[Skill config]\n" + "\n".join(lines) + "\n[/Skill config]"


def _append_skill_config(
    instructions: str,
    declarations: Iterable[SkillConfigDeclaration],
    config_values: dict[str, Any],
) -> str:
    rendered = _render_skill_config(declarations, config_values)
    if not rendered:
        return instructions
    return f"{instructions.rstrip()}{rendered}"


def _skill_invocation_text(manifest: SkillManifest, task: str) -> str:
    text = " ".join(task.strip().split())
    if not text:
        return ""
    removable = {
        manifest.slug.casefold(),
        manifest.name.casefold(),
        *(command.casefold() for command in manifest.commands),
    }
    normalized = text.casefold()
    for phrase in sorted((item for item in removable if item), key=len, reverse=True):
        normalized = re.sub(rf"\b{re.escape(phrase)}\b", " ", normalized)
    normalized = re.sub(
        r"\b(i|me|my|you|want|would|like|please|can|could|to|the|a|an)\b",
        " ",
        normalized,
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _missing_invocation_argument(invocation_text: str) -> bool:
    text = invocation_text.strip().casefold()
    if not text:
        return True
    vague_terms = {
        "something",
        "anything",
        "thing",
        "stuff",
        "whatever",
        "something new",
        "anything else",
    }
    if text in vague_terms:
        return True
    return len(text.split()) < 1


def _skill_instruction_loader(
    instructions: str,
    declarations: Tuple[SkillConfigDeclaration, ...],
    config_values: dict[str, Any],
) -> Callable[[], str]:
    def load() -> str:
        return _append_skill_config(instructions, declarations, config_values)

    return load


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
        if _is_skill_support_path(child):
            continue
        yield child


def _relative_posix(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _linked_file_kind(path: str) -> str:
    first = path.split("/", 1)[0]
    if first in _SKILL_SUPPORT_DIRS:
        return first
    return "file"


def _iter_linked_files(skill_root: Path) -> Iterable[SkillLinkedFile]:
    if not skill_root.exists() or not skill_root.is_dir():
        return
    for child in sorted(skill_root.rglob("*")):
        if child.is_dir():
            continue
        if child.name == "SKILL.md":
            continue
        if any(part in _SKIP_SCAN_DIRS for part in child.parts):
            continue
        rel_path = _relative_posix(skill_root, child)
        yield SkillLinkedFile(
            path=rel_path,
            kind=_linked_file_kind(rel_path),
            size_bytes=child.stat().st_size,
        )


def _resolve_linked_file(skill_root: Path, file_path: str) -> Path:
    candidate = Path(file_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Skill file path must stay within the skill directory")
    resolved_root = skill_root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            "Skill file path must stay within the skill directory"
        ) from exc
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"Skill file '{file_path}' not found")
    if resolved.name == "SKILL.md":
        raise ValueError("Use view_file(skill_id) to load SKILL.md")
    return resolved


def _is_skill_support_path(path: Path) -> bool:
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        if part not in _SKILL_SUPPORT_DIRS or index == 0:
            continue
        skill_root = Path(*parts[:index])
        if (skill_root / "SKILL.md").exists():
            return True
    return False


def load_skill_catalog_from_directories(
    directories: Iterable[str | Path],
    *,
    disabled: Iterable[str] = (),
    config_values: Optional[dict[str, Any]] = None,
    max_instruction_chars: int = 50_000,
) -> SkillCatalog:
    """Build a skill catalog from local SKILL.md packages."""

    catalog = SkillCatalog(max_instruction_chars=max_instruction_chars)
    disabled_names = {
        str(item).strip().lower() for item in disabled if str(item).strip()
    }
    skill_config = config_values or {}
    for directory in directories:
        base_dir = Path(directory).expanduser()
        if not base_dir.is_absolute():
            base_dir = Path.cwd() / base_dir
        for skill_file in _iter_skill_files(base_dir.resolve()):
            metadata, instructions = _read_skill_file(skill_file)
            skill_dir = skill_file.parent
            nested = _pori_metadata(metadata)
            slug = _normalize_slug(
                str(metadata.get("slug") or metadata.get("name") or skill_dir.name)
            )
            name = str(metadata.get("name") or slug.replace("-", " ").title())
            if slug.lower() in disabled_names or name.lower() in disabled_names:
                continue
            config_declarations = _extract_config_declarations(metadata)
            manifest = SkillManifest(
                slug=slug,
                name=name,
                version=str(metadata.get("version") or "1"),
                summary=str(
                    metadata.get("summary")
                    or metadata.get("description")
                    or _summary_from_body(instructions)
                )[:500],
                tags=_as_tuple(metadata.get("tags") or nested.get("tags")),
                category=str(
                    metadata.get("category") or nested.get("category") or "local"
                ),
                author=str(metadata.get("author") or nested.get("author") or ""),
                license=str(metadata.get("license") or nested.get("license") or ""),
                commands=_as_tuple(
                    metadata.get("commands")
                    or metadata.get("command")
                    or nested.get("commands")
                ),
                argument_hint=str(
                    metadata.get("argument-hint")
                    or metadata.get("argument_hint")
                    or nested.get("argument_hint")
                    or ""
                ),
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
                source_url=str(
                    metadata.get("source_url")
                    or metadata.get("url")
                    or nested.get("source_url")
                    or ""
                ),
                install_command=str(
                    metadata.get("install_command")
                    or metadata.get("installCmd")
                    or nested.get("install_command")
                    or f"copy {skill_dir} into .pori/skills"
                ),
                sensitivity=str(metadata.get("sensitivity") or "internal"),
                instructions_fingerprint=metadata.get("instructions_fingerprint"),
            )
            try:
                catalog.register(
                    manifest,
                    _skill_instruction_loader(
                        instructions,
                        config_declarations,
                        skill_config,
                    ),
                    root_path=skill_dir,
                    config_declarations=config_declarations,
                )
            except ValueError:
                continue
    return catalog


__all__ = [
    "load_skill_catalog_from_directories",
    "SkillConfigDeclaration",
    "SkillFileView",
    "SkillIndexEntry",
    "SkillInvocation",
    "SkillLinkedFile",
    "SkillSearchHit",
    "SelectedSkill",
    "SkillCatalog",
    "SkillManifest",
    "SkillSummary",
    "render_selected_skills",
]
