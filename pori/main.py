import asyncio
import json
import logging
import os
import re
import sys
from typing import Any, List, Optional, Tuple, cast

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import modules
from pathlib import Path

from .agent import Agent, AgentSettings
from .config import Config, LLMConfig, get_configured_llm
from .evolution import (
    EvolutionEvalResult,
    EvolutionProposal,
    FileEvolutionRepository,
    run_local_evolution_evals,
)
from .hitl import CLIHITLHandler
from .memory import AgentMemory, create_memory_store
from .observability import build_tool_preview
from .orchestrator import Orchestrator
from .skills import (
    SkillBundleCatalog,
    SkillCatalog,
    inspect_skill_source,
    install_skill_source,
    load_skill_bundles_from_directory,
    load_skill_catalog_from_directories,
    uninstall_skill_from_directory,
)
from .team import Team
from .tools.registry import ToolRegistry, tool_registry
from .tools.standard import register_all_tools

# Configure logging
from .utils.file_refs import expand_file_refs
from .utils.logging_config import setup_logging
from .utils.prompt_loader import set_prompts_dir


def _cli_log_file() -> Optional[str]:
    """Full INFO logs go to .pori/pori.log so the console can stay clean."""
    try:
        log_dir = Path.cwd() / ".pori"
        log_dir.mkdir(parents=True, exist_ok=True)
        return str(log_dir / "pori.log")
    except Exception:
        return None


# End-user default: clean console (warnings/errors + the "•" status lines); the
# full INFO trace still goes to .pori/pori.log. Developers set PORI_VERBOSE=1
# (e.g. in .env) to stream the full INFO logs to the console as before.
_verbose = bool(os.getenv("PORI_VERBOSE"))
loggers = setup_logging(
    level=logging.INFO,
    include_http=True,
    console_level=logging.INFO if _verbose else logging.WARNING,
    log_file=_cli_log_file(),
)
logger = logging.getLogger("pori.main")


def _console_safe_text(value: Any, encoding: Optional[str] = None) -> str:
    """Return text that can be written to the current console encoding."""
    text = str(value)
    target_encoding = encoding or getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        return text.encode(target_encoding, errors="replace").decode(
            target_encoding,
            errors="replace",
        )
    except LookupError:
        return text.encode("utf-8", errors="replace").decode(
            "utf-8",
            errors="replace",
        )


def _safe_print(value: Any = "", *, flush: bool = False) -> None:
    print(_console_safe_text(value), flush=flush)


# Define some example tools


def _load_cli_evolution_repository(config: Config) -> Optional[FileEvolutionRepository]:
    evolution_config = getattr(config, "evolution", None)
    if not evolution_config or not evolution_config.enabled:
        return None
    path = Path(evolution_config.path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return FileEvolutionRepository(path.resolve())


def _load_cli_skill_catalog(config: Config) -> Optional[SkillCatalog]:
    skills_config = getattr(config, "skills", None)
    if not skills_config or not skills_config.enabled:
        return None
    directories = [
        skills_config.default_dir,
        *skills_config.directories,
        *skills_config.external_dirs,
    ]
    resolved_directories = []
    seen = set()
    for directory in directories:
        path = Path(directory).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        resolved_directories.append(path)
    if resolved_directories:
        resolved_directories[0].mkdir(parents=True, exist_ok=True)
    return load_skill_catalog_from_directories(
        resolved_directories,
        disabled=skills_config.disabled,
        config_values=skills_config.config,
        max_instruction_chars=skills_config.max_instruction_chars,
    )


def _load_cli_skill_bundles(config: Config) -> Optional[SkillBundleCatalog]:
    skills_config = getattr(config, "skills", None)
    if not skills_config or not skills_config.enabled:
        return None
    path = Path(skills_config.bundles_dir).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path.mkdir(parents=True, exist_ok=True)
    return load_skill_bundles_from_directory(path.resolve())


def _cli_skill_default_dir(config: Optional[Config]) -> Path:
    skills_config = getattr(config, "skills", None) if config is not None else None
    directory = (
        getattr(skills_config, "default_dir", "./.pori/skills")
        if skills_config is not None
        else "./.pori/skills"
    )
    path = Path(directory).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _summarize_written_artifacts(tool_calls: List[Any]) -> List[str]:
    """Return display lines for successful file-writing tool calls."""
    artifact_lines: List[str] = []
    for call in tool_calls:
        if not getattr(call, "success", False):
            continue
        tool_name = getattr(call, "tool_name", "")
        if tool_name not in {"write_file", "sandbox_write_file"}:
            continue
        params = getattr(call, "parameters", {}) or {}
        result = getattr(call, "result", {}) or {}
        if not isinstance(params, dict):
            params = {}
        if not isinstance(result, dict):
            result = {}

        path = (
            params.get("file_path")
            or params.get("path")
            or result.get("path")
            or result.get("file_path")
        )
        if not path:
            file_info = result.get("file_info")
            if isinstance(file_info, dict):
                path = file_info.get("path")
        if not path:
            path = "(path unavailable)"

        byte_count = result.get("bytes_written")
        append = bool(params.get("append"))
        action = "appended" if append else "wrote"
        detail = f"{tool_name}: {action} {path}"
        if isinstance(byte_count, int):
            detail += f" ({byte_count} bytes)"
        artifact_lines.append(detail)
    return artifact_lines


def _summarize_loaded_skills(tool_calls: List[Any]) -> List[str]:
    """Return skill ids loaded by the agent at runtime through skill_view."""
    loaded: List[str] = []
    seen: set[str] = set()
    for call in tool_calls:
        if not getattr(call, "success", False):
            continue
        if getattr(call, "tool_name", "") != "skill_view":
            continue
        params = getattr(call, "parameters", {}) or {}
        result = getattr(call, "result", {}) or {}
        if not isinstance(params, dict):
            params = {}
        if not isinstance(result, dict):
            result = {}
        inner_result = result.get("result")
        if not isinstance(inner_result, dict):
            inner_result = {}
        skill_id = (
            inner_result.get("skill_id")
            or params.get("skill")
            or inner_result.get("name")
            or "(unknown skill)"
        )
        skill_id = str(skill_id)
        if skill_id not in seen:
            seen.add(skill_id)
            loaded.append(skill_id)
    return loaded


def _format_tool_call_parameters(parameters: Any) -> str:
    """Compact tool parameters for CLI display without dumping file contents."""
    if not isinstance(parameters, dict):
        return str(parameters)
    display: dict[str, Any] = {}
    for key, value in parameters.items():
        if key == "content" and isinstance(value, str):
            display[key] = f"<{len(value)} chars>"
        else:
            display[key] = value
    return str(display)


def _print_skill_catalog(
    skill_catalog: Optional[SkillCatalog],
    registry: ToolRegistry,
    query: str = "",
    bundle_catalog: Optional[SkillBundleCatalog] = None,
) -> None:
    if skill_catalog is None or not skill_catalog.manifests():
        print("No local skills configured.")
        print("Add SKILL.md packages under .pori/skills, then run /reload-skills.")
        return

    if query.strip():
        print(f"\n=== Skill Search: {query.strip()} ===")
        hits = skill_catalog.search(query, registry.snapshot())
        if not hits:
            print("No matching local skills.")
            return
        for hit in hits:
            item = hit.entry
            status = (
                "available"
                if item.eligible
                else f"ineligible: {', '.join(item.reasons)}"
            )
            tags = f" [{', '.join(item.tags)}]" if item.tags else ""
            matched = ", ".join(hit.matched_terms) if hit.matched_terms else "none"
            print(
                f"  {item.skill_id}{tags} - {status} "
                f"(score={hit.score}, matched={matched})"
            )
            print(f"    {item.summary}")
            print(f"    source: {item.source}")
            print(f"    readiness: {item.readiness}")
            if item.readiness_reasons:
                print(f"    readiness details: {', '.join(item.readiness_reasons)}")
            if item.install_command:
                print(f"    install: {item.install_command}")
        return

    print("\n=== Local Skills ===")
    for item in skill_catalog.index(registry.snapshot()):
        status = (
            "available" if item.eligible else f"ineligible: {', '.join(item.reasons)}"
        )
        tags = f" [{', '.join(item.tags)}]" if item.tags else ""
        print(f"  {item.skill_id}{tags} - {status}")
        print(f"    {item.summary}")
        print(f"    category: {item.category}")
        print(f"    source: {item.source}")
        print(f"    provenance: {item.provenance} ({item.trust_level})")
        print(f"    readiness: {item.readiness}")
        if item.readiness_reasons:
            print(f"    readiness details: {', '.join(item.readiness_reasons)}")
        if item.commands:
            print(f"    commands: {', '.join(item.commands)}")
        if item.install_command:
            print(f"    install: {item.install_command}")
    if bundle_catalog is not None and bundle_catalog.bundles():
        print("\n=== Skill Bundles ===")
        for bundle in bundle_catalog.bundles():
            print(f"  /{bundle.slug} - {bundle.description or bundle.name}")
            print(f"    skills: {', '.join(bundle.skills)}")
    print(
        "\nUse /skills <query> to search, or /<skill-slug> your task to force a skill."
    )


def _print_skill_detail(
    skill_catalog: Optional[SkillCatalog],
    registry: ToolRegistry,
    identifier: str,
    file_path: Optional[str] = None,
) -> None:
    if skill_catalog is None:
        print("No local skills configured.")
        return
    try:
        skill_id = skill_catalog.resolve_skill_id(identifier)
        view = skill_catalog.view_file(skill_id, file_path)
    except Exception as e:
        print(f"Skill error: {e}")
        return

    if file_path:
        print(f"\n=== {view.manifest.name}: {view.path} ===")
        print(view.content)
        return

    summary = next(
        item
        for item in skill_catalog.summaries(registry.snapshot())
        if item.skill_id == skill_id
    )
    status = (
        "available" if summary.eligible else f"ineligible: {', '.join(summary.reasons)}"
    )
    print(f"\n=== Skill: {view.manifest.name} ===")
    print(f"ID: {view.manifest.skill_id}")
    print(f"Status: {status}")
    print(f"Source: {view.manifest.source}")
    print(f"Category: {view.manifest.category}")
    if view.manifest.author:
        print(f"Author: {view.manifest.author}")
    if view.manifest.license:
        print(f"License: {view.manifest.license}")
    if view.manifest.source_url:
        print(f"Source URL: {view.manifest.source_url}")
    print(f"Provenance: {view.manifest.provenance} ({view.manifest.trust_level})")
    readiness = skill_catalog.readiness(skill_id)
    print(f"Readiness: {readiness.status}")
    if readiness.reasons:
        print(f"Readiness details: {', '.join(readiness.reasons)}")
    if view.manifest.commands:
        print(f"Commands: {', '.join(view.manifest.commands)}")
    if view.manifest.install_command:
        print(f"Install: {view.manifest.install_command}")
    if view.manifest.tags:
        print(f"Tags: {', '.join(view.manifest.tags)}")
    print(f"Summary: {view.manifest.summary}")

    declarations = skill_catalog.config_declarations(skill_id)
    if declarations:
        print("\nConfig:")
        for declaration in declarations:
            default = (
                f" (default: {declaration.default})"
                if declaration.default is not None
                else ""
            )
            print(f"  {declaration.key}: {declaration.description}{default}")

    if view.linked_files:
        print("\nLinked files:")
        for linked_file in view.linked_files:
            print(
                f"  {linked_file.path} "
                f"({linked_file.kind}, {linked_file.size_bytes} bytes)"
            )
    else:
        print("\nLinked files: none")
    print("\nUse /skill <name> <linked-file> to print a linked file.")


def _handle_skills_lifecycle_command(
    command: str,
    *,
    config: Optional[Config],
    skill_catalog: Optional[SkillCatalog],
    registry: ToolRegistry,
    bundle_catalog: Optional[SkillBundleCatalog] = None,
) -> bool:
    parts = command.strip().split()
    if len(parts) < 2 or parts[0].lower() != "/skills":
        return False
    subcommand = parts[1].lower()
    if subcommand not in {"inspect", "install", "uninstall", "check"}:
        return False

    target_dir = _cli_skill_default_dir(config)
    try:
        if subcommand == "inspect":
            if len(parts) < 3:
                print("Usage: /skills inspect <path-or-skill-md-url>")
                return True
            preview = inspect_skill_source(" ".join(parts[2:]))
            print(f"Skill: {preview.name} ({preview.slug}@{preview.version})")
            print(f"Summary: {preview.summary}")
            print(f"Source: {preview.source}")
            print(f"Support files: {'yes' if preview.has_support_files else 'no'}")
            return True

        if subcommand == "install":
            if len(parts) < 3:
                print("Usage: /skills install <path-or-skill-md-url> [--force]")
                return True
            force = "--force" in parts[2:]
            source = " ".join(part for part in parts[2:] if part != "--force")
            result = install_skill_source(source, target_dir, overwrite=force)
            action = "Reinstalled" if result.replaced else "Installed"
            print(
                f"{action} {result.name} "
                f"({result.slug}@{result.version}) -> {result.installed_path}"
            )
            print("Run /reload-skills to use the updated catalog.")
            return True

        if subcommand == "uninstall":
            if len(parts) != 3:
                print("Usage: /skills uninstall <skill-slug>")
                return True
            removed = uninstall_skill_from_directory(parts[2], target_dir)
            print(f"Uninstalled {parts[2]} from {removed}")
            print("Run /reload-skills to use the updated catalog.")
            return True

        if skill_catalog is None:
            print("No local skills configured.")
            return True
        print(
            f"Skills directory: {target_dir}\n"
            f"Installed skills: {len(skill_catalog.manifests())}\n"
            f"Skill bundles: {len(bundle_catalog.bundles()) if bundle_catalog else 0}"
        )
        _print_skill_catalog(skill_catalog, registry, bundle_catalog=bundle_catalog)
        return True
    except Exception as e:
        print(f"Skill {subcommand} error: {e}")
        return True


def _print_evolution_help() -> None:
    print(
        "Usage:\n"
        "  /evolution list\n"
        "  /evolution show <proposal-id>\n"
        "  /evolution propose <proposal.json>\n"
        "  /evolution eval <proposal-id> <results.json>\n"
        "  /evolution eval-local <proposal-id> [evaluator]\n"
        "  /evolution approve <proposal-id> [reviewer]\n"
        "  /evolution reject <proposal-id> [reviewer]\n"
        "  /evolution activate <proposal-id> [reviewer]\n"
        "  /evolution active <target>\n"
        "  /evolution rollback <target> [reviewer]"
    )


def _print_evolution_proposal(proposal: EvolutionProposal) -> None:
    review = proposal.review_summary()
    print(f"\n=== Evolution Proposal: {proposal.proposal_id} ===")
    print(f"Status: {proposal.status.value}")
    print(f"Target: {proposal.target}")
    print(f"Artifact: {proposal.artifact_kind.value}")
    print(
        f"Version: {proposal.current_version or '(none)'} -> {proposal.proposed_version}"
    )
    print(f"Title: {proposal.title}")
    print(f"Summary: {proposal.summary}")
    print(f"Fingerprint: {proposal.content_fingerprint}")
    print(f"Eval cases: {review.eval_case_count}")
    print(f"Eval results: {review.eval_result_count}")
    print(f"Passed evals: {review.passed_eval_count}/{review.eval_case_count}")
    if review.missing_eval_cases:
        print(f"Missing evals: {', '.join(review.missing_eval_cases)}")
    else:
        print("Missing evals: none")
    if proposal.approved_by:
        print(f"Reviewer: {proposal.approved_by}")


def _path_from_cli(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _handle_evolution_command(
    command: str,
    repository: Optional[FileEvolutionRepository],
) -> None:
    if repository is None:
        print("Evolution governance is disabled.")
        return
    parts = command.strip().split()
    if len(parts) < 2:
        _print_evolution_help()
        return
    subcmd = parts[1].lower()
    try:
        if subcmd == "list":
            proposals = repository.list()
            if not proposals:
                print("No evolution proposals.")
                return
            print("\n=== Evolution Proposals ===")
            for proposal in proposals:
                review = proposal.review_summary()
                print(
                    f"  {proposal.proposal_id} [{proposal.status.value}] "
                    f"{proposal.target} -> {proposal.proposed_version} "
                    f"evals={review.passed_eval_count}/{review.eval_case_count}: "
                    f"{proposal.title}"
                )
        elif subcmd == "show" and len(parts) >= 3:
            _print_evolution_proposal(repository.get(parts[2]))
        elif subcmd == "propose" and len(parts) >= 3:
            proposal = EvolutionProposal(
                **json.loads(_path_from_cli(parts[2]).read_text(encoding="utf-8"))
            )
            saved = repository.submit(proposal)
            print(f"Submitted proposal {saved.proposal_id}.")
        elif subcmd == "eval" and len(parts) >= 4:
            raw = json.loads(_path_from_cli(parts[3]).read_text(encoding="utf-8"))
            raw_results = raw.get("results", raw) if isinstance(raw, dict) else raw
            results = tuple(EvolutionEvalResult(**item) for item in raw_results)
            updated = repository.record_evaluations(parts[2], results)
            print(
                f"Recorded {len(updated.eval_results)} eval result(s) "
                f"for {updated.proposal_id}."
            )
        elif subcmd == "eval-local" and len(parts) >= 3:
            evaluator = parts[3] if len(parts) >= 4 else "local-eval-runner"
            proposal = repository.get(parts[2])
            results = run_local_evolution_evals(proposal, evaluator=evaluator)
            updated = repository.record_evaluations(parts[2], results)
            review = updated.review_summary()
            print(
                f"Recorded {len(updated.eval_results)} local eval result(s) "
                f"for {updated.proposal_id}; "
                f"passed {review.passed_eval_count}/{review.eval_case_count}."
            )
        elif subcmd == "approve" and len(parts) >= 3:
            reviewer = parts[3] if len(parts) >= 4 else "local-reviewer"
            approved = repository.approve(parts[2], reviewer=reviewer)
            print(f"Approved proposal {approved.proposal_id}.")
        elif subcmd == "reject" and len(parts) >= 3:
            reviewer = parts[3] if len(parts) >= 4 else "local-reviewer"
            rejected = repository.reject(parts[2], reviewer=reviewer)
            print(f"Rejected proposal {rejected.proposal_id}.")
        elif subcmd == "activate" and len(parts) >= 3:
            reviewer = parts[3] if len(parts) >= 4 else "local-reviewer"
            activated = repository.activate(parts[2], activated_by=reviewer)
            print(
                f"Activated {activated.target} at version {activated.version} "
                f"from {activated.proposal_id}."
            )
        elif subcmd == "active" and len(parts) >= 3:
            active_activation = repository.active(parts[2])
            if active_activation is None:
                print(f"No active evolution proposal for {parts[2]}.")
            else:
                print(
                    f"{active_activation.target}: {active_activation.version} "
                    f"({active_activation.proposal_id})"
                )
        elif subcmd == "rollback" and len(parts) >= 3:
            reviewer = parts[3] if len(parts) >= 4 else "local-reviewer"
            restored = repository.rollback(parts[2], rolled_back_by=reviewer)
            if restored is None:
                print(f"Rolled back {parts[2]}; no prior active version remains.")
            else:
                print(
                    f"Rolled back {parts[2]}; restored {restored.version} "
                    f"from {restored.proposal_id}."
                )
        else:
            _print_evolution_help()
    except Exception as e:
        print(f"Evolution error: {e}")


def _resolve_skill_command(
    skill_catalog: Optional[SkillCatalog],
    command: str,
) -> Optional[Tuple[str, str]]:
    if skill_catalog is None:
        return None
    parts = command.strip().split(maxsplit=1)
    if not parts:
        return None
    requested = parts[0][1:].lower()
    task = parts[1].strip() if len(parts) > 1 else ""
    for manifest in skill_catalog.manifests():
        if requested in {manifest.slug, manifest.skill_id.lower()}:
            return manifest.skill_id, task
    return None


def _resolve_skill_bundle_command(
    skill_catalog: Optional[SkillCatalog],
    bundle_catalog: Optional[SkillBundleCatalog],
    command: str,
) -> Optional[Tuple[List[str], str]]:
    if skill_catalog is None or bundle_catalog is None:
        return None
    parts = command.strip().split(maxsplit=1)
    if not parts:
        return None
    requested = parts[0][1:].lower().replace("_", "-")
    task = parts[1].strip() if len(parts) > 1 else ""
    try:
        skill_ids = list(bundle_catalog.resolve_skill_ids(requested, skill_catalog))
    except ValueError:
        return None
    return skill_ids, task


def _resolve_auto_skill_selection(
    skill_catalog: Optional[SkillCatalog],
    registry: ToolRegistry,
    task: str,
    *,
    skill_limit: int,
) -> Tuple[str, ...]:
    """Natural language does not activate skills before the agent runs.

    Hermes keeps skill choice model-mediated: slash commands explicitly select
    skills, while normal tasks receive skill metadata and can load instructions
    with skill_view when the task actually needs them.
    """
    return ()


def _missing_skill_argument_message(
    skill_catalog: Optional[SkillCatalog],
    selected_skill_ids: Optional[List[str]],
    task: str,
) -> Optional[str]:
    if skill_catalog is None or not selected_skill_ids:
        return None
    invocations = [
        skill_catalog.build_invocation(skill_id, task)
        for skill_id in selected_skill_ids
    ]
    missing = [item for item in invocations if item.missing_argument]
    if not missing:
        return None
    prompts = []
    for invocation in missing:
        hint = invocation.argument_hint or "What should this skill work on?"
        prompts.append(f"{invocation.skill_id}: {hint}")
    return "Skill needs more detail before it can run:\n  " + "\n  ".join(prompts)


def _resume_pending_skill_task(base_task: Optional[str], detail: str) -> str:
    base = base_task or ""
    clean_detail = detail.strip()
    if not base:
        return clean_detail
    replaced = re.sub(
        r"\b(something|anything|a thing|something new|stuff|whatever)\b",
        clean_detail,
        base,
        count=1,
        flags=re.IGNORECASE,
    )
    if replaced != base:
        return " ".join(replaced.split())
    return f"{base} {clean_detail}".strip()


def _handle_cli_command(
    command: str,
    memory: AgentMemory,
    *,
    config: Optional[Config] = None,
    skill_catalog: Optional[SkillCatalog] = None,
    bundle_catalog: Optional[SkillBundleCatalog] = None,
    registry: Optional[ToolRegistry] = None,
    evolution_repository: Optional[FileEvolutionRepository] = None,
) -> None:
    """Handle slash commands like /memory, /memory clear, etc."""
    parts = command.strip().split()
    cmd = parts[0].lower()

    if cmd in {"/new", "/reset", "/clear"}:
        # Start a fresh conversation: drop transient per-task state (messages,
        # tool-call history, open task records) so the next task doesn't
        # inherit prior-task context. Keep durable memory intact: experiences,
        # archival passages, core memory blocks, summaries.
        msg_count = len(memory.messages)
        tool_count = len(memory.tool_call_history)
        memory.messages.clear()
        memory.tool_call_history.clear()
        memory.tasks.clear()
        memory.current_task_id = None
        memory.state.clear()
        memory._persist()
        print(
            f"Cleared conversation ({msg_count} messages, "
            f"{tool_count} tool calls). Durable memory kept."
        )
        return

    if cmd == "/memory":
        subcmd = parts[1].lower() if len(parts) > 1 else "list"

        if subcmd == "list":
            print("\n=== Memory Overview ===")

            # Messages
            print(f"\nMessages: {len(memory.messages)}")
            for msg in memory.messages[-10:]:
                preview = msg.content[:80].replace("\n", " ")
                print(f"  [{msg.role}] {preview}...")

            # Experiences
            print(f"\nExperiences: {len(memory.experiences)}")
            for exp in memory.experiences:
                text = str(exp.get("text", ""))[:80].replace("\n", " ")
                imp = exp.get("importance", 1)
                print(f"  [imp={imp}] {text}")

            # Tasks
            print(f"\nTasks: {len(memory.tasks)}")
            for tid, task in memory.tasks.items():
                print(f"  [{task.status}] {tid}: {task.description[:60]}")

            # Tool calls
            print(f"\nTool calls: {len(memory.tool_call_history)}")

            # Core memory blocks
            if getattr(memory, "core_memory", None):
                print("\nCore memory blocks:")
                for label, block in memory.core_memory._blocks.items():
                    lines = block.value.strip()
                    if lines:
                        print(f"  [{label}] ({len(lines)} chars) {lines[:80]}...")
                    else:
                        print(f"  [{label}] (empty)")

            # Archival passages
            print(f"\nArchival passages: {len(memory.archival_passages)}")
            for rec in memory.archival_passages:
                text = str(rec.get("text", ""))[:80].replace("\n", " ")
                tags = rec.get("tags", [])
                print(f"  {text}" + (f"  tags={tags}" if tags else ""))

            print(f"\nTyped memory records: {len(memory.memory_records)}")
            for record in memory.memory_records:
                preview = record.content[:80].replace("\n", " ")
                print(
                    f"  [{record.status.value}] [{record.kind.value}] "
                    f"[{record.scope.namespace}] {preview}"
                )

        elif subcmd == "clear":
            target = parts[2].lower() if len(parts) > 2 else "all"
            if target == "all":
                memory.messages.clear()
                memory.experiences.clear()
                memory.tasks.clear()
                memory.tool_call_history.clear()
                memory.summaries.clear()
                memory.archival_passages.clear()
                memory.memory_records.clear()
                memory.state.clear()
                if getattr(memory, "core_memory", None):
                    for block in memory.core_memory._blocks.values():
                        block.value = ""
                memory._persist()
                print("All memory cleared.")
            elif target == "messages":
                memory.messages.clear()
                memory._persist()
                print("Messages cleared.")
            elif target == "experiences":
                memory.experiences.clear()
                memory.memory_records = [
                    record
                    for record in memory.memory_records
                    if record.metadata.get("legacy_collection") != "experience"
                ]
                memory._persist()
                print("Experiences cleared.")
            elif target == "tasks":
                memory.tasks.clear()
                memory._persist()
                print("Tasks cleared.")
            elif target == "archival":
                memory.archival_passages.clear()
                memory.memory_records = [
                    record
                    for record in memory.memory_records
                    if record.metadata.get("legacy_collection") != "archival"
                ]
                memory._persist()
                print("Archival passages cleared.")
            else:
                print(f"Unknown clear target: {target}")
                print("Usage: /memory clear [all|messages|experiences|tasks|archival]")

        else:
            print(f"Unknown subcommand: {subcmd}")
            print(
                "Usage: /memory [list|clear [all|messages|experiences|tasks|archival]]"
            )

    elif cmd == "/skills":
        if registry is None:
            print("No tool registry available.")
        elif _handle_skills_lifecycle_command(
            command,
            config=config,
            skill_catalog=skill_catalog,
            registry=registry,
            bundle_catalog=bundle_catalog,
        ):
            pass
        else:
            query = command.split(maxsplit=1)[1] if len(parts) > 1 else ""
            _print_skill_catalog(skill_catalog, registry, query, bundle_catalog)

    elif cmd == "/skill":
        if registry is None:
            print("No tool registry available.")
        elif len(parts) < 2:
            print("Usage: /skill <name-or-id> [linked-file]")
        else:
            file_path = parts[2] if len(parts) > 2 else None
            _print_skill_detail(skill_catalog, registry, parts[1], file_path)

    elif cmd == "/evolution":
        _handle_evolution_command(command, evolution_repository)

    else:
        print(f"Unknown command: {cmd}")
        print(
            "Available commands: /memory, /model, /new, /skills, "
            "/skill, /evolution, /reload-skills"
        )


# Small curated per-provider lists used by /model. Users can also type a
# raw model id, so these don't need to be exhaustive.
_PROVIDER_MODELS: dict[str, list[str]] = {
    "anthropic": [
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5",
        "claude-opus-4-5",
        "claude-3-5-sonnet-20241022",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-5-mini",
    ],
    "google": [
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ],
}

_PROVIDER_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def _switch_model_interactive(current: "LLMConfig"):
    """Two-step picker: choose provider, then model. Rebuilds LLM via create_llm().

    Returns (llm, new_llm_config) on success, or (None, None) if the user
    cancels or an error occurs.
    """
    from .config import LLMConfig, create_llm
    from .llm import pick_openrouter_model

    providers = list(_PROVIDER_ENV_KEYS.keys())

    print("\n=== Select a provider ===", flush=True)
    for i, name in enumerate(providers, start=1):
        has_key = bool(os.getenv(_PROVIDER_ENV_KEYS[name]))
        tag = "" if has_key else "  (API key not set)"
        marker = " *" if name == current.provider else "  "
        print(f"  {i}.{marker}{name}{tag}")
    print(f"\n  Current: {current.provider} / {current.model}")
    print("  Press Enter to cancel.\n", flush=True)

    try:
        raw = input("  > ").strip().lower()
    except EOFError:
        return None, None
    if not raw:
        return None, None

    if raw.isdigit() and 1 <= int(raw) <= len(providers):
        provider = providers[int(raw) - 1]
    elif raw in providers:
        provider = raw
    else:
        print(f"  Unknown provider: {raw}")
        return None, None

    if not os.getenv(_PROVIDER_ENV_KEYS[provider]):
        print(
            f"  {_PROVIDER_ENV_KEYS[provider]} is not set; "
            f"export it before switching to {provider}."
        )
        return None, None

    # Choose model for the selected provider
    if provider == "openrouter":
        default = current.model if current.provider == "openrouter" else None
        model = pick_openrouter_model(default_slug=default)
    else:
        options = _PROVIDER_MODELS.get(provider, [])
        print(f"\nSelect a {provider} model:")
        for i, m in enumerate(options, start=1):
            print(f"  {i}. {m}")
        print("\n  Type a number, paste any model id, or press Enter to cancel.")
        try:
            raw_m = input("  > ").strip()
        except EOFError:
            return None, None
        if not raw_m:
            return None, None
        if raw_m.isdigit() and 1 <= int(raw_m) <= len(options):
            model = options[int(raw_m) - 1]
        else:
            model = raw_m

    new_cfg = LLMConfig(
        provider=cast(Any, provider),
        model=model,
        temperature=current.temperature,
        max_tokens=current.max_tokens,
        top_p=current.top_p,
        extra_params=dict(current.extra_params),
    )
    try:
        new_llm = create_llm(new_cfg)
    except Exception as e:
        print(f"  Failed to build LLM: {e}")
        return None, None
    return new_llm, new_cfg


async def main():
    logger.info("Starting Pori Agent System")

    # Set up the tool registry
    logger.info("Initializing tool registry")
    registry = tool_registry()

    register_all_tools(registry)
    # Register sandbox tools (bash) so they appear in the same registry
    import pori.sandbox.sandbox_tools  # noqa: F401

    logger.info(f"Registered {len(registry.tools)} tools")

    # Create LLM from config file
    logger.info("Loading LLM configuration")
    try:
        llm, config = get_configured_llm()
        logger.info(
            f"Initialized LLM - Provider: {config.llm.provider}, Model: {config.llm.model}"
        )
    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        print(f"\nError: {e}")
        print("Please create a config.yaml file based on config.example.yaml")
        return
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        print(f"\nError initializing LLM: {e}")
        return

    # Prompts: allow overriding packaged prompt templates
    try:
        if (
            getattr(config, "prompts", None)
            and config.prompts
            and config.prompts.base_dir
        ):
            set_prompts_dir(config.prompts.base_dir)
            logger.info(f"Prompts override enabled; base_dir={config.prompts.base_dir}")
    except Exception:
        pass

    # Sandbox: if enabled, set provider and base dir for per-task workspace
    sandbox_base_dir = None
    if (
        getattr(config, "sandbox", None)
        and config.sandbox.enabled
        and config.sandbox.base_dir
    ):
        from pori.sandbox import LocalSandboxProvider, set_sandbox_provider

        set_sandbox_provider(LocalSandboxProvider())
        sandbox_base_dir = str(Path(config.sandbox.base_dir).resolve())
        logger.info(f"Sandbox enabled; base_dir={sandbox_base_dir}")

    memory_backend = (
        getattr(config, "memory", None).backend
        if getattr(config, "memory", None)
        else "memory"
    )
    memory_sqlite_path = (
        getattr(config, "memory", None).sqlite_path
        if getattr(config, "memory", None)
        else None
    )
    memory_store = create_memory_store(
        backend=memory_backend,
        sqlite_path=memory_sqlite_path,
    )
    shared_memory = AgentMemory(
        organization_id=getattr(config.memory, "organization_id", "default_org"),
        user_id=getattr(config.memory, "user_id", "default_user"),
        agent_id=getattr(config.memory, "agent_id", "default_agent"),
        session_id=getattr(config.memory, "session_id", None),
        store=memory_store,
    )

    skill_catalog = _load_cli_skill_catalog(config)
    bundle_catalog = _load_cli_skill_bundles(config)
    skill_limit = getattr(getattr(config, "skills", None), "skill_limit", 3)
    if skill_catalog is not None:
        logger.info(f"Loaded {len(skill_catalog.manifests())} local skill(s)")
    if bundle_catalog is not None and bundle_catalog.bundles():
        logger.info(f"Loaded {len(bundle_catalog.bundles())} skill bundle(s)")

    evolution_repository = _load_cli_evolution_repository(config)
    if evolution_repository is not None:
        logger.info(f"Evolution repository path: {evolution_repository.path}")

    # Create orchestrator
    logger.info("Creating orchestrator")
    orchestrator = Orchestrator(
        llm=llm,
        tools_registry=registry,
        shared_memory=shared_memory,
        skill_catalog=skill_catalog,
        skill_limit=skill_limit,
        evolution_repository=evolution_repository,
        soul_path=config.agent.soul_path,
        load_project_context=config.agent.load_project_context,
    )

    # HITL: check if enabled in config
    hitl_handler = None
    hitl_config = getattr(config, "hitl", None)
    if hitl_config and hitl_config.enabled:
        hitl_handler = CLIHITLHandler(timeout_seconds=hitl_config.timeout_seconds)
        logger.info("HITL enabled in CLI mode")

    # Team: check if configured
    team_config = getattr(config, "team", None)
    use_team = team_config is not None and len(team_config.members) > 0
    if use_team:
        logger.info(
            f"Team mode enabled: '{team_config.name}' ({team_config.mode.value}) "
            f"with {len(team_config.members)} members"
        )
        print(
            f"(Team mode: {team_config.name} [{team_config.mode.value}] "
            f"with {len(team_config.members)} members)"
        )

    # Define step lifecycle callbacks for monitoring. These fire via the
    # on_step_start/on_step_end hooks now wired into Agent.run() — proof that
    # the agent emits real per-step events (not polling).
    # Live-streaming state: tracks whether the current step has streamed text so
    # the completion line stays on its own row and we don't reprint the activity.
    _stream_state = {"active": False}

    def on_text_delta(chunk: str) -> None:
        if not chunk:
            return
        _stream_state["active"] = True
        try:
            sys.stdout.write(_console_safe_text(chunk))
            sys.stdout.flush()
        except Exception:
            pass

    def on_step_start(agent: Agent):
        # No loop-counter noise ("Step N"). Progress is shown as streamed text
        # (when enabled) and one clean action line per step, below.
        pass

    def on_step_end(agent: Agent):
        # If this step streamed text live, that IS the feedback — just close the
        # line and move on.
        if _stream_state["active"]:
            print(flush=True)
            _stream_state["active"] = False
            return
        # Otherwise show one clean action line describing what the step did,
        # e.g. "• Writing hi.py", "• Running python hi.py", "• Writing the answer".
        try:
            history = agent.memory.tool_call_history
            if not history:
                return
            tc = history[-1]
            preview = build_tool_preview(
                tc.tool_name, getattr(tc, "parameters", {}) or {}
            )
            mark = "" if getattr(tc, "success", True) else "  (failed)"
            _safe_print(f"  • {preview}{mark}")
        except Exception:
            pass
        logger.info(
            f"Completed step {agent.state.n_steps}",
            extra={
                "task_id": getattr(agent, "task_id", "unknown"),
                "step": agent.state.n_steps,
            },
        )

    logger.info("Starting interactive loop")
    if memory_backend == "memory":
        print("(Memory backend: in-memory; exiting clears session memory.)")
    else:
        print(
            f"(Memory backend: {memory_backend}; session namespace={shared_memory.namespace})"
        )

    pending_skill_ids: Optional[List[str]] = None
    pending_skill_task: Optional[str] = None

    # Interactive loop for tasks
    while True:
        print("\n Pori at your service!")
        try:
            if pending_skill_ids:
                prompt = "Skill detail needed. Enter detail, /cancel, or q to exit \n"
            else:
                prompt = "How can I help you today? enter q to exit \n"
            task = input(prompt).strip()
        except EOFError:
            logger.info("Input closed (EOF)")
            print("\nGoodbye!")
            break

        # Exit if the user provides no task
        if task == "q":
            logger.info("User requested exit")
            print("Goodbye!")
            break

        if task.lower() == "/cancel" and pending_skill_ids:
            pending_skill_ids = None
            pending_skill_task = None
            print("Cancelled pending skill invocation.")
            continue

        if not task:
            logger.warning("Empty task provided, skipping")
            continue

        selected_skill_ids = pending_skill_ids
        if pending_skill_ids:
            task = _resume_pending_skill_task(pending_skill_task, task)
            pending_skill_ids = None
            pending_skill_task = None

        # CLI commands
        if task.startswith("/") and selected_skill_ids is None:
            command_name = task.strip().split()[0].lower()
            if command_name == "/model":
                import sys as _sys

                _sys.stdout.flush()
                new_llm, new_llm_config = _switch_model_interactive(config.llm)
                _sys.stdout.flush()
                if new_llm is not None and new_llm_config is not None:
                    llm = new_llm
                    config.llm = new_llm_config
                    orchestrator.llm = llm
                    print(
                        "\n=== Model switched ===\n"
                        f"  Provider: {new_llm_config.provider}\n"
                        f"  Model:    {new_llm_config.model}\n"
                        "  Next task will use this model.\n"
                        "  Tip: type /new to start a fresh conversation "
                        "(drops prior-task context).\n",
                        flush=True,
                    )
                else:
                    print("  Model unchanged.", flush=True)
                continue

            if command_name == "/reload-skills":
                try:
                    skill_catalog = _load_cli_skill_catalog(config)
                    bundle_catalog = _load_cli_skill_bundles(config)
                    orchestrator.skill_catalog = skill_catalog
                    count = len(skill_catalog.manifests()) if skill_catalog else 0
                    bundle_count = (
                        len(bundle_catalog.bundles()) if bundle_catalog else 0
                    )
                    print(
                        f"Reloaded {count} local skill(s) and "
                        f"{bundle_count} skill bundle(s)."
                    )
                except Exception as e:
                    print(f"Failed to reload skills: {e}")
                continue

            bundle_command = _resolve_skill_bundle_command(
                skill_catalog, bundle_catalog, task
            )
            if bundle_command is not None:
                selected_skill_ids, task = bundle_command
                if not task:
                    print("Provide a task after the skill bundle command.")
                    continue
                print(f"Using skills: {', '.join(selected_skill_ids)}")
            else:
                skill_command = _resolve_skill_command(skill_catalog, task)
            if selected_skill_ids is None and skill_command is not None:
                skill_id, task = skill_command
                if not task:
                    print(f"Provide a task after /{skill_id.split('@', 1)[0]}.")
                    continue
                selected_skill_ids = [skill_id]
                print(f"Using skill: {skill_id}")
            else:
                _handle_cli_command(
                    task,
                    shared_memory,
                    config=config,
                    skill_catalog=skill_catalog,
                    bundle_catalog=bundle_catalog,
                    registry=registry,
                    evolution_repository=evolution_repository,
                )
                continue

        if selected_skill_ids is None:
            selected_skill_ids = list(
                _resolve_auto_skill_selection(
                    skill_catalog,
                    registry,
                    task,
                    skill_limit=skill_limit,
                )
            )
            if selected_skill_ids:
                print(f"Using skill: {', '.join(selected_skill_ids)}")

        missing_skill_argument = _missing_skill_argument_message(
            skill_catalog,
            selected_skill_ids,
            task,
        )
        if missing_skill_argument:
            print(f"\n{missing_skill_argument}")
            pending_skill_ids = selected_skill_ids
            pending_skill_task = task
            continue

        # Inline any @path file references the user typed (e.g. @main.py).
        expanded_task, ref_notes = expand_file_refs(task)
        if ref_notes:
            print("\nAttached files:")
            for note in ref_notes:
                print(note)
            logger.info(f"Expanded {len(ref_notes)} @path reference(s)")
        task = expanded_task

        logger.info("New task received")

        try:
            if use_team:
                if selected_skill_ids:
                    print("Explicit skill commands run in single-agent mode.")
                    continue
                # Execute via Team
                logger.info("Starting team execution")
                agent_defaults = AgentSettings(
                    max_steps=config.agent.max_steps,
                    planning_mode=config.agent.planning_mode,
                    reflection_mode=config.agent.reflection_mode,
                    context_window_tokens=config.agent.context_window_tokens,
                    context_window_reserve_tokens=config.agent.context_window_reserve_tokens,
                    validate_output=config.agent.validate_output,
                    max_validation_retries=config.agent.max_validation_retries,
                )
                team = Team(
                    task=task,
                    coordinator_llm=llm,
                    members=team_config.members,
                    mode=team_config.mode,
                    tools_registry=registry,
                    hitl_handler=hitl_handler,
                    hitl_config=hitl_config,
                    agent_defaults=agent_defaults,
                    max_delegation_steps=team_config.max_delegation_steps,
                    max_concurrent_members=team_config.max_concurrent_members,
                    name=team_config.name,
                )
                result = await team.run()

                logger.info(
                    f"Team execution completed - Success: {result['completed']}, Steps: {result['steps_taken']}"
                )

                print("\n=== Task Execution Summary ===")
                print(f"Task: {task}")
                print(f"Completed: {result['completed']}")
                print(f"Steps taken: {result['steps_taken']}")

                fs = result.get("final_state", {})
                if "plan_steps" in fs:
                    print(
                        f"Plan steps: {fs['plan_steps']} | Agent steps: {fs['agent_steps']}"
                    )
                if "chosen_member" in fs:
                    print(f"Routed to: {fs['chosen_member']}")

                final_answer = fs.get("final_answer")
                if final_answer:
                    _safe_print(f"\nFINAL ANSWER:\n  {final_answer}")
                else:
                    print("\nNO FINAL ANSWER FOUND")

            else:
                # Execute via single-agent Orchestrator
                logger.info("Starting task execution")
                result = await orchestrator.execute_task(
                    task=task,
                    agent_settings=AgentSettings(
                        max_steps=config.agent.max_steps,
                        planning_mode=config.agent.planning_mode,
                        reflection_mode=config.agent.reflection_mode,
                        context_window_tokens=config.agent.context_window_tokens,
                        context_window_reserve_tokens=config.agent.context_window_reserve_tokens,
                        validate_output=config.agent.validate_output,
                        max_validation_retries=config.agent.max_validation_retries,
                    ),
                    on_step_start=on_step_start,
                    on_step_end=on_step_end,
                    on_text_delta=(on_text_delta if config.llm.streaming else None),
                    sandbox_base_dir=sandbox_base_dir,
                    hitl_handler=hitl_handler,
                    hitl_config=hitl_config,
                    selected_skill_ids=selected_skill_ids,
                )

                logger.info(
                    f"Task execution completed - Success: {result['success']}, Steps: {result['steps_taken']}"
                )

                print("\n=== Task Execution Summary ===")
                print(f"Task: {task}")
                print(f"Success: {result['success']}")
                print(f"Steps taken: {result['steps_taken']}")
                selected_skills = result.get("selected_skills") or []
                if selected_skills:
                    print(f"Skills used: {', '.join(selected_skills)}")

                # Show basic run metrics if available
                metrics = result.get("result", {}).get("metrics")
                if metrics:
                    try:
                        tokens = metrics.get("tokens", {}) or {}
                        cost = metrics.get("cost_usd")
                        print(
                            f"Run metrics: duration={metrics.get('duration')}, "
                            f"steps={metrics.get('steps')}, "
                            f"llm_calls={metrics.get('llm_calls')}, "
                            f"tool_calls={metrics.get('tool_calls')}, "
                            f"tokens_in={tokens.get('input')}, "
                            f"tokens_out={tokens.get('output')}, "
                            f"tokens_total={tokens.get('total')}"
                        )
                        if cost is not None:
                            print(f"Estimated cost: {cost}")
                    except Exception:
                        pass

                # Show the final answer if available
                agent = result.get("agent")

                if agent:
                    final_answer = agent.memory.get_final_answer()

                    if final_answer:
                        logger.info("Final answer provided")
                        print("\nFINAL ANSWER:")
                        _safe_print(f"  {final_answer['final_answer']}")
                        if final_answer.get("reasoning"):
                            _safe_print(f"\n  Reasoning: {final_answer['reasoning']}")
                        print(
                            "\n(Memory recall used if 'Retrieved knowledge' logs appear above for this task.)"
                        )
                    else:
                        logger.warning("No final answer found")
                        print("\nNO FINAL ANSWER FOUND")

                    calls_this_task = [
                        tc
                        for tc in agent.memory.tool_call_history
                        if getattr(tc, "task_id", None) == agent.task_id
                    ]
                    tool_calls_count = len(calls_this_task)
                    logger.info(f"Tool calls made (this task): {tool_calls_count}")

                    artifact_lines = _summarize_written_artifacts(calls_this_task)
                    if artifact_lines:
                        print("\nArtifacts written (this task):")
                        for line in artifact_lines:
                            _safe_print(f"  - {line}")

                    loaded_skills = _summarize_loaded_skills(calls_this_task)
                    if loaded_skills:
                        print(f"\nSkills loaded: {', '.join(loaded_skills)}")

                    plan_items = getattr(
                        getattr(agent, "plan_store", None), "items", lambda: ()
                    )()
                    if plan_items:
                        marks = {
                            "completed": "[x]",
                            "in_progress": "[>]",
                            "pending": "[ ]",
                            "cancelled": "[~]",
                        }
                        print("\nPlan (this task):")
                        for item in plan_items:
                            _safe_print(
                                f"  {marks.get(item.status, '[ ]')} {item.content}"
                            )

                    print("\nTool Calls (this task):")
                    for i, tool_call in enumerate(calls_this_task, start=1):
                        status = "+" if tool_call.success else "x"
                        preview = build_tool_preview(
                            tool_call.tool_name, tool_call.parameters
                        )
                        _safe_print(f"  {i}. {preview} -> {status}")
                        _safe_print(
                            f"       {tool_call.tool_name}("
                            f"{_format_tool_call_parameters(tool_call.parameters)})"
                        )

                        log_level = (
                            logging.INFO if tool_call.success else logging.WARNING
                        )
                        logger.log(
                            log_level,
                            f"Tool call: {tool_call.tool_name} - {'Success' if tool_call.success else 'Failed'}",
                        )

        except Exception as e:
            logger.error(f"Error executing task: {e}", exc_info=True)
            _safe_print(f"Error executing task: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise
