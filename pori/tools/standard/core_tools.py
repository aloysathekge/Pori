"""
Core tools required for agent operation.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field

from pori.clarify import resolve_clarify_handler
from pori.evolution import EvolutionArtifactKind, EvolutionEvalCase, EvolutionProposal
from pori.skill_provenance import is_agent_origin, mark_agent_created
from pori.subagents import MAX_CONCURRENT_CHILDREN
from pori.threat_patterns import first_threat_message

from ..registry import tool_registry

Registry = tool_registry()
logger = logging.getLogger("pori.core_tools")


class AnswerArtifactReference(BaseModel):
    path: str | None = Field(
        default=None,
        description=(
            "Path of an artifact actually written during this run. Provide this "
            "and/or receipt_id (at least one); both are preferred."
        ),
    )
    receipt_id: str | None = Field(
        default=None,
        description="Runtime receipt id proving the artifact was written.",
    )
    description: str | None = Field(
        default=None,
        description="Optional short description of what the artifact is.",
    )


class AnswerParams(BaseModel):
    final_answer: str = Field(
        ..., description="The final answer to the user's question"
    )
    reasoning: str = Field(
        ..., description="Brief explanation of how you arrived at this answer"
    )
    artifact_references: list[AnswerArtifactReference] = Field(
        default_factory=list,
        description=(
            "UI-visible artifacts referenced by this answer. Only include "
            "artifacts listed in Runtime Facts for this run."
        ),
    )


@Registry.tool(
    name="answer",
    description="REQUIRED: Provide your final answer to the user's question with reasoning.",
)
def answer_tool(params: AnswerParams, context: Dict[str, Any]) -> Dict[str, Any]:
    """Provide a final answer to the user's question."""
    answer = {
        "final_answer": params.final_answer,
        # Empty reasoning is normal for a plain prose reply; keep it empty so the
        # CLI/UI can omit a hollow "Reasoning:" line.
        "reasoning": params.reasoning or "",
        "artifact_references": [
            reference.model_dump(exclude_none=True)
            for reference in params.artifact_references
        ],
    }

    logger.info(f"Answer tool called with final answer: {params.final_answer}")

    if context and "memory" in context:
        context["memory"].update_state("final_answer", answer)
        logger.info("Final answer stored in memory")
    else:
        logger.warning("Could not store final answer - memory not available in context")

    return answer


class DoneParams(BaseModel):
    success: bool = Field(
        True, description="Whether the task was completed successfully"
    )
    message: str = Field(
        "Task completed", description="Final message about task completion"
    )


@Registry.tool(name="done", description="Mark the task as complete")
def done_tool(params: DoneParams, context: Dict[str, Any]):
    """Mark the task as done."""
    return {"final_message": params.message, "success": params.success}


class AskUserParams(BaseModel):
    question: str = Field(..., description="The clarifying question to ask the user")
    reason: str = Field(..., description="Why you need this before proceeding")
    options: List[str] = Field(
        default_factory=list,
        description=(
            "Optional fixed choices to offer as a menu/buttons; leave empty for a "
            "free-text answer"
        ),
    )


@Registry.tool(
    name="ask_user",
    description=(
        "Ask the user a clarifying question when a missing detail blocks the task. "
        "Pass 'options' to offer a fixed set of choices (rendered as a menu, or "
        "buttons on a gateway); leave options empty for a free-text answer. Use "
        "BEFORE acting on ambiguity."
    ),
)
def ask_user_tool(params: AskUserParams, context: Dict[str, Any]) -> Dict[str, Any]:
    """Ask for clarification (structured options or free text) and return the answer.

    Presentation is delegated to a clarify handler from the context (a gateway can
    render buttons); the default is a CLI numbered menu. The user is never boxed in
    — an "Other" choice always allows a free-text answer.
    """
    handler = resolve_clarify_handler(context)
    answer = handler(params.question, list(params.options))
    return {"success": True, "user_response": answer}


class ThinkParams(BaseModel):
    thoughts: str = Field(..., description="What you're thinking about")
    next_action: str = Field(..., description="What you plan to do next")


@Registry.tool(name="think", description="Record thoughts and planning (optional)")
def think_tool(params: ThinkParams, context: Dict[str, Any]):
    """Allow the agent to record its thoughts and planning."""
    return {
        "thoughts": params.thoughts,
        "next_action": params.next_action,
        "timestamp": "thinking step recorded",
    }


class RememberParams(BaseModel):
    text: str = Field(..., description="The fact or note to remember long-term")
    importance: int = Field(
        1, ge=1, le=5, description="Importance from 1 (low) to 5 (high)"
    )


@Registry.tool(
    name="remember",
    description="Store a fact or note into long-term memory with optional importance",
)
def remember_tool(params: RememberParams, context: Dict[str, Any]):
    """Persist a fact to long-term memory (and vector index if enabled)."""
    memory = context.get("memory") if context else None
    if not memory:
        return {"success": False, "error": "Memory not available in context"}

    try:
        key = memory.add_experience(
            params.text,
            importance=params.importance,
            meta={"type": "fact", "source": "user"},
        )
        return {"success": True, "result": {"key": key}}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------- Recall + Archival memory tools (Letta-style) ----------


class ConversationSearchParams(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(10, ge=1, le=50, description="Max results to return")
    roles: list[str] | None = Field(
        default=None, description="Optional roles filter (e.g. ['user','assistant'])"
    )


@Registry.tool(
    name="conversation_search",
    description="Search the conversation history for relevant prior messages.",
)
def conversation_search_tool(params: ConversationSearchParams, context: Dict[str, Any]):
    memory = context.get("memory") if context else None
    if not memory or not hasattr(memory, "conversation_search"):
        return {"success": False, "error": "Memory search not available"}
    try:
        results = memory.conversation_search(
            query=params.query, limit=params.limit, roles=params.roles
        )
        return {"success": True, "results": results, "count": len(results)}
    except Exception as e:
        return {"success": False, "error": str(e)}


class ArchivalInsertParams(BaseModel):
    text: str = Field(..., description="Passage to store in archival memory")
    tags: list[str] | None = Field(default=None, description="Optional tags")
    importance: int = Field(1, ge=1, le=5, description="Importance from 1-5")


@Registry.tool(
    name="archival_memory_insert",
    description="Store a passage in long-term archival memory for later retrieval.",
)
def archival_memory_insert_tool(params: ArchivalInsertParams, context: Dict[str, Any]):
    memory = context.get("memory") if context else None
    if not memory or not hasattr(memory, "archival_memory_insert"):
        return {"success": False, "error": "Archival memory not available"}
    try:
        pid = memory.archival_memory_insert(
            text=params.text, tags=params.tags, importance=params.importance
        )
        return {"success": True, "result": {"id": pid}}
    except Exception as e:
        return {"success": False, "error": str(e)}


class ArchivalSearchParams(BaseModel):
    query: str = Field(..., description="Search query")
    k: int = Field(5, ge=1, le=20, description="Max results to return")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Minimum score")
    tags: list[str] | None = Field(default=None, description="Optional tag filter")


@Registry.tool(
    name="archival_memory_search",
    description="Search archival memory passages by query using semantic similarity.",
)
def archival_memory_search_tool(params: ArchivalSearchParams, context: Dict[str, Any]):
    memory = context.get("memory") if context else None
    if not memory or not hasattr(memory, "archival_memory_search"):
        return {"success": False, "error": "Archival memory not available"}
    try:
        results = memory.archival_memory_search(
            query=params.query,
            k=params.k,
            min_score=params.min_score,
            tags=params.tags,
        )
        return {
            "success": True,
            "results": [
                {"id": rid, "text": text, "score": score}
                for rid, text, score in results
            ],
            "count": len(results),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------- Letta-style core memory (editable in-context blocks) ----------


class CoreMemoryReadParams(BaseModel):
    label: str | None = Field(
        default=None,
        description="Optional block to read: 'persona', 'human', or 'notes'. If omitted, returns all blocks.",
    )


@Registry.tool(
    name="core_memory_read",
    description="Read-only inspection of core memory blocks. Use this to view what is stored without modifying anything.",
)
def core_memory_read_tool(
    params: CoreMemoryReadParams, context: Dict[str, Any]
) -> Dict[str, Any]:
    """Return one or all core memory blocks without mutation."""
    memory = context.get("memory") if context else None
    if not memory or not getattr(memory, "core_memory", None):
        return {"success": False, "error": "Core memory not available"}
    try:
        if params.label:
            block = memory.core_memory.get_block(params.label)
            return {"success": True, "blocks": {params.label: block.value}}
        else:
            blocks = {
                name: memory.core_memory.get_block(name).value
                for name in ["persona", "human", "notes"]
            }
            return {"success": True, "blocks": blocks}
    except ValueError as e:
        return {"success": False, "error": str(e)}


class CoreMemoryAppendParams(BaseModel):
    label: str = Field(
        ...,
        description="Block to edit: 'persona' (your traits), 'human' (user facts), or 'notes' (scratch)",
    )
    content: str = Field(..., description="Text to append to the block")


class CoreMemoryReplaceParams(BaseModel):
    label: str = Field(
        ...,
        description="Block to edit: 'persona', 'human', or 'notes'",
    )
    old_string: str = Field(..., description="Exact text to find and replace")
    new_string: str = Field(..., description="Replacement text")


class MemoryInsertParams(BaseModel):
    label: str = Field(
        ...,
        description="Block to edit: 'persona', 'human', or 'notes'",
    )
    new_str: str = Field(..., description="Text to insert")
    insert_line: int = Field(
        -1,
        description="Line index to insert at (0-based). Use -1 to append",
    )


class MemoryRethinkParams(BaseModel):
    label: str = Field(
        ...,
        description="Block to rewrite: 'persona', 'human', or 'notes'",
    )
    new_memory: str = Field(..., description="Complete new content for the block")


@Registry.tool(
    name="core_memory_append",
    description="Append a line to a core memory block (persona, human, or notes). Always in context.",
)
def core_memory_append_tool(params: CoreMemoryAppendParams, context: Dict[str, Any]):
    """Append content to a core memory block."""
    memory = context.get("memory") if context else None
    if not memory or not getattr(memory, "core_memory", None):
        return {"success": False, "error": "Core memory not available"}
    threat = first_threat_message(params.content)
    if threat:
        return {"success": False, "error": f"Refusing to persist to memory. {threat}"}
    try:
        memory.core_memory.get_block(params.label).append(params.content)
        if hasattr(memory, "persist"):
            memory.persist()
        return {"success": True, "message": f"Appended to {params.label}"}
    except ValueError as e:
        return {"success": False, "error": str(e)}


@Registry.tool(
    name="core_memory_replace",
    description="Replace exact text in a core memory block (persona, human, or notes).",
)
def core_memory_replace_tool(params: CoreMemoryReplaceParams, context: Dict[str, Any]):
    """Replace text in a core memory block."""
    memory = context.get("memory") if context else None
    if not memory or not getattr(memory, "core_memory", None):
        return {"success": False, "error": "Core memory not available"}
    try:
        memory.core_memory.get_block(params.label).replace(
            params.old_string, params.new_string
        )
        if hasattr(memory, "persist"):
            memory.persist()
        return {"success": True, "message": f"Updated {params.label}"}
    except ValueError as e:
        return {"success": False, "error": str(e)}


@Registry.tool(
    name="memory_insert",
    description="Insert text into a core memory block at a specific line index.",
)
def memory_insert_tool(params: MemoryInsertParams, context: Dict[str, Any]):
    memory = context.get("memory") if context else None
    if not memory or not getattr(memory, "core_memory", None):
        return {"success": False, "error": "Core memory not available"}
    threat = first_threat_message(params.new_str)
    if threat:
        return {"success": False, "error": f"Refusing to persist to memory. {threat}"}
    try:
        memory.core_memory.memory_insert(
            label=params.label,
            new_str=params.new_str,
            insert_line=params.insert_line,
        )
        if hasattr(memory, "persist"):
            memory.persist()
        return {"success": True, "message": f"Inserted text into {params.label}"}
    except ValueError as e:
        return {"success": False, "error": str(e)}


@Registry.tool(
    name="memory_rethink",
    description="Mutation only: Rewrite an entire core memory block with a validated replacement. Do not use this to inspect or summarize memory; use core_memory_read instead.",
)
def memory_rethink_tool(params: MemoryRethinkParams, context: Dict[str, Any]):
    memory = context.get("memory") if context else None
    if not memory or not getattr(memory, "core_memory", None):
        return {"success": False, "error": "Core memory not available"}
    try:
        memory.core_memory.memory_rethink(
            label=params.label,
            new_memory=params.new_memory,
        )
        if hasattr(memory, "persist"):
            memory.persist()
        return {"success": True, "message": f"Rewrote {params.label}"}
    except ValueError as e:
        return {"success": False, "error": str(e)}


@Registry.tool(
    name="core_memory_rethink",
    description="Mutation only: Compatibility alias for memory_rethink to rewrite a full core memory block. Do not use this to inspect or summarize memory; use core_memory_read instead.",
)
def core_memory_rethink_tool(params: MemoryRethinkParams, context: Dict[str, Any]):
    """Backwards-compatible alias for full core memory rewrites."""
    return memory_rethink_tool(params, context)


class EvolutionProposalParams(BaseModel):
    artifact_kind: Literal["skill", "prompt", "policy", "eval", "code", "config"]
    target: str = Field(..., description="Artifact target, e.g. skills/brainstorming")
    title: str = Field(..., description="Short proposal title")
    summary: str = Field(..., description="One-paragraph summary of the change")
    rationale: str = Field(..., description="Why this change should improve Pori")
    proposed_version: str = Field(..., description="Version label for the proposal")
    proposed_content: str = Field(..., description="Proposed artifact content")
    eval_cases: list[EvolutionEvalCase] = Field(
        ..., description="Evaluation cases that must pass before approval"
    )
    current_version: str | None = Field(default=None)
    supersedes_proposal_id: str | None = Field(default=None)


@Registry.tool(
    name="propose_evolution",
    description=(
        "Create an inert governed self-evolution proposal with eval cases. "
        "This never applies, approves, or activates the change."
    ),
)
def propose_evolution_tool(
    params: EvolutionProposalParams,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    repository = context.get("evolution_repository") if context else None
    if repository is None:
        return {"success": False, "error": "Evolution repository not available"}
    try:
        proposal = EvolutionProposal(
            artifact_kind=EvolutionArtifactKind(params.artifact_kind),
            target=params.target,
            title=params.title,
            summary=params.summary,
            rationale=params.rationale,
            current_version=params.current_version,
            proposed_version=params.proposed_version,
            proposed_content=params.proposed_content,
            eval_cases=tuple(params.eval_cases),
            supersedes_proposal_id=params.supersedes_proposal_id,
        )
        saved = repository.submit(proposal)
        return {
            "success": True,
            "proposal_id": saved.proposal_id,
            "status": saved.status.value,
            "target": saved.target,
            "proposed_version": saved.proposed_version,
            "content_fingerprint": saved.content_fingerprint,
            "next_step": "Run evals, then review with /evolution approve before activation.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


_SKILL_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")


class WriteSkillParams(BaseModel):
    slug: str = Field(
        ...,
        description="Lowercase-hyphenated skill name (matches the frontmatter name).",
    )
    content: str = Field(
        ..., description="The full SKILL.md text (--- frontmatter --- then body)."
    )
    overwrite: bool = Field(
        default=False, description="Replace an existing skill with the same slug."
    )


@Registry.tool(
    name="write_skill",
    param_model=WriteSkillParams,
    description=(
        "Author/install a reusable skill: writes a validated SKILL.md into the "
        "skills directory. Use this (not write_file) to save a skill."
    ),
)
def write_skill_tool(params: WriteSkillParams, context: Dict[str, Any]):
    """Install an authored SKILL.md (SK-1 layer 1).

    Validates the frontmatter, writes to ``{skills_dir}/{slug}/SKILL.md``, and —
    when the current write-origin is an autonomous agent origin (SK-2) — records
    the skill as agent-created so the curator may later manage it.
    """
    slug = params.slug.strip().lower()
    if not _SKILL_SLUG_RE.match(slug):
        return {
            "success": False,
            "error": "slug must be lowercase-hyphenated, 3-64 chars (e.g. search-arxiv)",
        }

    text = params.content.strip()
    if not text.startswith("---"):
        return {
            "success": False,
            "error": "SKILL.md must start with a --- YAML frontmatter block",
        }
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {
            "success": False,
            "error": "SKILL.md frontmatter is not closed with ---",
        }
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as exc:
        return {"success": False, "error": f"invalid frontmatter YAML: {exc}"}
    if not isinstance(meta, dict) or not meta.get("name"):
        return {"success": False, "error": "frontmatter needs a 'name'"}
    description = str(meta.get("description") or meta.get("summary") or "")
    if not description:
        return {"success": False, "error": "frontmatter needs a 'description'"}

    warnings = []
    if len(description) > 60:
        warnings.append(
            f"description is {len(description)} chars; the skill index truncates to "
            "60, so trim it for reliable routing"
        )

    skills_dir = Path((context or {}).get("skills_dir") or "./.pori/skills")
    destination = skills_dir / slug / "SKILL.md"
    if destination.exists() and not params.overwrite:
        return {
            "success": False,
            "error": f"skill '{slug}' already exists; pass overwrite=true to replace",
        }
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    except OSError as exc:
        return {"success": False, "error": f"could not write skill: {exc}"}

    version = str(meta.get("version") or "0.1.0")
    if is_agent_origin():
        mark_agent_created(f"{slug}@{version}")

    return {
        "success": True,
        "slug": slug,
        "path": str(destination),
        "warnings": warnings,
        "message": (
            f"Saved skill '{slug}'. It is available after /reload-skills or on the "
            "next session."
        ),
    }


class DelegateTaskItem(BaseModel):
    goal: str = Field(
        ..., description="A detailed, self-contained description of the subtask."
    )
    context: Optional[str] = Field(
        None, description="Optional background/context the sub-agent needs."
    )
    role: Literal["leaf", "orchestrator"] = Field(
        "leaf",
        description="'leaf' (default) cannot delegate further; 'orchestrator' may "
        "spawn its own sub-agents for reasoning-heavy independent subtasks.",
    )


class DelegateTaskParams(BaseModel):
    tasks: List[DelegateTaskItem] = Field(
        ...,
        description="Subtasks to delegate. One runs alone; several run CONCURRENTLY "
        "(a parallel batch) — only for INDEPENDENT subtasks.",
    )
    background: bool = Field(
        False,
        description="Run in the BACKGROUND without blocking: dispatch the task(s) and "
        "get handles back immediately; results arrive later as a new turn. Use for "
        "long-running work you don't need to wait on. Ignores 'role' (children are "
        "leaves).",
    )


@Registry.tool(
    name="delegate_task",
    param_model=DelegateTaskParams,
    description=(
        "Delegate one or more subtasks to focused sub-agents, each with its OWN "
        "context window and a restricted toolset built from the goal you give. "
        "Provide SEVERAL tasks to run them concurrently (a parallel batch). Use for "
        "context-heavy or parallelizable work (research, exploration, review) — you "
        "get back only their summaries, so your context stays clean. Give each a "
        "complete, self-contained goal."
    ),
)
def delegate_task_tool(params: DelegateTaskParams, context: Dict[str, Any]):
    """Delegate to isolated sub-agents (single, a concurrent batch, or background)."""
    ctx = context or {}
    if not params.tasks:
        return {"success": False, "error": "Provide at least one task."}
    if len(params.tasks) > MAX_CONCURRENT_CHILDREN:
        return {
            "success": False,
            "error": (
                f"Too many tasks ({len(params.tasks)}); max is "
                f"{MAX_CONCURRENT_CHILDREN}. Split into batches."
            ),
        }

    if params.background:
        dispatch = ctx.get("background_delegate")
        if not callable(dispatch):
            return {
                "success": False,
                "error": "Background delegation is not available in this context.",
            }
        dispatched = dispatch(
            [{"goal": item.goal, "context": item.context} for item in params.tasks]
        )
        return {
            "success": True,
            "background": True,
            "dispatched": dispatched,
            "note": "Running in the background; results arrive as a new turn when each finishes.",
        }

    runner = ctx.get("delegate_runner")
    if not callable(runner):
        return {
            "success": False,
            "error": (
                "Delegation is not available in this context "
                "(e.g. you are a leaf sub-agent, which cannot delegate further)."
            ),
        }
    try:
        results = runner(
            [
                {"goal": item.goal, "context": item.context, "role": item.role}
                for item in params.tasks
            ]
        )
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    return {"success": True, "count": len(results), "results": results}


def register_core_tools(registry=None):
    """Tools auto-register on import; kept for compatibility."""
    return None
