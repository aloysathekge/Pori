"""Helpers shared across the conversations route package.

Contract: conversation loading (ownership-checked), rendering of attached
text files into the model task, best-effort title generation, and the
artifact-receipt utilities (allowlist of files a conversation's runs wrote).
No routes live here.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from sqlalchemy.ext.asyncio import AsyncSession

from pori.llm import BaseChatModel

from ...deps import load_owned
from ...models import Conversation, Message
from ...tenancy import OrganizationContext

logger = logging.getLogger("aloy_backend")

_load_owned_conversation = load_owned(Conversation)


async def _load_conv(
    session: AsyncSession, context: OrganizationContext, conversation_id: str
) -> Conversation:
    """Load a conversation owned by the caller's org, or 404."""
    return await _load_owned_conversation(conversation_id, context, session)


def _render_file_block(name: str, content: str) -> str:
    """How an attached text file is shown to the model, appended to the task."""
    return f'\n\n<attached-file name="{name}">\n{content}\n</attached-file>'


async def _maybe_generate_title(
    session: AsyncSession,
    conv: Conversation,
    llm: BaseChatModel,
    first_user_content: str,
) -> None:
    """Give an untitled conversation a short topic title from its first message
    (like ChatGPT/Claude). Best-effort: an LLM title, else a clean heuristic."""
    if conv.title:
        return
    title = ""
    try:
        from pori import SystemMessage, UserMessage

        raw = await llm.ainvoke(
            [
                SystemMessage(
                    content=(
                        "Generate a concise 3-6 word title for a conversation that "
                        "opens with the user's message. Reply with ONLY the title — "
                        "no quotes, no trailing punctuation."
                    )
                ),
                UserMessage(content=first_user_content[:1000]),
            ]
        )
        title = (raw or "").strip().strip("\"'").splitlines()[0][:60].strip()
    except Exception:
        logger.debug("Title generation failed; using heuristic", exc_info=True)
    if not title:
        words = first_user_content.strip().split()
        title = " ".join(words[:6])[:60].strip() or "New conversation"
        title = title[:1].upper() + title[1:]
    conv.title = title
    session.add(conv)
    await session.commit()


# ---- artifacts ----

_LANG_BY_EXT = {
    ".py": "python",
    ".md": "markdown",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".html": "html",
    ".css": "css",
    ".sh": "bash",
    ".sql": "sql",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".txt": "text",
    ".toml": "toml",
    ".env": "bash",
}
_ARTIFACT_MAX_BYTES = 200_000


def _conversation_artifacts(messages: Iterable[Message]) -> dict:
    """path -> {path, tool_name, bytes_written, message_id} across a conversation
    (the allowlist of files its runs actually wrote)."""
    out: dict = {}
    for m in messages:
        for a in (m.metadata_ or {}).get("artifacts", []) or []:
            path = a.get("path")
            if path:
                out[path] = {
                    "path": path,
                    "tool_name": a.get("tool_name"),
                    "bytes_written": a.get("bytes_written"),
                    "message_id": m.id,
                    "file_id": a.get("file_id"),
                }
    return out
