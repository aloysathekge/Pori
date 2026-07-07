"""The gateway service: Telegram long-poll loop → durable runs → replies.

Flow per inbound message:
- Unpaired chat: the text is treated as a pairing code (minted by
  POST /v1/gateway/pair). A valid code creates the GatewayLink + a dedicated
  Conversation; anything else gets pairing instructions.
- Paired chat: the text is enqueued as an ordinary durable Run bound to the
  link's conversation — so gateway messages get the whole marathon treatment
  (worker execution, checkpointing, resume, salvage) for free — and the
  answer is sent back to the chat when the run finishes.

Run as its own process: ``pori-cloud-gateway`` (compose service `gateway`).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from sqlmodel import select

from ..config import settings
from ..database import async_session
from ..models import Conversation, GatewayLink, GatewayPairingCode, Run
from .delivery import DeliveryRouter
from .registry import build_adapters
from .telegram import TelegramAdapter

logger = logging.getLogger("pori_cloud.gateway")

PAIRING_HELP = (
    "This chat isn't paired with an Aloy account yet.\n\n"
    "Open the Aloy app, request a pairing code, and send it here as a "
    "message (it looks like AB12CD34)."
)
PAIRED_REPLY = "Paired! Send me a task and I'll get to work."
ACK_REPLY = "On it — I'll reply here when I'm done."


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime) -> datetime:
    """SQLite drops tzinfo on round-trip; normalize before comparing."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


async def try_pair(chat_id: str, chat_title: str, code: str) -> Optional[GatewayLink]:
    """Exchange a pairing code for a GatewayLink (+ dedicated conversation)."""
    normalized = code.strip().upper()
    if not normalized or len(normalized) > 16:
        return None
    async with async_session() as session:
        pairing = await session.get(GatewayPairingCode, normalized)
        if pairing is None or _aware(pairing.expires_at) < _utcnow():
            return None
        conversation = Conversation(
            organization_id=pairing.organization_id,
            user_id=pairing.user_id,
            title=f"Telegram: {chat_title or chat_id}",
        )
        session.add(conversation)
        await session.flush()
        link = GatewayLink(
            organization_id=pairing.organization_id,
            user_id=pairing.user_id,
            platform="telegram",
            chat_id=chat_id,
            chat_title=chat_title or None,
            conversation_id=conversation.id,
        )
        session.add(link)
        await session.delete(pairing)  # one-time use
        await session.commit()
        await session.refresh(link)
        logger.info(
            "Paired telegram chat %s to user %s (org %s)",
            chat_id,
            link.user_id,
            link.organization_id,
        )
        return link


async def find_link(chat_id: str) -> Optional[GatewayLink]:
    async with async_session() as session:
        result = await session.execute(
            select(GatewayLink).where(
                GatewayLink.platform == "telegram",
                GatewayLink.chat_id == chat_id,
            )
        )
        return result.scalars().first()


async def enqueue_run_for_link(link: GatewayLink, task: str) -> str:
    """Inbound message → ordinary durable Run on the worker queue."""
    async with async_session() as session:
        run = Run(
            user_id=link.user_id,
            organization_id=link.organization_id,
            agent_id="default_agent",
            session_id="pending",
            conversation_id=link.conversation_id,
            task=task,
            status="pending",
        )
        session.add(run)
        await session.flush()
        run.session_id = run.id
        run.root_run_id = run.id
        session.add(run)
        await session.commit()
        logger.info("Gateway enqueued run %s for chat %s", run.id, link.chat_id)
        return run.id


async def collect_finished(
    pending: Dict[str, str],
) -> Dict[str, tuple[str, Optional[str], bool]]:
    """Poll tracked runs; return {run_id: (chat_id, answer, success)} for the
    ones that reached a terminal state (and drop them from ``pending``)."""
    if not pending:
        return {}
    finished: Dict[str, tuple[str, Optional[str], bool]] = {}
    async with async_session() as session:
        for run_id in list(pending.keys()):
            run = await session.get(Run, run_id)
            if run is None:
                finished[run_id] = (pending.pop(run_id), None, False)
                continue
            if run.status in {"completed", "failed", "cancelled"}:
                finished[run_id] = (
                    pending.pop(run_id),
                    run.final_answer,
                    run.status == "completed" and run.success,
                )
    return finished


def _format_result(answer: Optional[str], success: bool) -> str:
    if answer:
        return answer if success else f"I couldn't fully finish, but:\n\n{answer}"
    return (
        "Done."
        if success
        else "Sorry — that task failed and produced no answer. Try rephrasing?"
    )


async def handle_message(
    adapter: TelegramAdapter, chat_id: str, chat_title: str, text: str
) -> Optional[str]:
    """Process one inbound text; returns an enqueued run id when one starts."""
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("/start"):
        text = text[len("/start") :].strip()
    link = await find_link(chat_id)
    if link is None:
        if text and await try_pair(chat_id, chat_title, text):
            await adapter.send(chat_id, PAIRED_REPLY)
        else:
            await adapter.send(chat_id, PAIRING_HELP)
        return None
    if not text:
        return None
    run_id = await enqueue_run_for_link(link, text)
    await adapter.send(chat_id, ACK_REPLY)
    return run_id


async def serve() -> None:
    adapters = build_adapters()
    telegram = adapters.get("telegram")
    if not isinstance(telegram, TelegramAdapter):
        raise SystemExit("Gateway has no platforms configured. Set TELEGRAM_BOT_TOKEN.")
    router = DeliveryRouter(adapters)
    logger.info("Aloy gateway started (telegram)")

    offset: Optional[int] = None
    pending: Dict[str, str] = {}  # run_id -> chat_id
    while True:
        try:
            updates = await telegram.get_updates(offset)
        except Exception:
            logger.exception("getUpdates failed; backing off")
            await asyncio.sleep(settings.gateway_error_backoff_seconds)
            continue
        for update in updates:
            offset = int(update.get("update_id", 0)) + 1
            message = update.get("message") or {}
            chat = message.get("chat") or {}
            chat_id = str(chat.get("id", "")).strip()
            if not chat_id:
                continue
            chat_title = str(chat.get("title") or chat.get("username") or "").strip()
            try:
                run_id = await handle_message(
                    telegram, chat_id, chat_title, str(message.get("text") or "")
                )
                if run_id:
                    pending[run_id] = chat_id
            except Exception:
                logger.exception("Failed to handle message from chat %s", chat_id)

        try:
            for _run_id, (chat_id, answer, success) in (
                await collect_finished(pending)
            ).items():
                link = await find_link(chat_id)
                text = _format_result(answer, success)
                if link is not None:
                    await router.deliver_to_link(link, text)
                else:
                    await telegram.send(chat_id, text)
        except Exception:
            logger.exception("Result delivery pass failed")

        if not updates:
            await asyncio.sleep(settings.gateway_idle_sleep_seconds)


def run() -> None:
    logging.basicConfig(level=settings.log_level)
    asyncio.run(serve())


if __name__ == "__main__":
    run()
