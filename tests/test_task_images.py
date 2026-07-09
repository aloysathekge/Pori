"""Multimodal turns: images attached to a task ride with the CURRENT TASK
message as content blocks, so provider adapters map them natively."""

import pytest

from pori import AgentSettings, ImageBlock, TextBlock
from pori.llm.messages import UserMessage
from pori.orchestrator.core import Orchestrator

pytestmark = pytest.mark.agent

PNG_1PX = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg=="
)


async def test_task_attachments_ride_with_the_task_message(mock_llm, tool_registry):
    image = ImageBlock(source="base64", media_type="image/png", data=PNG_1PX)
    orch = Orchestrator(llm=mock_llm, tools_registry=tool_registry)
    await orch.execute_task(
        "What is in this image?",
        agent_settings=AgentSettings(max_steps=2),
        task_attachments=[image],
    )

    assert mock_llm.ainvoke_calls, "the LLM was never invoked"
    messages = mock_llm.ainvoke_calls[0]
    final = messages[-1]
    assert isinstance(final, UserMessage)
    assert isinstance(final.content, list), "task message should be content blocks"
    kinds = [type(b).__name__ for b in final.content]
    assert "ImageBlock" in kinds and "TextBlock" in kinds
    # Image first, task text last (text carries the CURRENT TASK directive).
    assert isinstance(final.content[0], ImageBlock)
    assert final.content[0].data == PNG_1PX
    text_block = final.content[-1]
    assert isinstance(text_block, TextBlock)
    assert "What is in this image?" in text_block.text


async def test_no_images_keeps_plain_string_task(mock_llm, tool_registry):
    orch = Orchestrator(llm=mock_llm, tools_registry=tool_registry)
    await orch.execute_task("hello", agent_settings=AgentSettings(max_steps=2))
    final = mock_llm.ainvoke_calls[0][-1]
    assert isinstance(final.content, str)
    assert "hello" in final.content


def test_document_block_maps_to_anthropic_document():
    from pori import DocumentBlock
    from pori.llm.anthropic import _to_anthropic_content
    from pori.llm.messages import TextBlock

    blocks = _to_anthropic_content(
        [
            DocumentBlock(media_type="application/pdf", data="QUJD", name="r.pdf"),
            TextBlock(text="summarize"),
        ]
    )
    assert blocks[0]["type"] == "document"
    assert blocks[0]["source"]["media_type"] == "application/pdf"
    assert blocks[0]["source"]["data"] == "QUJD"
    assert blocks[1] == {"type": "text", "text": "summarize"}


async def test_documents_ride_with_the_task(mock_llm, tool_registry):
    from pori import AgentSettings, DocumentBlock
    from pori.orchestrator.core import Orchestrator

    doc = DocumentBlock(media_type="application/pdf", data="QUJD", name="r.pdf")
    orch = Orchestrator(llm=mock_llm, tools_registry=tool_registry)
    await orch.execute_task(
        "Summarize the attached report",
        agent_settings=AgentSettings(max_steps=2),
        task_attachments=[doc],
    )
    final = mock_llm.ainvoke_calls[0][-1]
    assert isinstance(final.content, list)
    assert any(type(b).__name__ == "DocumentBlock" for b in final.content)
