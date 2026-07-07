"""Multimodal message plumbing (Hermes gap Tier 1.1): content blocks in the
message types and their mapping in every provider adapter."""

import base64

import pytest

from pori.llm.anthropic import _to_anthropic_content
from pori.llm.messages import (
    ImageBlock,
    TextBlock,
    UserMessage,
    content_has_images,
    content_text,
)
from pori.llm.openai import _to_openai_content

PNG_BYTES = b"\x89PNG\r\n\x1a\nfakebytes"
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")


class TestMessageTypes:
    def test_plain_string_content_unchanged(self):
        msg = UserMessage(content="hello")
        assert msg.content == "hello"
        assert content_text(msg.content) == "hello"
        assert content_has_images(msg.content) is False

    def test_block_content(self):
        msg = UserMessage(
            content=[
                TextBlock(text="what is in this image?"),
                ImageBlock(source="base64", media_type="image/png", data=PNG_B64),
            ]
        )
        assert content_has_images(msg.content) is True
        text = content_text(msg.content)
        assert "what is in this image?" in text
        assert "[image: image/png]" in text

    def test_image_block_from_bytes(self):
        block = ImageBlock.from_bytes(PNG_BYTES, media_type="image/png")
        assert base64.b64decode(block.data) == PNG_BYTES
        assert block.source == "base64"

    def test_image_block_from_file(self, tmp_path):
        p = tmp_path / "shot.jpg"
        p.write_bytes(PNG_BYTES)
        block = ImageBlock.from_file(p)
        assert block.media_type == "image/jpeg"
        assert base64.b64decode(block.data) == PNG_BYTES


BLOCKS = [
    TextBlock(text="describe"),
    ImageBlock(source="base64", media_type="image/png", data=PNG_B64),
    ImageBlock(source="url", url="https://example.com/cat.png"),
]


class TestAnthropicMapping:
    def test_string_passthrough(self):
        assert _to_anthropic_content("plain") == "plain"

    def test_blocks(self):
        mapped = _to_anthropic_content(BLOCKS)
        assert mapped[0] == {"type": "text", "text": "describe"}
        assert mapped[1]["type"] == "image"
        assert mapped[1]["source"] == {
            "type": "base64",
            "media_type": "image/png",
            "data": PNG_B64,
        }
        assert mapped[2]["source"] == {
            "type": "url",
            "url": "https://example.com/cat.png",
        }


class TestOpenAIMapping:
    def test_string_passthrough(self):
        assert _to_openai_content("plain") == "plain"

    def test_blocks(self):
        mapped = _to_openai_content(BLOCKS)
        assert mapped[0] == {"type": "text", "text": "describe"}
        assert mapped[1] == {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{PNG_B64}"},
        }
        assert mapped[2] == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/cat.png"},
        }


class TestGoogleMapping:
    def test_mapping(self):
        pytest.importorskip("google.genai")
        from pori.llm.google import _to_google_parts

        parts = _to_google_parts(BLOCKS)
        assert parts[0].text == "describe"
        assert parts[1].inline_data is not None
        assert parts[1].inline_data.mime_type == "image/png"
        assert parts[1].inline_data.data == PNG_BYTES
        # URL sources degrade to a visible placeholder, never silently dropped
        assert "example.com/cat.png" in parts[2].text

    def test_string_becomes_single_text_part(self):
        pytest.importorskip("google.genai")
        from pori.llm.google import _to_google_parts

        parts = _to_google_parts("plain")
        assert len(parts) == 1
        assert parts[0].text == "plain"
