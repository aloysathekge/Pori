"""Image attachments: request validation + user-message persistence."""

import pytest
from pydantic import ValidationError

from aloy_backend.schemas import ImageAttachment, SendMessageRequest

pytestmark = pytest.mark.asyncio

PNG_1PX = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg=="
)


class TestValidation:
    def test_accepts_up_to_three_images(self):
        req = SendMessageRequest(
            content="what is this?",
            images=[{"data": PNG_1PX, "media_type": "image/png"}] * 3,
        )
        assert len(req.images) == 3

    def test_rejects_four_images(self):
        with pytest.raises(ValidationError):
            SendMessageRequest(
                content="x",
                images=[{"data": PNG_1PX, "media_type": "image/png"}] * 4,
            )

    def test_rejects_non_image_media_type(self):
        with pytest.raises(ValidationError):
            ImageAttachment(data=PNG_1PX, media_type="application/pdf")

    def test_rejects_svg(self):
        # SVG can carry scripts; only raster formats are allowed.
        with pytest.raises(ValidationError):
            ImageAttachment(data=PNG_1PX, media_type="image/svg+xml")


class TestPersistence:
    async def test_user_message_carries_images_in_metadata(self, client):
        created = await client.post("/v1/conversations", json={"title": "img"})
        conv_id = created.json()["id"]
        resp = await client.post(
            f"/v1/conversations/{conv_id}/messages",
            json={
                "content": "what is in this image?",
                "max_steps": 1,
                "images": [{"data": PNG_1PX, "media_type": "image/png"}],
            },
        )
        assert resp.status_code == 202

        detail = await client.get(f"/v1/conversations/{conv_id}")
        user_msgs = [m for m in detail.json()["messages"] if m["role"] == "user"]
        assert user_msgs, "user message not persisted"
        images = (user_msgs[0].get("metadata") or {}).get("images") or []
        assert len(images) == 1
        assert images[0]["media_type"] == "image/png"
        assert images[0]["data"] == PNG_1PX
