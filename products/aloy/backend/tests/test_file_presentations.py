"""Trusted file presentations: format routing, Office previews, and ranges."""

from __future__ import annotations

import io
import zipfile

import pytest

import aloy_backend.config as config_mod
import aloy_backend.storage as storage_mod
from aloy_backend.file_presentations import (
    DOCX_MIME,
    PPTX_MIME,
    XLSX_MIME,
    build_office_preview,
    presentation_kind,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod.settings, "storage_dir", str(tmp_path / "storage"))
    monkeypatch.setattr(storage_mod, "_STORE", None)
    yield
    monkeypatch.setattr(storage_mod, "_STORE", None)


def _archive(parts: dict[str, str]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        for name, value in parts.items():
            archive.writestr(name, value)
    return out.getvalue()


def _docx() -> bytes:
    return _archive(
        {
            "word/document.xml": (
                '<w:document xmlns:w="http://schemas.openxmlformats.org/'
                'wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>'
                "University plan</w:t></w:r></w:p><w:p><w:r><w:t>Week one"
                "</w:t></w:r></w:p></w:body></w:document>"
            )
        }
    )


def _pptx() -> bytes:
    return _archive(
        {
            "ppt/slides/slide1.xml": (
                '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/'
                '2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/'
                '2006/main"><p:cSld><a:t>Opening slide</a:t></p:cSld></p:sld>'
            ),
            "ppt/slides/slide2.xml": (
                '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/'
                '2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/'
                '2006/main"><p:cSld><a:t>Next steps</a:t></p:cSld></p:sld>'
            ),
        }
    )


def test_routes_supported_formats_to_fixed_host_renderers():
    assert presentation_kind("paper.pdf", "application/octet-stream") == "pdf"
    assert presentation_kind("lecture.mp4", "video/mp4") == "video"
    assert presentation_kind("camera.MP4", "application/octet-stream") == "video"
    assert presentation_kind("notes.docx", DOCX_MIME) == "document"
    assert presentation_kind("marks.xlsx", XLSX_MIME) == "spreadsheet"
    assert presentation_kind("deck.pptx", PPTX_MIME) == "slides"
    assert presentation_kind("archive.bin", "application/octet-stream") == "unknown"


def test_docx_and_pptx_become_inert_structured_previews():
    document = build_office_preview("document", _docx())
    assert document["blocks"] == ["University plan", "Week one"]
    slides = build_office_preview("slides", _pptx())
    assert [slide["text"] for slide in slides["slides"]] == [
        "Opening slide",
        "Next steps",
    ]


async def _upload(client, name: str, data: bytes, content_type: str) -> str:
    conversation = await client.post("/v1/conversations", json={"title": "files"})
    response = await client.post(
        f"/v1/conversations/{conversation.json()['id']}/files",
        files={"file": (name, data, content_type)},
    )
    assert response.status_code == 201, response.text
    return response.json()["file_id"]


async def test_presentation_endpoint_returns_docx_preview(client):
    file_id = await _upload(client, "plan.docx", _docx(), DOCX_MIME)
    response = await client.get(f"/v1/files/{file_id}/presentation")
    assert response.status_code == 200
    body = response.json()
    assert body["renderer"] == "document"
    assert body["preview"]["blocks"] == ["University plan", "Week one"]
    assert body["source_url"] is None  # local storage uses authenticated streaming


async def test_corrupt_office_file_keeps_original_available(client):
    file_id = await _upload(client, "broken.docx", b"not-a-zip", DOCX_MIME)
    response = await client.get(f"/v1/files/{file_id}/presentation")
    assert response.status_code == 200
    assert response.json()["renderer"] == "document"
    assert response.json()["preview"] is None
    assert "Invalid Office archive" in response.json()["preview_error"]
    assert (await client.get(f"/v1/files/{file_id}")).content == b"not-a-zip"


async def test_local_stream_supports_single_byte_range(client):
    file_id = await _upload(client, "lecture.mp4", b"0123456789", "video/mp4")
    response = await client.get(f"/v1/files/{file_id}", headers={"Range": "bytes=2-5"})
    assert response.status_code == 206
    assert response.content == b"2345"
    assert response.headers["content-range"] == "bytes 2-5/10"
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-disposition"].startswith("inline;")


async def test_file_presentation_is_user_scoped(client):
    file_id = await _upload(client, "private.pdf", b"%PDF", "application/pdf")
    response = await client.get(
        f"/v1/files/{file_id}/presentation",
        headers={"X-Test-User": "another-user"},
    )
    assert response.status_code == 404
