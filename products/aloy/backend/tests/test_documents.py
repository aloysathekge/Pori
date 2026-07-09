"""Document attachments: docx/xlsx extraction (Hermes-harvested) + pdf routing."""

import base64
import io
import zipfile

import pytest
from aloy_backend.doc_extract import (
    ExtractionError,
    extract_docx_text,
    extract_xlsx_text,
)
from aloy_backend.schemas import DOCX_MIME, DocumentAttachment

pytestmark = pytest.mark.asyncio


def make_docx(paragraphs: list[str]) -> bytes:
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    doc = (
        '<?xml version="1.0"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", doc)
    return buf.getvalue()


def make_xlsx(rows: list[list[str]]) -> bytes:
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    cells = "".join(
        f'<row r="{ri + 1}">'
        + "".join(
            f'<c r="{chr(65 + ci)}{ri + 1}" t="inlineStr"><is><t>{v}</t></is></c>'
            for ci, v in enumerate(row)
        )
        + "</row>"
        for ri, row in enumerate(rows)
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{ns}" xmlns:r="{rel}">'
            '<sheets><sheet name="Data" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        zf.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{ns}"><sheetData>{cells}</sheetData></worksheet>',
        )
    return buf.getvalue()


class TestExtraction:
    def test_docx_paragraphs(self):
        text = extract_docx_text(make_docx(["Hello world", "Second para"]))
        assert "Hello world" in text and "Second para" in text

    def test_docx_garbage_raises(self):
        with pytest.raises(ExtractionError):
            extract_docx_text(b"not a zip at all")

    def test_xlsx_rows(self):
        text = extract_xlsx_text(make_xlsx([["Name", "Score"], ["Aloy", "10"]]))
        assert "Sheet: Data" in text
        assert "Name\tScore" in text and "Aloy\t10" in text

    def test_xlsx_garbage_raises(self):
        with pytest.raises(ExtractionError):
            extract_xlsx_text(b"nope")


class TestRequests:
    def test_media_type_validation(self):
        with pytest.raises(ValueError):
            DocumentAttachment(name="x.bin", data="aGk=", media_type="application/zip")

    async def test_docx_message_persists_chip(self, client):
        created = await client.post("/v1/conversations", json={"title": "docs"})
        conv_id = created.json()["id"]
        docx_b64 = base64.b64encode(make_docx(["quarterly report body"])).decode()
        resp = await client.post(
            f"/v1/conversations/{conv_id}/messages",
            json={
                "content": "summarize",
                "max_steps": 1,
                "documents": [
                    {"name": "report.docx", "data": docx_b64, "media_type": DOCX_MIME}
                ],
            },
        )
        assert resp.status_code == 202
        detail = await client.get(f"/v1/conversations/{conv_id}")
        user_msgs = [m for m in detail.json()["messages"] if m["role"] == "user"]
        files = (user_msgs[0].get("metadata") or {}).get("files") or []
        assert files and files[0]["name"] == "report.docx"

    async def test_corrupt_docx_rejected(self, client):
        created = await client.post("/v1/conversations", json={"title": "bad"})
        conv_id = created.json()["id"]
        resp = await client.post(
            f"/v1/conversations/{conv_id}/messages",
            json={
                "content": "read this",
                "max_steps": 1,
                "documents": [
                    {
                        "name": "broken.docx",
                        "data": base64.b64encode(b"garbage").decode(),
                        "media_type": DOCX_MIME,
                    }
                ],
            },
        )
        assert resp.status_code == 422
