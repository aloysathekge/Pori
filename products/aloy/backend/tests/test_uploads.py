"""Durable uploads: endpoint (caps/quota/tenancy) + provisioning IN."""

import pytest

import aloy_backend.config as config_mod
import aloy_backend.storage as storage_mod
from aloy_backend.models import Run, StoredFile
from aloy_backend.provisioning import (
    provision_event_uploads,
    resolve_message_file_refs,
    resolve_upload_refs,
    uploads_task_block,
)
from aloy_backend.run_surface import resolve_run_surface
from aloy_backend.tenancy import OrganizationPolicy

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config_mod.settings, "sandbox_base_dir", str(tmp_path / "sandbox")
    )
    monkeypatch.setattr(config_mod.settings, "storage_dir", str(tmp_path / "storage"))
    monkeypatch.setattr(storage_mod, "_STORE", None)
    yield
    monkeypatch.setattr(storage_mod, "_STORE", None)


async def _conv(client) -> str:
    created = await client.post("/v1/conversations", json={"title": "up"})
    return created.json()["id"]


async def _event(client, title: str = "University") -> tuple[str, str]:
    created = await client.post(
        "/v1/events",
        json={"title": title, "setup_mode": "simple"},
    )
    assert created.status_code == 201, created.text
    body = created.json()
    return body["id"], body["conversation_id"]


class TestUploadEndpoint:
    async def test_library_upload_is_immediately_available(self, client):
        response = await client.post(
            "/v1/files",
            files={"file": ("reference.pdf", b"%PDF-library", "application/pdf")},
        )
        assert response.status_code == 201, response.text
        assert response.json()["in_library"] is True
        assert response.json()["conversation_id"] is None

        library = await client.get("/v1/files")
        assert [row["name"] for row in library.json()] == ["reference.pdf"]

    async def test_event_picker_is_scoped_and_life_picker_sees_all(self, client):
        life_conv = await _conv(client)
        life_upload = await client.post(
            f"/v1/conversations/{life_conv}/files",
            files={"file": ("life.txt", b"life", "text/plain")},
        )
        event_id, event_conv = await _event(client)
        event_upload = await client.post(
            f"/v1/conversations/{event_conv}/files",
            files={"file": ("course.txt", b"course", "text/plain")},
        )
        assert life_upload.status_code == event_upload.status_code == 201

        event_files = await client.get(f"/v1/conversations/{event_conv}/files")
        assert event_files.status_code == 200
        assert {row["name"] for row in event_files.json()} == {"course.txt"}
        assert {row["event_id"] for row in event_files.json()} == {event_id}

        life_files = await client.get(f"/v1/conversations/{life_conv}/files")
        assert life_files.status_code == 200
        assert {row["name"] for row in life_files.json()} == {
            "life.txt",
            "course.txt",
        }

        searched = await client.get(
            f"/v1/conversations/{life_conv}/files",
            params={"q": "course"},
        )
        assert [row["name"] for row in searched.json()] == ["course.txt"]

    async def test_upload_roundtrip_and_eager_provisioning(self, client, tmp_path):
        conv_id = await _conv(client)
        resp = await client.post(
            f"/v1/conversations/{conv_id}/files",
            files={"file": ("data.csv", b"a,b\n1,2\n", "text/csv")},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["name"] == "data.csv"
        assert body["size_bytes"] == 8

        # Bytes are durable in the store...
        store = storage_mod.get_object_store()
        blobs = list(store.root.rglob("data.csv"))
        assert len(blobs) == 1
        # ...and already provisioned into the conversation's sandbox uploads.
        detail = await client.get(f"/v1/conversations/{conv_id}")
        event_id = detail.json()["event_id"]
        provisioned = tmp_path / "sandbox" / "events" / event_id
        assert (provisioned / "user-data" / "uploads" / "data.csv").is_file()

        # Downloadable through /files/{id}.
        dl = await client.get(f"/v1/files/{body['file_id']}")
        assert dl.status_code == 200
        assert dl.content == b"a,b\n1,2\n"

    async def test_oversize_upload_rejected(self, client, monkeypatch):
        monkeypatch.setattr(config_mod.settings, "storage_max_file_mb", 0)
        conv_id = await _conv(client)
        resp = await client.post(
            f"/v1/conversations/{conv_id}/files",
            files={"file": ("big.bin", b"x" * 10, "application/octet-stream")},
        )
        assert resp.status_code == 413

    async def test_org_quota_enforced(self, client, monkeypatch):
        monkeypatch.setattr(config_mod.settings, "storage_org_quota_mb", 0)
        conv_id = await _conv(client)
        resp = await client.post(
            f"/v1/conversations/{conv_id}/files",
            files={"file": ("x.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 413
        assert "quota" in resp.json()["detail"].lower()

    async def test_empty_upload_rejected(self, client):
        conv_id = await _conv(client)
        resp = await client.post(
            f"/v1/conversations/{conv_id}/files",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert resp.status_code == 422

    async def test_ref_merges_onto_inline_chip_of_same_file(self, client):
        """A small text file rides inline AND durable: one chip, with both
        content and file_id — not two chips."""
        conv_id = await _conv(client)
        up = await client.post(
            f"/v1/conversations/{conv_id}/files",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        file_id = up.json()["file_id"]
        resp = await client.post(
            f"/v1/conversations/{conv_id}/messages",
            json={
                "content": "read my notes",
                "max_steps": 1,
                "files": [{"name": "notes.txt", "content": "hello"}],
                "file_refs": [file_id],
            },
        )
        assert resp.status_code == 202
        detail = await client.get(f"/v1/conversations/{conv_id}")
        user_msgs = [m for m in detail.json()["messages"] if m["role"] == "user"]
        chips = (user_msgs[0].get("metadata") or {}).get("files") or []
        assert len(chips) == 1
        assert chips[0]["file_id"] == file_id
        assert chips[0]["content"] == "hello"

    async def test_message_with_ref_gets_chip_metadata(self, client):
        conv_id = await _conv(client)
        up = await client.post(
            f"/v1/conversations/{conv_id}/files",
            files={"file": ("cv.pdf", b"%PDF-fake", "application/pdf")},
        )
        file_id = up.json()["file_id"]
        resp = await client.post(
            f"/v1/conversations/{conv_id}/messages",
            json={"content": "summarize my cv", "max_steps": 1, "file_refs": [file_id]},
        )
        assert resp.status_code == 202
        detail = await client.get(f"/v1/conversations/{conv_id}")
        user_msgs = [m for m in detail.json()["messages"] if m["role"] == "user"]
        chips = (user_msgs[0].get("metadata") or {}).get("files") or []
        assert chips and chips[0]["file_id"] == file_id
        assert chips[0]["name"] == "cv.pdf"

    async def test_only_selected_file_enters_durable_run_task(
        self, client, db_session_maker
    ):
        _, conv_id = await _event(client)
        selected = await client.post(
            f"/v1/conversations/{conv_id}/files",
            files={"file": ("selected.csv", b"a,b\n1,2", "text/csv")},
        )
        unselected = await client.post(
            f"/v1/conversations/{conv_id}/files",
            files={"file": ("private.csv", b"secret", "text/csv")},
        )
        assert selected.status_code == unselected.status_code == 201

        response = await client.post(
            f"/v1/conversations/{conv_id}/messages",
            json={
                "content": "Use the attached dataset",
                "file_refs": [selected.json()["file_id"]],
            },
        )
        assert response.status_code == 202, response.text
        async with db_session_maker() as session:
            run = await session.get(Run, response.json()["run_id"])
            assert run is not None
            assert "selected.csv" in run.task
            assert "private.csv" not in run.task

    async def test_life_may_explicitly_reference_an_event_file(self, client):
        life_conv = await _conv(client)
        _, event_conv = await _event(client)
        uploaded = await client.post(
            f"/v1/conversations/{event_conv}/files",
            files={"file": ("semester-plan.pdf", b"%PDF-plan", "application/pdf")},
        )
        response = await client.post(
            f"/v1/conversations/{life_conv}/messages",
            json={
                "content": "Compare this with my other commitments",
                "file_refs": [uploaded.json()["file_id"]],
            },
        )
        assert response.status_code == 202, response.text
        detail = await client.get(f"/v1/conversations/{life_conv}")
        user_message = next(
            message
            for message in detail.json()["messages"]
            if message["role"] == "user"
        )
        assert user_message["metadata"]["files"][0]["name"] == "semester-plan.pdf"

    async def test_dedicated_event_drops_out_of_scope_file_ref(self, client):
        life_conv = await _conv(client)
        life_upload = await client.post(
            f"/v1/conversations/{life_conv}/files",
            files={"file": ("personal.txt", b"private", "text/plain")},
        )
        _, event_conv = await _event(client)
        response = await client.post(
            f"/v1/conversations/{event_conv}/messages",
            json={
                "content": "Use a file I did not grant",
                "file_refs": [life_upload.json()["file_id"]],
            },
        )
        assert response.status_code == 202, response.text
        detail = await client.get(f"/v1/conversations/{event_conv}")
        user_message = next(
            message
            for message in detail.json()["messages"]
            if message["role"] == "user"
        )
        assert not (user_message.get("metadata") or {}).get("files")

    async def test_run_library_respects_event_authority(self, client, db_session_maker):
        life_conv = await _conv(client)
        life_detail = await client.get(f"/v1/conversations/{life_conv}")
        life_event_id = life_detail.json()["event_id"]
        life_file = await client.post(
            "/v1/files",
            files={"file": ("life-plan.txt", b"life", "text/plain")},
        )

        event_id, event_conv = await _event(client)
        event_file = await client.post(
            f"/v1/conversations/{event_conv}/files",
            files={"file": ("course-plan.txt", b"course", "text/plain")},
        )
        await client.post(f"/v1/files/{event_file.json()['file_id']}/library")
        explicit_file = await client.post(
            f"/v1/conversations/{event_conv}/files",
            files={"file": ("selected-notes.txt", b"selected", "text/plain")},
        )

        async with db_session_maker() as session:
            dedicated = await resolve_run_surface(
                session,
                organization_id="user:test-user",
                user_id="test-user",
                event_id=event_id,
                policy=OrganizationPolicy(),
            )
            life = await resolve_run_surface(
                session,
                organization_id="user:test-user",
                user_id="test-user",
                event_id=life_event_id,
                policy=OrganizationPolicy(),
            )
            selected = await resolve_run_surface(
                session,
                organization_id="user:test-user",
                user_id="test-user",
                event_id=event_id,
                policy=OrganizationPolicy(),
                explicit_file_ids=(explicit_file.json()["file_id"],),
            )
            life_with_foreign_selection = await resolve_run_surface(
                session,
                organization_id="user:test-user",
                user_id="test-user",
                event_id=life_event_id,
                policy=OrganizationPolicy(),
                explicit_file_ids=(explicit_file.json()["file_id"],),
            )

        assert life_file.status_code == event_file.status_code == 201
        assert {row["name"] for row in dedicated.library} == {"course-plan.txt"}
        assert {row["name"] for row in life.library} == {
            "life-plan.txt",
            "course-plan.txt",
        }
        assert {row["name"] for row in selected.library} == {
            "course-plan.txt",
            "selected-notes.txt",
        }
        assert {row["name"] for row in life_with_foreign_selection.library} == {
            "life-plan.txt",
            "course-plan.txt",
        }


def _record(rec_id: str, name: str, sha: str, **kw) -> StoredFile:
    return StoredFile(
        id=rec_id,
        organization_id=kw.get("org", "org-1"),
        user_id="u1",
        event_id=kw.get("event", "evt-upload"),
        origin_session_id=kw.get("conv", "conv-1"),
        conversation_id=kw.get("conv", "conv-1"),
        kind=kw.get("kind", "upload"),
        name=name,
        content_type="text/plain",
        size_bytes=kw.get("size", 5),
        sha256=sha,
        storage_key=kw.get("key", ""),
    )


class TestProvisioning:
    def _put(self, name: str, data: bytes, conv="conv-1") -> StoredFile:
        import hashlib
        import io
        import uuid

        from aloy_backend.storage import upload_key

        store = storage_mod.get_object_store()
        rec = _record(
            uuid.uuid4().hex, name, hashlib.sha256(data).hexdigest(), size=len(data)
        )
        rec.storage_key = upload_key("org-1", conv, rec.id, name)
        store.put(rec.storage_key, io.BytesIO(data), content_type="text/plain")
        return rec

    def test_provisions_and_skips_when_hash_matches(self, tmp_path):
        rec = self._put("notes.txt", b"hello")
        entries = provision_event_uploads("evt-upload", [rec])
        assert entries[0]["name"] == "notes.txt"
        target = (
            tmp_path
            / "sandbox"
            / "events"
            / "evt-upload"
            / "user-data"
            / "uploads"
            / "notes.txt"
        )
        assert target.read_bytes() == b"hello"
        # Second call: skip (delete the blob to PROVE no re-copy happens).
        storage_mod.get_object_store().delete(rec.storage_key)
        entries2 = provision_event_uploads("evt-upload", [rec])
        assert entries2[0]["name"] == "notes.txt"
        assert target.read_bytes() == b"hello"

    def test_same_name_different_content_does_not_overwrite(self, tmp_path):
        a = self._put("data.csv", b"version-a")
        b = self._put("data.csv", b"version-b")
        entries = provision_event_uploads("evt-upload", [a, b])
        names = {e["name"] for e in entries}
        assert "data.csv" in names and len(names) == 2
        uploads = (
            tmp_path / "sandbox" / "events" / "evt-upload" / "user-data" / "uploads"
        )
        contents = {p.read_bytes() for p in uploads.iterdir() if p.suffix == ".csv"}
        assert contents == {b"version-a", b"version-b"}

    def test_task_block_lists_files_with_guidance(self):
        block = uploads_task_block(
            [
                {
                    "name": "data.csv",
                    "size_bytes": 198_000_000,
                    "content_type": "text/csv",
                }
            ]
        )
        assert "/mnt/user-data/uploads/data.csv" in block
        assert "188.8MB" in block
        assert "head" in block  # sampled-access guidance

    def test_resolve_refs_drops_foreign_rows(self):
        mine = _record("f1", "a.txt", "s1")
        other_org = _record("f2", "b.txt", "s2", org="org-2")
        other_event = _record("f3", "c.txt", "s3", event="evt-other")
        artifact = _record("f4", "d.txt", "s4", kind="artifact")
        out = resolve_upload_refs(
            [mine, other_org, other_event, artifact, None],
            organization_id="org-1",
            event_id="evt-upload",
        )
        assert out == [mine]

    def test_message_ref_scope_accepts_artifacts_and_life_cross_event(self):
        upload = _record("f1", "a.txt", "s1")
        artifact = _record("f2", "report.md", "s2", kind="artifact")
        other_event = _record("f3", "other.txt", "s3", event="evt-other")
        other_user = _record("f4", "foreign.txt", "s4")
        other_user.user_id = "u2"

        dedicated = resolve_message_file_refs(
            [upload, artifact, other_event, other_user],
            organization_id="org-1",
            user_id="u1",
            event_id="evt-upload",
            life_scope=False,
        )
        assert dedicated == [upload, artifact]
        life = resolve_message_file_refs(
            [upload, artifact, other_event, other_user],
            organization_id="org-1",
            user_id="u1",
            event_id="evt-life",
            life_scope=True,
        )
        assert life == [upload, artifact, other_event]


class TestExtractedCompanion:
    def _docx_bytes(self) -> bytes:
        import io
        import zipfile

        # Minimal OOXML: one paragraph "Hello Aloy CV".
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr(
                "word/document.xml",
                '<?xml version="1.0"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/'
                'wordprocessingml/2006/main"><w:body><w:p><w:r>'
                "<w:t>Hello Aloy CV</w:t></w:r></w:p></w:body></w:document>",
            )
        return buf.getvalue()

    def test_docx_gets_plain_text_companion(self, tmp_path):
        import hashlib
        import io
        import uuid
        from datetime import datetime, timezone
        from types import SimpleNamespace

        from aloy_backend.provisioning import provision_event_uploads
        from aloy_backend.storage import upload_key

        data = self._docx_bytes()
        rec = SimpleNamespace(
            id=uuid.uuid4().hex,
            name="cv.docx",
            size_bytes=len(data),
            content_type="application/vnd.openxmlformats-officedocument"
            ".wordprocessingml.document",
            sha256=hashlib.sha256(data).hexdigest(),
            storage_key="",
            created_at=datetime.now(timezone.utc),
        )
        rec.storage_key = upload_key("org-1", "conv-x", rec.id, rec.name)
        storage_mod.get_object_store().put(
            rec.storage_key, io.BytesIO(data), content_type=rec.content_type
        )

        entries = provision_event_uploads("evt-docx", [rec])
        assert entries[0]["extracted_text"] == "cv.docx.extracted.txt"
        companion = (
            tmp_path
            / "sandbox"
            / "events"
            / "evt-docx"
            / "user-data"
            / "uploads"
            / "cv.docx.extracted.txt"
        )
        assert "Hello Aloy CV" in companion.read_text(encoding="utf-8")

    def test_task_block_mentions_the_companion(self):
        from aloy_backend.provisioning import uploads_task_block

        block = uploads_task_block(
            [
                {
                    "name": "cv.docx",
                    "size_bytes": 12345,
                    "content_type": "application/vnd.openxmlformats",
                    "extracted_text": "cv.docx.extracted.txt",
                }
            ]
        )
        assert "plain-text copy: /mnt/user-data/uploads/cv.docx.extracted.txt" in block
