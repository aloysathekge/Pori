"""The user file library: save/remove lifecycle (memory pointer moves with
the flag) + the fetch_my_file tool (materialize on demand, self-correcting
on miss)."""

import pytest

import aloy_backend.config as config_mod
import aloy_backend.storage as storage_mod
from aloy_backend.library import library_entry_id, library_manifest
from aloy_backend.tools.library import FetchMyFileParams, fetch_my_file_tool

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


async def _upload(client, name=b"cv.pdf".decode(), data=b"%PDF-my-cv"):
    created = await client.post("/v1/conversations", json={"title": "lib"})
    conv_id = created.json()["id"]
    up = await client.post(
        f"/v1/conversations/{conv_id}/files",
        files={"file": (name, data, "application/pdf")},
    )
    return conv_id, up.json()["file_id"]


class TestLibraryLifecycle:
    async def test_save_writes_flag_and_memory_pointer(self, client, db_session_maker):
        _, file_id = await _upload(client)
        resp = await client.post(f"/v1/files/{file_id}/library")
        assert resp.status_code == 200
        assert resp.json()["in_library"] is True

        from aloy_backend.models import KnowledgeEntry

        async with db_session_maker() as s:
            entry = await s.get(KnowledgeEntry, library_entry_id(file_id))
        assert entry is not None
        assert "cv.pdf" in entry.content
        assert "fetch_my_file" in entry.content
        assert entry.session_id is None  # available to sibling chats in the Event
        assert entry.event_id is not None
        assert entry.tags == ["file-library"]

        listed = await client.get("/v1/files")
        assert [f["file_id"] for f in listed.json()] == [file_id]
        assert listed.json()[0]["event_title"]
        assert listed.json()[0]["event_is_life"] is True

    async def test_listed_file_identifies_its_origin_event(self, client):
        event = await client.post(
            "/v1/events",
            json={"title": "University 2026", "setup_mode": "simple"},
        )
        conversation_id = event.json()["conversation_id"]
        uploaded = await client.post(
            f"/v1/conversations/{conversation_id}/files",
            files={"file": ("timetable.ts", b"export const term = 2", "video/mp2t")},
        )
        file_id = uploaded.json()["file_id"]
        await client.post(f"/v1/files/{file_id}/library")

        listed = await client.get("/v1/files")
        source = next(file for file in listed.json() if file["file_id"] == file_id)
        assert source["event_title"] == "University 2026"
        assert source["event_is_life"] is False

        surface = (await client.get(f"/v1/events/{event.json()['id']}")).json()
        files_section = next(
            section for section in surface["surface"]["sections"]
            if section["kind"] == "files"
        )
        event_file = next(file for file in files_section["files"] if file["id"] == file_id)
        assert event_file["in_library"] is True

    async def test_remove_deletes_the_pointer_too(self, client, db_session_maker):
        _, file_id = await _upload(client)
        await client.post(f"/v1/files/{file_id}/library")
        resp = await client.delete(f"/v1/files/{file_id}/library")
        assert resp.status_code == 200
        assert resp.json()["in_library"] is False

        from aloy_backend.models import KnowledgeEntry

        async with db_session_maker() as s:
            entry = await s.get(KnowledgeEntry, library_entry_id(file_id))
        assert entry is None  # memory never points at nothing
        assert (await client.get("/v1/files")).json() == []

    async def test_saving_twice_is_idempotent(self, client):
        _, file_id = await _upload(client)
        await client.post(f"/v1/files/{file_id}/library")
        resp = await client.post(f"/v1/files/{file_id}/library")
        assert resp.status_code == 200
        assert len((await client.get("/v1/files")).json()) == 1

    async def test_other_users_files_are_invisible(self, client):
        _, file_id = await _upload(client)
        resp = await client.post(
            f"/v1/files/{file_id}/library", headers={"X-Test-User": "someone-else"}
        )
        assert resp.status_code == 404

    async def test_delete_upload_removes_file_library_pointer_and_event_row(
        self, client, db_session_maker
    ):
        conversation_id, file_id = await _upload(client)
        await client.post(f"/v1/files/{file_id}/library")

        deleted = await client.delete(f"/v1/files/{file_id}")
        assert deleted.status_code == 204
        assert (await client.get(f"/v1/files/{file_id}")).status_code == 404
        assert (await client.get("/v1/files")).json() == []

        conversation = (await client.get(f"/v1/conversations/{conversation_id}")).json()
        surface = (await client.get(f"/v1/events/{conversation['event_id']}")).json()
        files_section = next(
            section for section in surface["surface"]["sections"]
            if section["kind"] == "files"
        )
        assert all(file["id"] != file_id for file in files_section["files"])

        from aloy_backend.models import KnowledgeEntry

        async with db_session_maker() as session:
            assert await session.get(KnowledgeEntry, library_entry_id(file_id)) is None

    async def test_delete_rejects_generated_artifacts(self, client, db_session_maker):
        _, file_id = await _upload(client)
        from aloy_backend.models import StoredFile

        async with db_session_maker() as session:
            record = await session.get(StoredFile, file_id)
            assert record is not None
            record.kind = "artifact"
            session.add(record)
            await session.commit()

        deleted = await client.delete(f"/v1/files/{file_id}")
        assert deleted.status_code == 409
        assert (await client.get(f"/v1/files/{file_id}")).status_code == 200


class TestFetchMyFileTool:
    def _manifest_for(self, name: str, data: bytes) -> list[dict]:
        import hashlib
        import io
        import uuid
        from datetime import datetime, timezone
        from types import SimpleNamespace

        from aloy_backend.storage import upload_key

        rec = SimpleNamespace(
            id=uuid.uuid4().hex,
            name=name,
            size_bytes=len(data),
            content_type="application/pdf",
            sha256=hashlib.sha256(data).hexdigest(),
            storage_key="",
            created_at=datetime.now(timezone.utc),
            in_library=True,
        )
        rec.storage_key = upload_key("org-1", "conv-lib", rec.id, name)
        storage_mod.get_object_store().put(
            rec.storage_key, io.BytesIO(data), content_type=rec.content_type
        )
        return library_manifest([rec])

    def test_fetch_materializes_into_current_thread(self, tmp_path):
        manifest = self._manifest_for("cv.pdf", b"%PDF-my-cv")
        # A DIFFERENT conversation than the one that uploaded it — the point.
        context = {
            "library_files": manifest,
            "workspace_id": "event-library",
            "run_id": "run-fetch",
        }
        result = fetch_my_file_tool(FetchMyFileParams(name="cv.pdf"), context)
        assert result.get("path") == "/mnt/user-data/uploads/cv.pdf"
        on_disk = (
            tmp_path
            / "sandbox"
            / "events"
            / "event-library"
            / "user-data"
            / "uploads"
            / "cv.pdf"
        )
        assert on_disk.read_bytes() == b"%PDF-my-cv"

    def test_partial_name_matches_when_unambiguous(self, tmp_path):
        manifest = self._manifest_for("cv_2026_final.pdf", b"%PDF")
        context = {
            "library_files": manifest,
            "workspace_id": "event-library",
            "run_id": "run-fetch",
        }
        result = fetch_my_file_tool(FetchMyFileParams(name="cv"), context)
        assert result.get("path", "").endswith("cv_2026_final.pdf")

    def test_miss_lists_available_files(self, tmp_path):
        manifest = self._manifest_for("cv.pdf", b"%PDF")
        context = {"library_files": manifest, "thread_id": "t3"}
        result = fetch_my_file_tool(FetchMyFileParams(name="passport.jpg"), context)
        assert "error" in result
        assert result["available_files"] == ["cv.pdf"]

    def test_empty_library_reports_cleanly(self):
        result = fetch_my_file_tool(
            FetchMyFileParams(name="cv.pdf"), {"library_files": [], "thread_id": "t"}
        )
        assert "empty" in result["error"]
