"""S3ObjectStore against a faithful fake client + retention on delete."""

import io

import pytest

import aloy_backend.config as config_mod
import aloy_backend.storage as storage_mod
from aloy_backend.storage import S3ObjectStore

pytestmark = pytest.mark.asyncio


class _NoSuchKey(Exception):
    pass


class FakeS3Client:
    """The four boto3 s3 calls S3ObjectStore uses, over a dict."""

    def __init__(self):
        self.blobs: dict[str, bytes] = {}
        self.content_types: dict[str, str] = {}

    def upload_fileobj(self, fileobj, bucket, key, ExtraArgs=None):
        self.blobs[f"{bucket}/{key}"] = fileobj.read()
        self.content_types[f"{bucket}/{key}"] = (ExtraArgs or {}).get("ContentType", "")

    def get_object(self, Bucket, Key):
        blob = self.blobs.get(f"{Bucket}/{Key}")
        if blob is None:
            raise _NoSuchKey(Key)
        body = io.BytesIO(blob)
        return {"Body": body}

    def delete_object(self, Bucket, Key):
        self.blobs.pop(f"{Bucket}/{Key}", None)

    def generate_presigned_url(self, op, Params, ExpiresIn):
        return (
            f"https://signed.example/{Params['Bucket']}/{Params['Key']}?exp={ExpiresIn}"
        )


# Make the fake's miss look like botocore's (duck-typed by class name).
_NoSuchKey.__name__ = "NoSuchKey"


class TestS3ObjectStore:
    def _store(self):
        client = FakeS3Client()
        return S3ObjectStore(bucket="test-bucket", client=client), client

    def test_put_open_roundtrip_with_content_type(self):
        store, client = self._store()
        n = store.put(
            "org/o/conv/c/uploads/f/cv.pdf",
            io.BytesIO(b"%PDF"),
            content_type="application/pdf",
        )
        assert n == 4
        with store.open("org/o/conv/c/uploads/f/cv.pdf") as fh:
            assert fh.read() == b"%PDF"
        assert client.content_types["test-bucket/org/o/conv/c/uploads/f/cv.pdf"] == (
            "application/pdf"
        )

    def test_missing_key_raises_filenotfound(self):
        store, _ = self._store()
        with pytest.raises(FileNotFoundError):
            store.open("nope")

    def test_delete_then_open_misses(self):
        store, _ = self._store()
        store.put("k", io.BytesIO(b"x"), content_type="text/plain")
        store.delete("k")
        with pytest.raises(FileNotFoundError):
            store.open("k")

    def test_presigned_url_carries_expiry(self):
        store, _ = self._store()
        url = store.url("k", expires_s=120)
        assert url == "https://signed.example/test-bucket/k?exp=120"


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config_mod.settings, "sandbox_base_dir", str(tmp_path / "sandbox")
    )
    monkeypatch.setattr(config_mod.settings, "storage_dir", str(tmp_path / "storage"))
    monkeypatch.setattr(storage_mod, "_STORE", None)
    yield
    monkeypatch.setattr(storage_mod, "_STORE", None)


class TestRetentionOnConversationDelete:
    async def _upload(self, client, conv_id, name):
        up = await client.post(
            f"/v1/conversations/{conv_id}/files",
            files={"file": (name, b"data-" + name.encode(), "text/plain")},
        )
        return up.json()["file_id"]

    async def test_event_files_survive_session_deletion(self, client):
        created = await client.post("/v1/conversations", json={"title": "r"})
        conv_id = created.json()["id"]
        plain_id = await self._upload(client, conv_id, "scratch.txt")
        saved_id = await self._upload(client, conv_id, "cv.txt")
        await client.post(f"/v1/files/{saved_id}/library")

        resp = await client.delete(f"/v1/conversations/{conv_id}")
        assert resp.status_code == 204

        # Both files belong to the Event; deleting only its Session cannot
        # erase either blob or pointer row.
        plain = await client.get(f"/v1/files/{plain_id}")
        assert plain.status_code == 200
        assert plain.content == b"data-scratch.txt"
        dl = await client.get(f"/v1/files/{saved_id}")
        assert dl.status_code == 200
        assert dl.content == b"data-cv.txt"
        # And it still lists in My Files (its memory pointer stays valid).
        listed = (await client.get("/v1/files")).json()
        assert [f["file_id"] for f in listed] == [saved_id]
