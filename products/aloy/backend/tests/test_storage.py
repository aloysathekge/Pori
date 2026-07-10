"""The object-store seam: roundtrip, atomicity, containment, name safety."""

import io

import pytest

from aloy_backend.storage import LocalDiskObjectStore, artifact_key, safe_name


def test_put_open_roundtrip(tmp_path):
    store = LocalDiskObjectStore(str(tmp_path))
    key = artifact_key("org1", "conv1", "f1", "report.md")
    written = store.put(key, io.BytesIO(b"hello world"), content_type="text/markdown")
    assert written == 11
    with store.open(key) as fh:
        assert fh.read() == b"hello world"


def test_no_partial_files_left_behind(tmp_path):
    store = LocalDiskObjectStore(str(tmp_path))

    class Boom(io.RawIOBase):
        def read(self, n=-1):
            raise RuntimeError("mid-stream failure")

    key = artifact_key("org1", "conv1", "f2", "big.bin")
    with pytest.raises(RuntimeError):
        store.put(key, Boom(), content_type="application/octet-stream")
    # Neither the final key nor a torn .part temp file exists.
    assert not list(tmp_path.rglob("big.bin"))
    assert not list(tmp_path.rglob("*.part"))


def test_delete_is_idempotent(tmp_path):
    store = LocalDiskObjectStore(str(tmp_path))
    key = artifact_key("org1", "conv1", "f3", "x.txt")
    store.put(key, io.BytesIO(b"x"), content_type="text/plain")
    store.delete(key)
    store.delete(key)  # second delete: no error
    with pytest.raises(FileNotFoundError):
        store.open(key)


def test_key_escape_is_rejected(tmp_path):
    store = LocalDiskObjectStore(str(tmp_path))
    with pytest.raises(ValueError):
        store.put(
            "org/../../../etc/passwd", io.BytesIO(b"x"), content_type="text/plain"
        )


def test_personal_org_ids_make_valid_keys(tmp_path):
    """Personal orgs are 'user:<uuid>' — ':' is illegal in Windows paths
    (WinError 123, the exact live failure). Keys must sanitize segments."""
    store = LocalDiskObjectStore(str(tmp_path))
    key = artifact_key(
        "user:bb22cd19-63a0-4b96-81f8-b8a35febd3d7", "conv1", "f9", "hello.py"
    )
    assert ":" not in key
    store.put(key, io.BytesIO(b"print()"), content_type="text/x-python")
    with store.open(key) as fh:
        assert fh.read() == b"print()"


def test_safe_name_strips_separators_and_traversal():
    assert safe_name("../../etc/passwd") == "passwd"
    assert safe_name("dir\\evil.exe") == "evil.exe"
    assert safe_name("...hidden") == "hidden"
    assert safe_name("") == "file"
    assert "/" not in safe_name("a/b/c.txt")
