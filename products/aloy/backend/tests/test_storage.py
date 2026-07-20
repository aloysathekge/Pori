"""The object-store seam: roundtrip, atomicity, containment, name safety."""

import io

import pytest

from aloy_backend.storage import (
    LocalDiskObjectStore,
    _filesystem_path,
    artifact_key,
    safe_name,
    surface_bundle_key,
    surface_preview_artifact_key,
)


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


def test_long_surface_bundle_key_roundtrips_without_changing_object_key(tmp_path):
    """Real Surface keys exceed legacy MAX_PATH on Windows local development."""
    store = LocalDiskObjectStore(str(tmp_path))
    key = surface_bundle_key(
        "user:bb22cd19-63a0-4b96-81f8-b8a35febd3d7",
        "57231f85d37045f192f9878cae43c714",
        "sbuild_19c82486ac84400fb08b2f695d7cb269",
        "a" * 64,
    )
    target = store._path(key)
    assert len(str(target)) > 260

    store.put(key, io.BytesIO(b"surface bundle"), content_type="application/zip")
    with store.open(key) as fh:
        assert fh.read() == b"surface bundle"
    store.delete(key)
    with pytest.raises(FileNotFoundError):
        store.open(key)


def test_surface_preview_artifact_key_is_tenant_scoped_and_name_safe():
    key = surface_preview_artifact_key(
        "user:person-1",
        "event/unsafe",
        "build:1",
        "capture:sha",
        "../../mobile.png",
    )

    assert key == (
        "org/user_person-1/events/event_unsafe/surface-builds/build_1/"
        "previews/capture_sha/mobile.png"
    )


def test_windows_filesystem_path_uses_extended_length_prefix(tmp_path):
    extended = _filesystem_path(tmp_path / "bundle.zip", windows=True)
    assert extended.startswith("\\\\?\\")
    assert extended.endswith("bundle.zip")


def test_safe_name_strips_separators_and_traversal():
    assert safe_name("../../etc/passwd") == "passwd"
    assert safe_name("dir\\evil.exe") == "evil.exe"
    assert safe_name("...hidden") == "hidden"
    assert safe_name("") == "file"
    assert "/" not in safe_name("a/b/c.txt")
