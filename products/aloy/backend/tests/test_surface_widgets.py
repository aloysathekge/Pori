from __future__ import annotations

import pytest

from aloy_backend.surface_manifest import SurfaceManifest
from aloy_backend.surface_widgets import (
    SURFACE_WIDGET_REGISTRY_VERSION,
    surface_widget_definition,
    validate_surface_widgets,
)


def test_registry_contains_only_reviewed_domain_neutral_widgets():
    assert SURFACE_WIDGET_REGISTRY_VERSION == "1"
    assert surface_widget_definition("table") is not None
    assert surface_widget_definition("map") is None


def test_unknown_and_duplicate_widgets_fail_closed():
    with pytest.raises(ValueError, match="Unsupported Surface widget"):
        validate_surface_widgets(["table", "map"])
    with pytest.raises(ValueError, match="must be unique"):
        validate_surface_widgets(["table", "table"])


def test_privileged_widget_requires_its_host_capability():
    with pytest.raises(ValueError, match="requires capabilities: proposals"):
        validate_surface_widgets(["approval"])
    assert validate_surface_widgets(["approval"], ["proposals"]) == ["approval"]


def test_manifest_validates_widget_capabilities_before_persistence():
    with pytest.raises(ValueError, match="requires capabilities: files"):
        SurfaceManifest(widgets=["file_viewer"])
    manifest = SurfaceManifest(capabilities=["files"], widgets=["file_viewer"])
    assert manifest.widgets == ["file_viewer"]
