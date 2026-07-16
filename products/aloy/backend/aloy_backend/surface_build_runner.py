"""Fail-closed validation and isolated execution for Surface builds."""

from __future__ import annotations

import asyncio
import base64
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pori import LocalSandboxProvider, SandboxProvider, get_sandbox_provider

from .surface_manifest import SurfaceManifest

SURFACE_TOOLCHAIN_VERSION = "aloy-surface-toolchain@1"
MAX_SURFACE_BUNDLE_BYTES = 2 * 1024 * 1024
MAX_SURFACE_BUILD_LOG_CHARS = 100_000
_FIXED_BUILD_COMMAND = (
    "timeout 60s /opt/aloy-surface-toolchain/bin/build-surface "
    "--source /workspace --output /output"
)

_ALLOWED_IMPORTS = {
    "@aloy/surface",
    "react",
    "react-dom/client",
    "react/jsx-runtime",
}
_IMPORT_PATTERN = re.compile(
    r"(?:\bfrom\s*|\bimport\s*)[\"']([^\"']+)[\"']",
    re.MULTILINE,
)
_FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "dynamic_import",
        re.compile(r"\bimport\s*\("),
        "Dynamic imports are not allowed in generated Surfaces",
    ),
    (
        "dynamic_code",
        re.compile(r"\b(?:eval|Function)\s*\("),
        "Dynamic code evaluation is not allowed",
    ),
    (
        "direct_network",
        re.compile(r"\b(?:fetch|XMLHttpRequest|WebSocket|EventSource)\b"),
        "Direct network APIs are unavailable; use declared Surface SDK intents",
    ),
    (
        "ambient_storage",
        re.compile(r"\b(?:localStorage|sessionStorage|indexedDB)\b"),
        "Ambient browser storage is unavailable",
    ),
    (
        "host_escape",
        re.compile(
            r"\b(?:window|globalThis|self)\.(?:parent|top|opener)\b"
            r"|\b(?:parent|top|opener)\.(?:document|location|postMessage)\b"
        ),
        "Generated code cannot access the host browsing context",
    ),
    (
        "cookie_access",
        re.compile(r"\bdocument\.cookie\b"),
        "Generated code cannot access cookies",
    ),
    (
        "unsafe_html",
        re.compile(r"\bdangerouslySetInnerHTML\b"),
        "Raw HTML injection is not allowed",
    ),
    (
        "commonjs_require",
        re.compile(r"\brequire\s*\("),
        "CommonJS require is not supported by the fixed Surface toolchain",
    ),
)


def _line_number(content: str, offset: int) -> int:
    return content.count("\n", 0, offset) + 1


def _diagnostic(
    code: str,
    message: str,
    *,
    path: str | None = None,
    line: int | None = None,
    severity: Literal["error", "warning"] = "error",
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "path": path,
        "line": line,
    }


def validate_surface_source(
    files: dict[str, str],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return deterministic diagnostics without executing generated source."""
    diagnostics: list[dict[str, Any]] = []
    try:
        SurfaceManifest.model_validate(manifest)
    except ValueError as exc:
        diagnostics.append(
            _diagnostic("invalid_manifest", f"Invalid Surface manifest: {exc}")
        )
    entrypoint = str(manifest.get("entrypoint") or "")
    if entrypoint != "/src/App.tsx":
        diagnostics.append(
            _diagnostic(
                "invalid_entrypoint",
                "The fixed Surface entrypoint must be /src/App.tsx",
            )
        )
    if entrypoint not in files:
        diagnostics.append(
            _diagnostic(
                "missing_entrypoint",
                "Surface source must include /src/App.tsx",
                path=entrypoint or None,
            )
        )
    if manifest.get("sdk_version") != "1":
        diagnostics.append(
            _diagnostic(
                "unsupported_sdk_version",
                "Surface SDK version must be 1",
            )
        )

    for path in sorted(files):
        content = files[path]
        if path.endswith(".json"):
            try:
                json.loads(content)
            except ValueError as exc:
                diagnostics.append(
                    _diagnostic(
                        "invalid_json",
                        f"Invalid JSON: {exc}",
                        path=path,
                    )
                )
        if path.endswith((".js", ".jsx", ".ts", ".tsx")):
            for match in _IMPORT_PATTERN.finditer(content):
                dependency = match.group(1)
                if dependency.startswith(".") or dependency in _ALLOWED_IMPORTS:
                    continue
                diagnostics.append(
                    _diagnostic(
                        "undeclared_import",
                        f"Import is not provided by the Surface SDK: {dependency}",
                        path=path,
                        line=_line_number(content, match.start()),
                    )
                )
            for code, pattern, message in _FORBIDDEN_PATTERNS:
                for match in pattern.finditer(content):
                    diagnostics.append(
                        _diagnostic(
                            code,
                            message,
                            path=path,
                            line=_line_number(content, match.start()),
                        )
                    )
        if path.endswith(".css"):
            for match in re.finditer(r"@import\s+(?:url\()?[^;]+", content):
                diagnostics.append(
                    _diagnostic(
                        "css_import",
                        "CSS imports are not allowed; include styles in the project",
                        path=path,
                        line=_line_number(content, match.start()),
                    )
                )
            for match in re.finditer(r"url\(\s*[\"']?(?:https?:|//)", content):
                diagnostics.append(
                    _diagnostic(
                        "external_asset",
                        "External CSS assets are not allowed",
                        path=path,
                        line=_line_number(content, match.start()),
                    )
                )
    return diagnostics


@dataclass(frozen=True)
class SurfaceBuildRunnerResult:
    status: Literal["succeeded", "failed", "blocked"]
    bundle: bytes | None = None
    build_log: str = ""
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    preview_artifacts: list[dict[str, Any]] = field(default_factory=list)
    resource_metrics: dict[str, Any] = field(default_factory=dict)


class SurfaceBuildRunner(Protocol):
    toolchain_version: str

    async def build(
        self,
        *,
        build_id: str,
        files: dict[str, str],
        manifest: dict[str, Any],
    ) -> SurfaceBuildRunnerResult: ...


class UnavailableSurfaceBuildRunner:
    toolchain_version = SURFACE_TOOLCHAIN_VERSION

    async def build(
        self,
        *,
        build_id: str,
        files: dict[str, str],
        manifest: dict[str, Any],
    ) -> SurfaceBuildRunnerResult:
        del build_id, files, manifest
        return SurfaceBuildRunnerResult(
            status="blocked",
            diagnostics=[
                _diagnostic(
                    "isolated_builder_unavailable",
                    "No isolated Surface build provider is configured",
                )
            ],
        )


class SandboxSurfaceBuildRunner:
    """Run only the fixed Aloy toolchain inside a non-local sandbox provider."""

    toolchain_version = SURFACE_TOOLCHAIN_VERSION

    def __init__(self, provider: SandboxProvider):
        if isinstance(provider, LocalSandboxProvider):
            raise ValueError("Local host subprocesses cannot build generated Surfaces")
        self._provider = provider

    async def build(
        self,
        *,
        build_id: str,
        files: dict[str, str],
        manifest: dict[str, Any],
    ) -> SurfaceBuildRunnerResult:
        return await asyncio.to_thread(
            self._build_sync,
            build_id=build_id,
            files=files,
            manifest=manifest,
        )

    def _build_sync(
        self,
        *,
        build_id: str,
        files: dict[str, str],
        manifest: dict[str, Any],
    ) -> SurfaceBuildRunnerResult:
        sandbox_id = self._provider.acquire(f"surface-build:{build_id}")
        sandbox = self._provider.get(sandbox_id)
        if sandbox is None:
            return SurfaceBuildRunnerResult(
                status="blocked",
                diagnostics=[
                    _diagnostic(
                        "isolated_builder_unavailable",
                        "The isolated build sandbox could not be acquired",
                    )
                ],
            )
        try:
            for path, content in files.items():
                sandbox.write_file("/workspace" + path, content)
            sandbox.write_file(
                "/workspace/.aloy-build.json",
                json.dumps(
                    {
                        "toolchain_version": self.toolchain_version,
                        "manifest": manifest,
                    },
                    sort_keys=True,
                ),
            )
            output = sandbox.execute_command(_FIXED_BUILD_COMMAND, cwd="/workspace")
            log = output[-MAX_SURFACE_BUILD_LOG_CHARS:]
            if output.startswith("Error:") or " exit_code=" in output:
                return SurfaceBuildRunnerResult(
                    status="failed",
                    build_log=log,
                    diagnostics=[
                        _diagnostic(
                            "compiler_failed",
                            "The isolated Surface compiler reported an error",
                        )
                    ],
                )
            encoded = sandbox.read_file("/output/bundle.zip.b64").strip()
            bundle = base64.b64decode(encoded, validate=True)
            metadata: dict[str, Any] = {}
            try:
                metadata = json.loads(sandbox.read_file("/output/metadata.json"))
            except (OSError, ValueError):
                metadata = {}
            return SurfaceBuildRunnerResult(
                status="succeeded",
                bundle=bundle,
                build_log=log,
                preview_artifacts=list(metadata.get("preview_artifacts") or []),
                resource_metrics=dict(metadata.get("resource_metrics") or {}),
            )
        except Exception as exc:
            return SurfaceBuildRunnerResult(
                status="failed",
                diagnostics=[
                    _diagnostic(
                        "builder_protocol_error",
                        f"The isolated builder did not satisfy its contract: {exc}",
                    )
                ],
            )
        finally:
            self._provider.release(sandbox_id)


def configured_surface_build_runner() -> SurfaceBuildRunner:
    provider = get_sandbox_provider()
    if provider is None or isinstance(provider, LocalSandboxProvider):
        return UnavailableSurfaceBuildRunner()
    return SandboxSurfaceBuildRunner(provider)


__all__ = [
    "MAX_SURFACE_BUILD_LOG_CHARS",
    "MAX_SURFACE_BUNDLE_BYTES",
    "SURFACE_TOOLCHAIN_VERSION",
    "SandboxSurfaceBuildRunner",
    "SurfaceBuildRunner",
    "SurfaceBuildRunnerResult",
    "UnavailableSurfaceBuildRunner",
    "configured_surface_build_runner",
    "validate_surface_source",
]
