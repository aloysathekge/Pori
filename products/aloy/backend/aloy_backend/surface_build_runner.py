"""Fail-closed validation and isolated execution for Surface builds."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from pori import LocalSandboxProvider, SandboxProvider, get_sandbox_provider

from .surface_manifest import SurfaceManifest

SURFACE_TOOLCHAIN_VERSION = "aloy-surface-toolchain@2"
MAX_SURFACE_BUNDLE_BYTES = 2 * 1024 * 1024
MAX_SURFACE_BUILD_LOG_CHARS = 100_000
_FIXED_BUILD_COMMAND = (
    "timeout 60s /opt/aloy-surface-toolchain/bin/build-surface "
    "--typecheck --source /workspace --output /output"
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
        "direct_bridge",
        re.compile(r"\bwindow\.postMessage\s*\("),
        "Generated code cannot implement its own host bridge; use @aloy/surface",
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
    (
        "node_global",
        re.compile(r"\b(?:process|Buffer|__dirname|__filename)\b"),
        "Node.js globals are unavailable in browser Surfaces",
    ),
    (
        "swallowed_surface_failure",
        re.compile(
            r"\.catch\s*\(\s*(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>\s*"
            r"(?:undefined|\{\s*\})\s*\)"
        ),
        "Surface SDK failures must produce visible error state; do not swallow them",
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


def _typescript_diagnostics(log: str, *, source: Path) -> list[dict[str, Any]]:
    """Convert host TypeScript output into bounded model-repair diagnostics."""
    pattern = re.compile(
        r"^(?P<path>.+)\((?P<line>\d+),(?P<column>\d+)\): "
        r"error (?P<code>TS\d+): (?P<message>.*)$",
        re.MULTILINE,
    )
    diagnostics: list[dict[str, Any]] = []
    for match in pattern.finditer(log):
        raw_path = Path(match.group("path"))
        candidate = raw_path if raw_path.is_absolute() else source.parent / raw_path
        surface_path: str | None = None
        try:
            surface_path = "/" + candidate.resolve().relative_to(source).as_posix()
        except ValueError:
            surface_path = None
        diagnostics.append(
            _diagnostic(
                "typescript_contract_error",
                f"{match.group('code')}: {match.group('message')}",
                path=surface_path,
                line=int(match.group("line")),
            )
        )
        if len(diagnostics) >= 100:
            break
    return diagnostics or [
        _diagnostic(
            "typescript_contract_error",
            "The host TypeScript contract check rejected the generated source",
        )
    ]


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

    imports_surface_sdk = False
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
                if dependency == "@aloy/surface":
                    imports_surface_sdk = True
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
    if (
        manifest.get("capabilities")
        or manifest.get("intents")
        or manifest.get("widgets")
    ) and not imports_surface_sdk:
        diagnostics.append(
            _diagnostic(
                "missing_surface_sdk",
                "Interactive Surfaces must use the host-provided @aloy/surface SDK",
            )
        )
    intents = set(dict(manifest.get("intents") or {}))
    covered = {
        str(dict(check.get("expect") or {}).get("name") or "")
        for check in list(manifest.get("interaction_checks") or [])
        if isinstance(check, dict)
    }
    for name in sorted(intents - covered):
        diagnostics.append(
            _diagnostic(
                "missing_interaction_check",
                f"Declared intent {name!r} has no executable accessible UI check",
                path="/surface.json",
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


class LocalDevelopmentSurfaceBuildRunner:
    """Compile with Aloy's pinned host toolchain for local development only.

    Generated source supplies no commands, package manifest, plugins, or build
    configuration. The host creates an ephemeral project and invokes one fixed
    Vite entrypoint. Production must use ``SandboxSurfaceBuildRunner``.
    """

    toolchain_version = SURFACE_TOOLCHAIN_VERSION + "+local-dev"

    def __init__(self, *, repository_root: Path | None = None) -> None:
        self._repository_root = repository_root or Path(__file__).resolve().parents[4]

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

    def _toolchain_paths(self) -> dict[str, Path] | None:
        app_modules = (
            self._repository_root / "products" / "aloy" / "app" / "node_modules"
        )
        paths = {
            "vite": app_modules / "vite" / "bin" / "vite.js",
            "react": app_modules / "react" / "index.js",
            "react_jsx": app_modules / "react" / "jsx-runtime.js",
            "react_dom": app_modules / "react-dom" / "client.js",
            "sdk": self._repository_root
            / "packages"
            / "aloy-surface"
            / "dist"
            / "index.js",
            "sdk_types": self._repository_root
            / "packages"
            / "aloy-surface"
            / "dist"
            / "index.d.ts",
            "typescript": app_modules / "typescript" / "bin" / "tsc",
            "react_types": app_modules / "@types" / "react" / "index.d.ts",
            "react_jsx_types": app_modules / "@types" / "react" / "jsx-runtime.d.ts",
            "react_dom_types": app_modules / "@types" / "react-dom" / "client.d.ts",
        }
        return paths if all(path.is_file() for path in paths.values()) else None

    def _build_sync(
        self,
        *,
        build_id: str,
        files: dict[str, str],
        manifest: dict[str, Any],
    ) -> SurfaceBuildRunnerResult:
        del manifest
        node = shutil.which("node")
        toolchain = self._toolchain_paths()
        if node is None or toolchain is None:
            return SurfaceBuildRunnerResult(
                status="blocked",
                diagnostics=[
                    _diagnostic(
                        "local_toolchain_unavailable",
                        "The pinned local Surface toolchain is unavailable; run the Aloy workspace install",
                    )
                ],
            )
        started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(
                prefix=f"aloy-surface-{build_id[:16]}-"
            ) as temp:
                root = Path(temp).resolve()
                source = root / "source"
                output = root / "output"
                source.mkdir()
                output.mkdir()
                for path, content in files.items():
                    target = (source / path.lstrip("/")).resolve()
                    if source not in target.parents:
                        raise ValueError(
                            f"Surface source escapes local build root: {path}"
                        )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")

                css_imports = "\n".join(
                    f'import "./{path.removeprefix("/")}";'
                    for path in sorted(files)
                    if path.endswith(".css")
                )
                entry = source / ".aloy-entry.tsx"
                entry.write_text(
                    "import React from 'react';\n"
                    "import { createRoot } from 'react-dom/client';\n"
                    "import '@aloy/surface';\n"
                    "import App from './src/App.tsx';\n"
                    f"{css_imports}\n"
                    "class AloySurfaceErrorBoundary extends React.Component<"
                    "{ children?: React.ReactNode }, { error: string | null }> {\n"
                    "  state = { error: null as string | null };\n"
                    "  static getDerivedStateFromError(error: unknown) {\n"
                    "    return { error: error instanceof Error ? error.message : 'Surface runtime error' };\n"
                    "  }\n"
                    "  componentDidCatch(error: unknown) {\n"
                    "    (window as unknown as { __aloyRuntimeError?: string }).__aloyRuntimeError = "
                    "error instanceof Error ? error.message : 'Surface runtime error';\n"
                    "    console.error('Aloy Surface runtime error', error);\n"
                    "  }\n"
                    "  render() {\n"
                    "    if (this.state.error) return React.createElement('main', { "
                    "style: { minHeight: '100vh', display: 'grid', placeItems: 'center', "
                    "padding: '2rem', fontFamily: 'system-ui', color: '#3f3f46', background: '#fafafa' } }, "
                    "React.createElement('section', { style: { maxWidth: '28rem', textAlign: 'center' } }, "
                    "React.createElement('h1', { style: { fontSize: '1rem', marginBottom: '.5rem' } }, "
                    "'This Surface needs a safe repair'), React.createElement('p', { style: { fontSize: '.875rem', "
                    "lineHeight: '1.5', color: '#71717a' } }, 'Aloy kept your Event data safe. Reload the Surface or ask Aloy to repair this view.')));\n"
                    "    return this.props.children;\n"
                    "  }\n"
                    "}\n"
                    "createRoot(document.getElementById('root')!).render("
                    "React.createElement(React.StrictMode, null, "
                    "React.createElement(AloySurfaceErrorBoundary, null, React.createElement(App))));\n",
                    encoding="utf-8",
                )
                declarations = source / ".aloy-types.d.ts"
                declarations.write_text(
                    "declare module '*.css' { const value: string; export default value; }\n"
                    "declare module '*.svg' { const value: string; export default value; }\n",
                    encoding="utf-8",
                )
                typecheck_config = root / "tsconfig.surface.json"
                typecheck_config.write_text(
                    json.dumps(
                        {
                            "compilerOptions": {
                                "target": "ES2022",
                                "lib": ["ES2022", "DOM", "DOM.Iterable"],
                                "module": "ESNext",
                                "moduleResolution": "Bundler",
                                "jsx": "react-jsx",
                                "strict": True,
                                "noEmit": True,
                                "allowJs": True,
                                "checkJs": True,
                                "skipLibCheck": True,
                                "allowSyntheticDefaultImports": True,
                                "esModuleInterop": True,
                                "resolveJsonModule": True,
                                "forceConsistentCasingInFileNames": True,
                                "baseUrl": str(source),
                                "paths": {
                                    "@aloy/surface": [str(toolchain["sdk_types"])],
                                    "react": [str(toolchain["react_types"])],
                                    "react/jsx-runtime": [
                                        str(toolchain["react_jsx_types"])
                                    ],
                                    "react-dom/client": [
                                        str(toolchain["react_dom_types"])
                                    ],
                                },
                            },
                            "files": [
                                str(declarations),
                                *[
                                    str((source / path.lstrip("/")).resolve())
                                    for path in sorted(files)
                                    if path.endswith((".js", ".jsx", ".ts", ".tsx"))
                                ],
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                typecheck_started = time.perf_counter()
                try:
                    checked = subprocess.run(
                        [
                            node,
                            str(toolchain["typescript"]),
                            "--project",
                            str(typecheck_config),
                            "--pretty",
                            "false",
                        ],
                        cwd=root,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    return SurfaceBuildRunnerResult(
                        status="failed",
                        diagnostics=[
                            _diagnostic(
                                "typecheck_timeout",
                                "The host TypeScript contract check exceeded 30 seconds",
                            )
                        ],
                    )
                typecheck_ms = (time.perf_counter() - typecheck_started) * 1000
                typecheck_log = (checked.stdout + checked.stderr)[
                    -MAX_SURFACE_BUILD_LOG_CHARS:
                ]
                if checked.returncode != 0:
                    return SurfaceBuildRunnerResult(
                        status="failed",
                        build_log=typecheck_log,
                        diagnostics=_typescript_diagnostics(
                            typecheck_log,
                            source=source,
                        ),
                        resource_metrics={
                            "backend": "local_dev",
                            "typecheck_ms": round(typecheck_ms, 3),
                        },
                    )
                config = root / "vite.config.mjs"
                config.write_text(
                    "export default "
                    + json.dumps(
                        {
                            "root": str(source),
                            # React's CommonJS entrypoint selects its browser
                            # production build through this expression. Vite
                            # must replace it at compile time because generated
                            # iframe code has no Node.js `process` global.
                            "define": {
                                "process.env.NODE_ENV": json.dumps("production")
                            },
                            "resolve": {
                                "alias": {
                                    "@aloy/surface": str(toolchain["sdk"]),
                                    "react/jsx-runtime": str(toolchain["react_jsx"]),
                                    "react-dom/client": str(toolchain["react_dom"]),
                                    "react": str(toolchain["react"]),
                                }
                            },
                            "build": {
                                "outDir": str(output),
                                "emptyOutDir": True,
                                "cssCodeSplit": False,
                                "assetsInlineLimit": 6 * 1024 * 1024,
                                "minify": True,
                                "sourcemap": False,
                                "lib": {
                                    "entry": str(entry),
                                    "formats": ["iife"],
                                    "name": "AloySurface",
                                    "fileName": "surface",
                                },
                                "rollupOptions": {
                                    "output": {
                                        "entryFileNames": "surface.js",
                                        "assetFileNames": "surface.css",
                                    }
                                },
                            },
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                compile_started = time.perf_counter()
                completed = subprocess.run(
                    [
                        node,
                        str(toolchain["vite"]),
                        "build",
                        "--config",
                        str(config),
                        "--logLevel",
                        "info",
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                log = (completed.stdout + completed.stderr)[
                    -MAX_SURFACE_BUILD_LOG_CHARS:
                ]
                compile_ms = (time.perf_counter() - compile_started) * 1000
                if completed.returncode != 0:
                    return SurfaceBuildRunnerResult(
                        status="failed",
                        build_log=log,
                        diagnostics=[
                            _diagnostic(
                                "compiler_failed",
                                "The fixed local Surface compiler reported an error",
                            )
                        ],
                    )
                allowed = {"surface.js", "surface.css"}
                outputs = {
                    path.name: path for path in output.iterdir() if path.is_file()
                }
                if "surface.js" not in outputs or set(outputs) - allowed:
                    raise ValueError(
                        "Local compiler produced an invalid Surface bundle file set"
                    )
                bundle_stream = io.BytesIO()
                with zipfile.ZipFile(
                    bundle_stream,
                    mode="w",
                    compression=zipfile.ZIP_DEFLATED,
                ) as archive:
                    for name in sorted(outputs):
                        archive.writestr(name, outputs[name].read_bytes())
                bundle = bundle_stream.getvalue()
                return SurfaceBuildRunnerResult(
                    status="succeeded",
                    bundle=bundle,
                    build_log=log,
                    resource_metrics={
                        "backend": "local_dev",
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                        "typecheck_ms": round(typecheck_ms, 3),
                        "compile_ms": round(compile_ms, 3),
                        "bundle_bytes": len(bundle),
                    },
                )
        except subprocess.TimeoutExpired:
            return SurfaceBuildRunnerResult(
                status="failed",
                diagnostics=[
                    _diagnostic(
                        "compiler_timeout",
                        "The fixed local Surface compiler exceeded 60 seconds",
                    )
                ],
            )
        except Exception as exc:
            return SurfaceBuildRunnerResult(
                status="failed",
                diagnostics=[
                    _diagnostic(
                        "builder_protocol_error",
                        f"The local developer builder did not satisfy its contract: {exc}",
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
                resource_metrics={
                    **dict(metadata.get("resource_metrics") or {}),
                    "backend": "isolated",
                },
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
    from .config import settings

    if settings.surface_build_backend == "local_dev":
        return LocalDevelopmentSurfaceBuildRunner()
    provider = get_sandbox_provider()
    if provider is None or isinstance(provider, LocalSandboxProvider):
        return UnavailableSurfaceBuildRunner()
    return SandboxSurfaceBuildRunner(provider)


__all__ = [
    "MAX_SURFACE_BUILD_LOG_CHARS",
    "MAX_SURFACE_BUNDLE_BYTES",
    "SURFACE_TOOLCHAIN_VERSION",
    "LocalDevelopmentSurfaceBuildRunner",
    "SandboxSurfaceBuildRunner",
    "SurfaceBuildRunner",
    "SurfaceBuildRunnerResult",
    "UnavailableSurfaceBuildRunner",
    "configured_surface_build_runner",
    "validate_surface_source",
]
