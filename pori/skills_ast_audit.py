"""Opt-in AST audit for skill / plugin Python (SK-7).

A stdlib-``ast`` diagnostic that flags dynamic or risky code patterns in a skill's
``scripts/*.py`` (or a plugin's code) for **human review**. It is explicitly *not
a security boundary* — a determined author can evade any static scan; real
isolation is the sandbox (``pori/sandbox/``). Use it as fast triage when reviewing
auto-authored (SK-1) or third-party (SK-3) code, not as a gate.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

_DANGEROUS_CALLS = {
    "eval": "dynamic code evaluation",
    "exec": "dynamic code execution",
    "compile": "dynamic code compilation",
    "__import__": "dynamic import",
    "system": "shell execution (os.system)",
    "popen": "process spawn",
}
_DANGEROUS_MODULES = {
    "subprocess": "process execution",
    "pickle": "arbitrary deserialization",
    "marshal": "arbitrary deserialization",
    "ctypes": "native memory access",
    "socket": "raw network access",
}


@dataclass(frozen=True)
class AuditFinding:
    line: int
    category: str
    message: str


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.findings: List[AuditFinding] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in _DANGEROUS_MODULES:
                self.findings.append(
                    AuditFinding(
                        node.lineno,
                        "import",
                        f"imports {alias.name} ({_DANGEROUS_MODULES[root]})",
                    )
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".")[0]
        if root in _DANGEROUS_MODULES:
            self.findings.append(
                AuditFinding(
                    node.lineno,
                    "import",
                    f"imports from {node.module} ({_DANGEROUS_MODULES[root]})",
                )
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in _DANGEROUS_CALLS:
            self.findings.append(
                AuditFinding(
                    node.lineno, "call", f"calls {name}() — {_DANGEROUS_CALLS[name]}"
                )
            )
        if name in {"getattr", "setattr", "delattr"} and len(node.args) >= 2:
            if not isinstance(node.args[1], ast.Constant):
                self.findings.append(
                    AuditFinding(
                        node.lineno,
                        "dynamic-attr",
                        f"{name}() with a computed attribute name",
                    )
                )
        self.generic_visit(node)


def audit_source(source: str, filename: str = "<skill>") -> List[AuditFinding]:
    """Return review findings (hints) for a Python source string."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [AuditFinding(exc.lineno or 0, "syntax", f"could not parse: {exc.msg}")]
    visitor = _Visitor()
    visitor.visit(tree)
    return sorted(visitor.findings, key=lambda f: (f.line, f.category, f.message))


def audit_path(path: Union[str, Path]) -> List[AuditFinding]:
    """Audit a ``.py`` file, or every ``*.py`` under a skill/plugin directory."""
    root = Path(path)
    files: List[Path] = []
    if root.is_dir():
        files = sorted(root.rglob("*.py"))
    elif root.suffix == ".py":
        files = [root]
    findings: List[AuditFinding] = []
    for file in files:
        try:
            findings.extend(audit_source(file.read_text(encoding="utf-8"), str(file)))
        except OSError:
            continue
    return findings
