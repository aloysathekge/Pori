"""Execution-receipt + tool-artifact tracking. These are `Agent` methods, grouped
here for readability and bound onto the class in `core` (they take `self`).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..runtime import ReceiptStatus, ToolExecutionReceipt, stable_fingerprint, utc_now


def _record_tool_receipt(
    self,
    tool_name: str,
    params: Dict[str, Any],
    status: ReceiptStatus,
    *,
    started_at: Optional[datetime] = None,
    duration_seconds: float = 0.0,
    error: Optional[str] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ToolExecutionReceipt:
    receipt_artifacts = [dict(artifact) for artifact in artifacts or []]
    receipt = ToolExecutionReceipt(
        run_id=self.run_context.run_id,
        tool_name=tool_name,
        status=status,
        parameters_fingerprint=stable_fingerprint(params),
        started_at=started_at or utc_now(),
        finished_at=utc_now(),
        duration_seconds=max(0.0, duration_seconds),
        error=error,
        artifacts=receipt_artifacts,
        metadata=metadata or {},
    )
    for artifact in receipt.artifacts:
        artifact.setdefault("receipt_id", receipt.receipt_id)
    self.execution_receipts.append(receipt)
    return receipt


def _extract_tool_artifacts(
    self, tool_name: str, params: Dict[str, Any], tool_result: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Extract user-visible artifacts from successful tool results."""
    if not tool_result.get("success"):
        return []
    result = tool_result.get("result")
    if not isinstance(result, dict):
        return []
    # bash produces files by running commands; the sandbox observes which
    # files changed and reports them under "files_written".
    if tool_name == "bash":
        bash_artifacts: List[Dict[str, Any]] = []
        for entry in result.get("files_written") or []:
            if not isinstance(entry, dict):
                continue
            bash_art: Dict[str, Any] = {
                "kind": "file",
                "tool_name": tool_name,
                "path": entry.get("path") or "(path unavailable)",
                "operation": entry.get("operation", "write"),
            }
            bytes_written = entry.get("bytes_written")
            if isinstance(bytes_written, int):
                bash_art["bytes_written"] = bytes_written
            bash_artifacts.append(bash_art)
        return bash_artifacts
    if tool_name not in {"write_file", "sandbox_write_file"}:
        return []
    path = (
        params.get("file_path")
        or params.get("path")
        or result.get("path")
        or result.get("file_path")
    )
    file_info = result.get("file_info")
    if not path and isinstance(file_info, dict):
        path = file_info.get("path")
    artifact: Dict[str, Any] = {
        "kind": "file",
        "tool_name": tool_name,
        "path": path or "(path unavailable)",
        "operation": "append" if bool(params.get("append")) else "write",
    }
    bytes_written = result.get("bytes_written")
    if isinstance(bytes_written, int):
        artifact["bytes_written"] = bytes_written
    return [artifact]


def _run_artifacts(self) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    for receipt in self.execution_receipts:
        artifacts.extend(receipt.artifacts)
    return artifacts


def _runtime_fact_summary(self) -> Dict[str, Any]:
    """Return runtime-owned facts the model may cite but not invent."""
    return {
        "artifacts_written": self._run_artifacts(),
        "tool_receipts": [
            {
                "receipt_id": receipt.receipt_id,
                "tool_name": receipt.tool_name,
                "status": receipt.status.value,
                "artifacts": receipt.artifacts,
                "error": receipt.error,
            }
            for receipt in self.execution_receipts[-10:]
        ],
        "selected_skills": [skill.manifest.skill_id for skill in self.selected_skills],
        "final_answer_present": self.memory.get_state("final_answer") is not None,
    }


def _artifact_reference_errors(self, references: Any) -> List[str]:
    """Validate final-answer artifact references against receipt evidence."""
    if references in (None, "", []):
        return []
    if not isinstance(references, list):
        return ["artifact_references must be a list when provided."]

    artifacts = self._run_artifacts()
    by_path = {
        str(artifact.get("path", "")).strip().lower(): artifact
        for artifact in artifacts
        if artifact.get("path")
    }
    by_receipt: Dict[str, List[Dict[str, Any]]] = {}
    for artifact in artifacts:
        receipt_id = str(artifact.get("receipt_id", "")).strip()
        if receipt_id:
            by_receipt.setdefault(receipt_id, []).append(artifact)

    errors: List[str] = []
    for index, reference in enumerate(references, start=1):
        if not isinstance(reference, dict):
            errors.append(f"artifact_references[{index}] must be an object.")
            continue
        path = str(reference.get("path") or "").strip()
        receipt_id = str(reference.get("receipt_id") or "").strip()
        if not path and not receipt_id:
            errors.append(
                f"artifact_references[{index}] must include a path or receipt_id."
            )
            continue
        receipt_group = None
        if receipt_id:
            receipt_group = by_receipt.get(receipt_id)
            if not receipt_group:
                errors.append(
                    f"artifact_references[{index}] receipt_id '{receipt_id}' "
                    "does not match a successful artifact receipt."
                )
                continue
        if path:
            path_match = by_path.get(path.lower())
            if path_match is None:
                errors.append(
                    f"artifact_references[{index}] path '{path}' was not "
                    "written in this run."
                )
                continue
            if receipt_group is not None and path_match not in receipt_group:
                errors.append(
                    f"artifact_references[{index}] path '{path}' does not "
                    f"belong to receipt_id '{receipt_id}'."
                )
                continue
    return errors
