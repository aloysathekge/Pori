"""Portable capability metadata and eligibility contracts."""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


@dataclass(frozen=True)
class CapabilityPrerequisites:
    """Runtime requirements shared by a capability group."""

    environment: Tuple[str, ...] = ()
    modules: Tuple[str, ...] = ()
    platforms: Tuple[str, ...] = ()

    def missing(
        self,
        *,
        environ: Optional[Mapping[str, str]] = None,
        platform: Optional[str] = None,
    ) -> Tuple[str, ...]:
        values = os.environ if environ is None else environ
        current_platform = platform or sys.platform
        missing = [
            f"environment:{name}" for name in self.environment if not values.get(name)
        ]
        missing.extend(
            f"module:{name}" for name in self.modules if not _module_available(name)
        )
        if self.platforms and current_platform not in self.platforms:
            missing.append(f"platform:{current_platform}")
        return tuple(missing)


@dataclass(frozen=True)
class CapabilityGroup:
    """An immutable, explainable collection of related tools."""

    name: str
    description: str
    tool_names: frozenset[str]
    prerequisites: CapabilityPrerequisites = field(
        default_factory=CapabilityPrerequisites
    )
    protected: bool = False
    max_output_chars: Optional[int] = None


@dataclass(frozen=True)
class EligibilityReport:
    eligible: bool
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillEligibility:
    """Metadata used before any full skill instructions enter context."""

    required_tools: frozenset[str] = frozenset()
    required_credentials: Tuple[str, ...] = ()
    required_platforms: Tuple[str, ...] = ()
    required_model_capabilities: frozenset[str] = frozenset()
    version: str = "1"
    source: str = "local"
    sensitivity: str = "internal"

    def evaluate(
        self,
        *,
        available_tools: frozenset[str],
        model_capabilities: frozenset[str] = frozenset(),
        environ: Optional[Mapping[str, str]] = None,
        platform: Optional[str] = None,
    ) -> EligibilityReport:
        values = os.environ if environ is None else environ
        current_platform = platform or sys.platform
        reasons = [
            f"missing_tool:{name}"
            for name in sorted(self.required_tools - available_tools)
        ]
        reasons.extend(
            f"missing_credential:{name}"
            for name in self.required_credentials
            if not values.get(name)
        )
        reasons.extend(
            f"missing_model_capability:{name}"
            for name in sorted(self.required_model_capabilities - model_capabilities)
        )
        if self.required_platforms and current_platform not in self.required_platforms:
            reasons.append(f"unsupported_platform:{current_platform}")
        return EligibilityReport(eligible=not reasons, reasons=tuple(reasons))


__all__ = [
    "CapabilityGroup",
    "CapabilityPrerequisites",
    "EligibilityReport",
    "SkillEligibility",
]
