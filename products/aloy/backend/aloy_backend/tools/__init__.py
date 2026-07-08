"""Aloy product tools — capabilities the PRODUCT adds on top of the kernel.

Registered onto the kernel's ToolRegistry from the Aloy backend; the kernel
stays product-agnostic (one-way dependency rule).
"""

from .gmail import GMAIL_TOOL_NAMES, register_gmail_tools

__all__ = ["GMAIL_TOOL_NAMES", "register_gmail_tools"]
