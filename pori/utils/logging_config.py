"""Logging plumbing for the pori package: ``PoriFormatter`` (level colors +
request/task context), ``RequestIdFilter`` (injects the ``request_id``
contextvar), and ``ensure_logger_configured`` — the accessor modules use to
get a namespaced, configured logger without double-adding handlers.
``setup_logging`` does one-time global configuration for the CLI.
"""

import logging
import sys
from typing import Any, Dict

from pori.utils.context import request_id_var


class RequestIdFilter(logging.Filter):
    """
    Injects the request_id into the log record.
    """

    def filter(self, record):
        record.request_id = request_id_var.get()
        return True


class PoriFormatter(logging.Formatter):
    """Custom formatter with context and colors."""

    # Color codes for different log levels
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        # Add context if available
        if hasattr(record, "request_id") and record.request_id:
            record.msg = f"[Req:{record.request_id[:8]}] {record.msg}"
        if hasattr(record, "task_id"):
            record.msg = f"[Task:{record.task_id}] {record.msg}"
        if hasattr(record, "step"):
            record.msg = f"[Step:{record.step}] {record.msg}"

        # Format the message
        formatted_message = super().format(record)

        # Add color if terminal supports it
        if sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, "")
            if color:
                formatted_message = f"{color}{formatted_message}{self.RESET}"

        return formatted_message


class ImmediateStreamHandler(logging.StreamHandler):
    """Stream handler that flushes immediately after each emit."""

    def emit(self, record):
        super().emit(record)
        self.flush()


def ensure_logger_configured(logger_name: str, level=logging.INFO):
    """Ensure a logger is properly configured even if it was created before setup_logging."""
    logger = logging.getLogger(logger_name)

    # Only configure if it doesn't already have handlers
    if not logger.handlers:
        formatter = PoriFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        console_handler = ImmediateStreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        logger.setLevel(level)
        logger.addHandler(console_handler)
        logger.propagate = False

    return logger


def setup_logging(
    level=logging.INFO, include_http=False, console_level=None, log_file=None
):
    """Configure logging for Pori framework.

    Args:
        level: Level captured by the loggers (and the file handler).
        include_http: If False, quiet httpx/anthropic to WARNING.
        console_level: Level shown on the console. Defaults to ``level``. Set to
            WARNING to keep the interactive console clean while a file captures
            the full INFO trace.
        log_file: Optional path; when given, INFO+ is written here (rotating) so
            the console can stay quiet without losing the debug trail.
    """

    formatter = PoriFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    # Add the request ID filter to the console handler
    console_handler = ImmediateStreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RequestIdFilter())
    console_handler.setLevel(console_level if console_level is not None else level)

    handlers: list[logging.Handler] = [console_handler]

    if log_file:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            log_file, maxBytes=2_000_000, backupCount=2, encoding="utf-8"
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        file_handler.addFilter(RequestIdFilter())
        file_handler.setLevel(level)
        handlers.append(file_handler)

    # Set up component loggers
    loggers = {
        "pori.main": logging.getLogger("pori.main"),
        "pori.agent": logging.getLogger("pori.agent"),
        "pori.orchestrator": logging.getLogger("pori.orchestrator"),
        "pori.tools": logging.getLogger("pori.tools"),
        "pori.memory": logging.getLogger("pori.memory"),
        "pori.llm": logging.getLogger("pori.llm"),
    }

    for logger in loggers.values():
        logger.setLevel(level)
        # Clear any existing handlers to avoid duplicates
        logger.handlers.clear()
        for handler in handlers:
            logger.addHandler(handler)
        logger.propagate = False

    # Control external library logging
    if not include_http:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("anthropic").setLevel(logging.WARNING)

    # Ensure root logger doesn't interfere
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)

    return loggers
