import logging
import sys
from typing import Dict, Any


class PoriFormatter(logging.Formatter):
    """Custom formatter with context."""

    def format(self, record):
        # Add context if available
        if hasattr(record, "task_id"):
            record.msg = f"[Task:{record.task_id}] {record.msg}"
        if hasattr(record, "step"):
            record.msg = f"[Step:{record.step}] {record.msg}"
        return super().format(record)


def setup_logging(level=logging.INFO, include_http=False):
    """Configure logging for Pori framework."""

    formatter = PoriFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Set up component loggers
    loggers = {
        "pori.agent": logging.getLogger("pori.agent"),
        "pori.orchestrator": logging.getLogger("pori.orchestrator"),
        "pori.tools": logging.getLogger("pori.tools"),
        "pori.memory": logging.getLogger("pori.memory"),
    }

    for logger in loggers.values():
        logger.setLevel(level)
        logger.addHandler(console_handler)
        logger.propagate = False

    # Control external library logging
    if not include_http:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("anthropic").setLevel(logging.WARNING)

    return loggers
