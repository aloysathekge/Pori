from __future__ import annotations

import logging
import logging.config

from .config import settings


def setup_logging() -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "pori_cloud": {
                    "level": settings.log_level,
                    "handlers": ["console"],
                    "propagate": False,
                },
                "uvicorn": {
                    "level": settings.log_level,
                    "handlers": ["console"],
                    "propagate": False,
                },
            },
            "root": {
                "level": "WARNING",
                "handlers": ["console"],
            },
        }
    )
