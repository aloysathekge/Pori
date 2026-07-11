"""Uvicorn launcher for the backend: forces UTF-8 stdio (Windows), reads
HOST / PORT / RELOAD from the environment, and serves
``aloy_backend.api:app``. Invoked as a console script or
``python -m aloy_backend.main``.
"""

import io
import os
import sys

import uvicorn


def run() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        for stream in (sys.stdout, sys.stderr):
            if isinstance(stream, io.TextIOWrapper):
                stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "").lower() in {"1", "true", "yes", "y", "on"}

    uvicorn.run("aloy_backend.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run()
