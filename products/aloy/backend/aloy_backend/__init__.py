"""Aloy backend package — the FastAPI service that hosts the Pori kernel for
the Aloy product. Only the package version lives here.
"""

from pathlib import Path

from dotenv import load_dotenv

# The backend's own ``.env`` must win over any other dotenv file on this
# machine (the kernel's ``pori.config`` also calls ``load_dotenv`` at import
# time and finds the repo-root ``.env``; with override=False the first load
# wins). Loading here — package import runs before any submodule or the kernel
# — gives the deterministic precedence: real environment > backend/.env >
# repo-root .env. Real env vars are never overridden, so production is
# unaffected.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

__all__ = ["__version__"]

__version__ = "0.1"
