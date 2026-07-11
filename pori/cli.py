"""Console-script entry point (the ``pori`` command, wired in
``pyproject.toml``): applies the Windows UTF-8 stdio bootstrap, then runs the
interactive CLI loop in :func:`pori.main.main`.
"""

import asyncio

from .bootstrap import apply_windows_utf8_bootstrap
from .main import main


def run() -> None:
    apply_windows_utf8_bootstrap()
    asyncio.run(main())


if __name__ == "__main__":
    run()
