import asyncio

from .bootstrap import apply_windows_utf8_bootstrap
from .main import main


def run() -> None:
    apply_windows_utf8_bootstrap()
    asyncio.run(main())


if __name__ == "__main__":
    run()
