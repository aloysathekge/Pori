import asyncio

from .bootstrap import apply_windows_utf8_bootstrap
from .main import main

if __name__ == "__main__":
    apply_windows_utf8_bootstrap()
    asyncio.run(main())
