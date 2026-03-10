import os
from functools import lru_cache
from pathlib import Path

_DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent / "prompts"
_PROMPTS_DIR_OVERRIDE: Path | None = None


def set_prompts_dir(path: str | Path | None) -> None:
    """Override the prompts base directory for this process.

    This is intended to be called by an embedding application (CLI, FastAPI, etc.)
    after loading config.
    """
    global _PROMPTS_DIR_OVERRIDE
    if path is None:
        _PROMPTS_DIR_OVERRIDE = None
        load_prompt.cache_clear()
        return
    p = Path(path).expanduser()
    _PROMPTS_DIR_OVERRIDE = p
    load_prompt.cache_clear()


def get_prompts_dir() -> Path:
    """Return the base directory to load prompts from.

    Resolution order:
    1) in-process override via `set_prompts_dir`
    2) env var `PORI_PROMPTS_DIR`
    3) packaged prompts directory
    """
    if _PROMPTS_DIR_OVERRIDE is not None:
        return _PROMPTS_DIR_OVERRIDE
    env = os.getenv("PORI_PROMPTS_DIR")
    if env:
        return Path(env).expanduser()
    return _DEFAULT_BASE_DIR


@lru_cache
def load_prompt(file_name: str) -> str:
    """Load a prompt text file from the prompts directory.

    Args:
        file_name: Relative path inside the prompts folder (e.g. 'agent_core.md')

    Returns:
        The file contents as a UTF-8 string.
    """
    path = get_prompts_dir() / file_name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")
