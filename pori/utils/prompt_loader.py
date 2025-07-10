from functools import lru_cache
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache
def load_prompt(file_name: str) -> str:
    """Load a prompt text file from the prompts directory.

    Args:
        file_name: Relative path inside the prompts folder (e.g. 'agent_core.md')

    Returns:
        The file contents as a UTF-8 string.
    """
    path = _BASE_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")
