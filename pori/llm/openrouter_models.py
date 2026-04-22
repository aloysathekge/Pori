"""Curated catalog of OpenRouter models + an interactive picker.

Add entries to ``OPENROUTER_CATALOG`` over time — the picker renders them
grouped by ``category`` so the list stays readable as it grows.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class OpenRouterModel:
    slug: str
    label: str
    category: str
    context: int | None = None
    free: bool = False
    note: str = ""


# Keep this list curated — only models you've actually tested or plan to support.
# Slugs follow OpenRouter's "vendor/model[:variant]" convention.
# Reference: https://openrouter.ai/models
OPENROUTER_CATALOG: list[OpenRouterModel] = [
    # Free tier — good for testing
    OpenRouterModel(
        slug="inclusionai/ling-2.6-flash:free",
        label="Ling 2.6 Flash (free)",
        category="Free",
        free=True,
        note="Verified working via Pori",
    ),
    OpenRouterModel(
        slug="meta-llama/llama-3.3-70b-instruct:free",
        label="Llama 3.3 70B Instruct (free)",
        category="Free",
        context=131_072,
        free=True,
    ),
    OpenRouterModel(
        slug="deepseek/deepseek-chat:free",
        label="DeepSeek Chat (free)",
        category="Free",
        free=True,
    ),
    OpenRouterModel(
        slug="google/gemma-2-9b-it:free",
        label="Gemma 2 9B IT (free)",
        category="Free",
        free=True,
    ),
    # Open-source — paid
    OpenRouterModel(
        slug="meta-llama/llama-3.3-70b-instruct",
        label="Llama 3.3 70B Instruct",
        category="Open Source",
        context=131_072,
    ),
    OpenRouterModel(
        slug="qwen/qwen-2.5-72b-instruct",
        label="Qwen 2.5 72B Instruct",
        category="Open Source",
        context=131_072,
    ),
    OpenRouterModel(
        slug="mistralai/mistral-nemo",
        label="Mistral Nemo",
        category="Open Source",
        context=131_072,
    ),
    OpenRouterModel(
        slug="deepseek/deepseek-chat",
        label="DeepSeek Chat",
        category="Open Source",
    ),
    # Hosted frontier
    OpenRouterModel(
        slug="anthropic/claude-sonnet-4-5",
        label="Claude Sonnet 4.5",
        category="Hosted",
    ),
    OpenRouterModel(
        slug="openai/gpt-4o-mini",
        label="GPT-4o Mini",
        category="Hosted",
    ),
    OpenRouterModel(
        slug="google/gemini-2.0-flash-001",
        label="Gemini 2.0 Flash",
        category="Hosted",
    ),
]


# Config-model values that trigger the interactive picker.
SELECT_SENTINELS = {"select", "prompt", "pick", "?"}


def is_select_sentinel(model: str | None) -> bool:
    return bool(model) and str(model).strip().lower() in SELECT_SENTINELS


def _grouped(catalog: Iterable[OpenRouterModel]) -> dict[str, list[OpenRouterModel]]:
    groups: dict[str, list[OpenRouterModel]] = {}
    for m in catalog:
        groups.setdefault(m.category, []).append(m)
    return groups


def render_catalog(catalog: list[OpenRouterModel] | None = None) -> str:
    """Return the catalog formatted as a numbered menu string."""
    catalog = catalog or OPENROUTER_CATALOG
    lines: list[str] = []
    idx = 1
    for category, entries in _grouped(catalog).items():
        lines.append(f"\n  -- {category} --")
        for entry in entries:
            tag = " [free]" if entry.free else ""
            ctx = f"  ({entry.context // 1000}k ctx)" if entry.context else ""
            note = f"  - {entry.note}" if entry.note else ""
            lines.append(f"  {idx:>2}. {entry.label}{tag}{ctx}")
            lines.append(f"      {entry.slug}{note}")
            idx += 1
    return "\n".join(lines)


def pick_openrouter_model(
    default_slug: str | None = None,
    catalog: list[OpenRouterModel] | None = None,
    stream=None,
) -> str:
    """Interactive OpenRouter model picker.

    - Enter a number from the catalog
    - Press Enter to accept the default (if provided)
    - Type any raw slug (e.g. ``meta-llama/llama-3.3-70b-instruct``) to use a
      model not listed in the catalog

    Returns the chosen model slug.
    """
    catalog = catalog or OPENROUTER_CATALOG
    out = stream or sys.stdout

    print("\n=== Select an OpenRouter model ===", file=out, flush=True)
    print(render_catalog(catalog), file=out, flush=True)
    print(
        "\n  Type a number, paste any OpenRouter slug, or press Enter "
        f"{'to use default' if default_slug else 'to pick #1'}.",
        file=out,
    )
    print(
        "  Tip: slugs use 'vendor/model' format (e.g. meta-llama/llama-3.3-70b-instruct).\n",
        file=out,
        flush=True,
    )
    if default_slug:
        print(f"  Default: {default_slug}", file=out)

    while True:
        try:
            raw = input("  > ").strip()
        except EOFError:
            raw = ""

        if not raw:
            return default_slug or catalog[0].slug

        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(catalog):
                return catalog[n - 1].slug
            print(
                f"  Out of range (1–{len(catalog)}). Try again.",
                file=out,
            )
            continue

        # Accept any raw slug — OpenRouter will reject it at call time if
        # invalid, which is a clearer signal than us trying to pre-validate.
        if "/" in raw:
            return raw

        print(
            "  Not a number and not a slug (expected 'vendor/model'). Try again.",
            file=out,
        )
