"""The `/learn` prompt builder — user-triggered skill authoring (SK-1, layer 1).

`/learn` turns whatever the user describes (a directory, a doc URL, a workflow
they just walked the agent through, pasted notes) into one reusable skill. There
is no separate distillation engine and no new model-tool footprint beyond the
`write_skill` tool: :func:`build_learn_prompt` returns ONE instruction that the
live agent runs as a normal turn, gathering sources with the tools it already has
and authoring a single SKILL.md. This works identically on every backend.
"""

from __future__ import annotations

# House-style authoring rules, adapted to Pori's SKILL.md contract (a `---` YAML
# frontmatter with name/description/version, then the instructions body). Embedded
# in the prompt so the agent authors skills the way a maintainer would by hand.
_AUTHORING_STANDARDS = """\
Follow these skill-authoring standards exactly:

Frontmatter (a YAML block fenced by --- at the very top):
- name: lowercase-hyphenated, <=64 chars, no spaces (this is the slug).
- description: ONE sentence, **<=60 characters**, ending with a period. State the
  capability, not the implementation. No marketing words (powerful, comprehensive,
  seamless, advanced, robust). Do NOT repeat the skill name. The skill index
  truncates the description to 60 chars and loads it every session, so anything
  past char 60 is silently cut and never routes — COUNT the characters and cut it
  down before saving.
    Good (<=60): Search arXiv papers by keyword, author, or ID.
    Bad  (123):  A comprehensive skill that lets the agent search arXiv for
                 academic papers using keywords, authors, and categories.
- version: 0.1.0
- author: the literal value `Pori`. Never fill it from the host environment.

Body (after the closing ---):
- A short title, then the procedure as clear numbered steps a future agent can
  follow. Frame steps around Pori's real tools (read_file, search_files,
  write_file, web_search, run_command) — do NOT invent tools or commands.
- Only document commands you have actually seen work in this session or the
  gathered sources. If you are unsure a command exists, say so rather than guess.
- Keep it tight: a skill is a checklist for next time, not an essay.
"""


def build_learn_prompt(request: str) -> str:
    """Build the instruction that has the agent author one skill from ``request``.

    The result is fed to the agent as a normal turn; it uses its existing tools to
    gather sources and then calls ``write_skill`` once to save the SKILL.md.
    """
    request = (request or "").strip()
    focus = request or (
        "the reusable procedure demonstrated in this conversation so far"
    )
    return (
        "You are authoring a reusable skill. Learn and capture: "
        f"{focus}.\n\n"
        "Work in three steps:\n"
        "1. GATHER — collect the source material using the tools you already have: "
        "read_file / search_files for a directory or file, web_search for a URL or "
        "topic, and this conversation itself for a workflow you just performed. Do "
        "not author anything until you understand the procedure.\n"
        "2. AUTHOR — write ONE SKILL.md as a checklist a future agent can follow.\n"
        "3. SAVE — call the write_skill tool exactly once with the slug (the "
        "frontmatter name) and the full SKILL.md content. Do not use write_file for "
        "this; write_skill installs it into the skills directory.\n\n"
        f"{_AUTHORING_STANDARDS}\n"
        "After saving, briefly tell the user what skill you created and that it is "
        "available after /reload-skills (or on the next session)."
    )


def build_background_review_prompt(digest: str) -> str:
    """Build the instruction for the autonomous post-run review agent (layer 2).

    It looks at a digest of a just-finished session and decides — conservatively —
    whether it contained a genuinely reusable procedure worth saving as a skill.
    """
    return (
        "You are reviewing a just-finished agent session to decide whether it "
        "contained a REUSABLE procedure worth saving as a skill.\n\n"
        "Session digest:\n"
        f"{digest}\n\n"
        "Be conservative — most sessions are one-off and NOT worth a skill. Only "
        "save a genuinely general, repeatable procedure (a workflow a future agent "
        "would follow again), never a specific answer or a trivial task.\n\n"
        "- If nothing is reusable: call done immediately and write nothing.\n"
        "- If something is: author ONE SKILL.md and call write_skill exactly once.\n\n"
        f"{_AUTHORING_STANDARDS}"
    )
