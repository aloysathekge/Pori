"""Purpose-scoped run profiles owned by the Aloy product."""

from pori import RunProfile

from .skills import SURFACE_BUILDER_SKILL_ID

SURFACE_BUILDER_RUN_PROFILE = RunProfile(
    profile_id="aloy.surface-builder",
    version="1",
    system_prompt=(
        "You are building or revising one Event Surface. Treat generated UI "
        "source as a proposed artifact: use only the Surface SDK and declared "
        "intents, preserve canonical Event data and the last-good revision, "
        "and never bypass validation, preview, approval, or publication rails."
    ),
    required_skill_ids=frozenset({SURFACE_BUILDER_SKILL_ID}),
    required_model_capabilities=frozenset({"tools"}),
)


__all__ = ["SURFACE_BUILDER_RUN_PROFILE"]
