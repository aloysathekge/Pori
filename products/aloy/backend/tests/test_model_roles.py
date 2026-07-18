from __future__ import annotations

import importlib

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect

from aloy_backend.model_roles import (
    ModelRole,
    ModelRoleUnavailableError,
    clear_model_role_cache,
    resolve_model_assignment,
)


def _write_roles(tmp_path, *, status: str = "qualified"):
    path = tmp_path / "aloy.models.yaml"
    path.write_text(
        f"""
version: 1
roles:
  surface_builder:
    provider: openai
    model: frontier-builder
    temperature: 0.1
    max_tokens: 16000
    reasoning_mode: none
    required_capabilities: [tools]
    skill_id: surface-builder@1
    qualification:
      status: {status}
      suite: aloy-surface-builder-v1
      evidence: eval:builder-1
  surface_critic:
    provider: openai
    model: frontier-critic
    required_capabilities: [vision, structured_output]
    qualification:
      status: qualified
      suite: aloy-surface-critic-v1
      evidence: eval:critic-1
""".strip(),
        encoding="utf-8",
    )
    clear_model_role_cache()
    return path


def test_qualified_model_role_resolves_to_tamper_evident_assignment(tmp_path):
    path = _write_roles(tmp_path)

    assignment = resolve_model_assignment(
        ModelRole.SURFACE_BUILDER,
        path=path,
        required_capabilities=frozenset({"tools"}),
        expected_skill_id="surface-builder@1",
        allowed_provider_profiles=("openai",),
        allowed_models=("frontier-builder",),
        environ={"OPENAI_API_KEY": "configured"},
    )

    assert assignment.provider == "openai"
    assert assignment.model == "frontier-builder"
    assert assignment.skill_id == "surface-builder@1"
    assert assignment.qualification_suite == "aloy-surface-builder-v1"
    assert assignment.resolution_ms >= 0
    assignment.verify_fingerprint()

    tampered = assignment.model_copy(update={"model": "other-model"})
    with pytest.raises(ModelRoleUnavailableError, match="fingerprint"):
        tampered.verify_fingerprint()


def test_unqualified_or_policy_denied_builder_fails_closed(tmp_path):
    path = _write_roles(tmp_path, status="unqualified")

    with pytest.raises(ModelRoleUnavailableError, match="qualification suite"):
        resolve_model_assignment(
            ModelRole.SURFACE_BUILDER,
            path=path,
            environ={"OPENAI_API_KEY": "configured"},
        )

    path = _write_roles(tmp_path, status="qualified")
    with pytest.raises(ModelRoleUnavailableError, match="organization policy"):
        resolve_model_assignment(
            ModelRole.SURFACE_BUILDER,
            path=path,
            allowed_models=("different-model",),
            environ={"OPENAI_API_KEY": "configured"},
        )


def test_builder_requires_available_provider_capabilities_and_exact_skill(tmp_path):
    path = _write_roles(tmp_path)

    with pytest.raises(ModelRoleUnavailableError, match="missing_credential"):
        resolve_model_assignment(
            ModelRole.SURFACE_BUILDER,
            path=path,
            environ={},
        )
    with pytest.raises(ModelRoleUnavailableError, match="must use skill"):
        resolve_model_assignment(
            ModelRole.SURFACE_BUILDER,
            path=path,
            expected_skill_id="surface-builder@2",
            environ={"OPENAI_API_KEY": "configured"},
        )
    with pytest.raises(ModelRoleUnavailableError, match="lacks required capabilities"):
        resolve_model_assignment(
            ModelRole.SURFACE_BUILDER,
            path=path,
            required_capabilities=frozenset({"audio"}),
            environ={"OPENAI_API_KEY": "configured"},
        )


def test_run_model_assignment_migration_round_trip(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'model-assignment.db'}")
    metadata = sa.MetaData()
    sa.Table("runs", metadata, sa.Column("id", sa.String(), primary_key=True))
    metadata.create_all(engine)

    with engine.begin() as connection:
        migration = importlib.import_module(
            "aloy_backend.alembic.versions.f8c9d0e1a2b3_run_model_assignments"
        )
        original_op = migration.op
        migration.op = Operations(MigrationContext.configure(connection))
        try:
            migration.upgrade()
            columns = {
                column["name"] for column in inspect(connection).get_columns("runs")
            }
            assert "model_assignment" in columns

            migration.downgrade()
            columns = {
                column["name"] for column in inspect(connection).get_columns("runs")
            }
            assert "model_assignment" not in columns
        finally:
            migration.op = original_op
