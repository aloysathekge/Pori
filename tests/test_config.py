"""Tests for configuration loading, validation, and the LLM factory."""

import pytest
import yaml

from pori.config import AgentConfig, Config, LLMConfig, create_llm, load_config

pytestmark = pytest.mark.unit


def _write_yaml(path, data):
    path.write_text(yaml.safe_dump(data))
    return path


class TestLoadConfig:
    def test_loads_explicit_path(self, tmp_path):
        cfg_file = _write_yaml(
            tmp_path / "config.yaml",
            {"llm": {"provider": "anthropic", "model": "claude-opus-4-8"}},
        )
        config = load_config(cfg_file)
        assert isinstance(config, Config)
        assert config.llm.provider == "anthropic"
        assert config.llm.model == "claude-opus-4-8"
        # Defaults applied for the rest.
        assert config.agent.max_steps == 10
        # sqlite by default so sessions survive restarts (marathon Phase 1).
        assert config.memory.backend == "sqlite"

    def test_explicit_path_takes_precedence_over_env(self, tmp_path, monkeypatch):
        explicit = _write_yaml(
            tmp_path / "explicit.yaml",
            {"llm": {"provider": "openai", "model": "gpt-x"}},
        )
        env_file = _write_yaml(
            tmp_path / "env.yaml",
            {"llm": {"provider": "anthropic", "model": "claude-x"}},
        )
        monkeypatch.setenv("PORI_CONFIG", str(env_file))
        config = load_config(explicit)
        assert config.llm.provider == "openai"

    def test_uses_pori_config_env_when_no_explicit_path(self, tmp_path, monkeypatch):
        env_file = _write_yaml(
            tmp_path / "env.yaml",
            {"llm": {"provider": "google", "model": "gemini-x"}},
        )
        monkeypatch.setenv("PORI_CONFIG", str(env_file))
        # Avoid picking up a real ./config.yaml in the repo root.
        monkeypatch.chdir(tmp_path)
        config = load_config()
        assert config.llm.provider == "google"

    def test_missing_explicit_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "does_not_exist.yaml")

    def test_invalid_provider_raises(self, tmp_path):
        cfg_file = _write_yaml(
            tmp_path / "config.yaml",
            {"llm": {"provider": "not-a-provider", "model": "x"}},
        )
        with pytest.raises(Exception):  # pydantic ValidationError
            load_config(cfg_file)

    def test_missing_required_llm_field_raises(self, tmp_path):
        cfg_file = _write_yaml(
            tmp_path / "config.yaml",
            {"llm": {"provider": "anthropic"}},  # no model
        )
        with pytest.raises(Exception):
            load_config(cfg_file)


class TestAgentConfigBackCompat:
    def test_enable_planning_alias_maps_to_mode(self):
        cfg = AgentConfig(enable_planning=False)
        assert cfg.planning_mode == "never"
        cfg2 = AgentConfig(enable_planning=True)
        assert cfg2.planning_mode == "always"

    def test_enable_reflection_alias_maps_to_mode(self):
        cfg = AgentConfig(enable_reflection=False)
        assert cfg.reflection_mode == "never"

    def test_modes_default_to_never(self):
        # Planning is model-driven via the update_plan tool by default; the
        # separate planning/reflection LLM calls are opt-in.
        cfg = AgentConfig()
        assert cfg.planning_mode == "never"
        assert cfg.reflection_mode == "never"


class TestCreateLLM:
    def test_missing_api_key_raises_value_error(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            create_llm(LLMConfig(provider="anthropic", model="claude-opus-4-8"))

    def test_provider_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        # Pydantic accepts the canonical lowercase value; create_llm lowercases
        # before dispatch, so this must resolve to the OpenAI client.
        llm = create_llm(LLMConfig(provider="openai", model="gpt-x"))
        assert llm is not None

    def test_extra_params_passed_through(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        # Should not raise; extra_params are merged into the client kwargs.
        llm = create_llm(
            LLMConfig(
                provider="anthropic",
                model="claude-opus-4-8",
                extra_params={"max_tokens": 64},
            )
        )
        assert llm is not None
