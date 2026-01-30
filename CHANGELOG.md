# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-30

### Added
- Initial release of Pori agent framework
- Core agent implementation with Plan → Act → Reflect loop
- Tiered memory system (messages, tool history, task state, experiences)
- Tool registry with Pydantic validation and decorator-based registration
- Built-in tools: answer, done, think, remember, calculate, fibonacci, random number generation, filesystem operations, Spotify integration
- LLM integration layer with OpenAI and Anthropic support (no LangChain dependency)
- Orchestrator for task management and parallel execution
- Task evaluation and completion detection
- Comprehensive logging and error handling
- FastAPI server for async task execution (optional dependency)
- CLI interface via `pori` command
- Test suite with pytest
- Type hints and Pydantic models throughout

### Features
- **Agent**: Autonomous reasoning with planning and reflection
- **Memory**: Session-based memory with conversation history, tool tracking, and basic semantic recall
- **Tools**: Extensible tool system with automatic validation
- **Orchestration**: Manage multiple agents and parallel tasks
- **LLM Support**: OpenAI (GPT-4, GPT-3.5) and Anthropic (Claude) models
- **Structured Output**: Force LLM responses into Pydantic models
- **Retry Logic**: Automatic retry for failed tool calls
- **Duplicate Detection**: Reuse results from identical tool calls

### Documentation
- README with quick start guide
- Contributing guidelines
- Roadmap for future development
- Code examples in `examples/` directory
- MIT License

[0.1.0]: https://github.com/aloysathekge/pori/releases/tag/v0.1.0
