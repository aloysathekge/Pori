# Pori Roadmap

This document outlines the planned features, improvements, and areas where we're seeking community contributions.

## Vision

Pori aims to be the simplest, most extensible AI agent framework for Python developers. Our focus is on **clarity, modularity, and ease of use** rather than trying to be feature-complete.

## Current Version: 0.1.0

### ✅ Implemented Features

- Core agent with planning and reflection
- Memory system with conversation history and tool tracking
- Tool registry with Pydantic validation
- Task orchestration with parallel execution support
- Built-in tools (core, math, numbers, filesystem,)
- LangChain integration for LLM providers
- Comprehensive logging
- FastAPI server for async task execution

### 🚧 Core Framework Evolution (Upcoming)

- [ ] **Remove LangChain Dependency**
  - Replace `BaseChatModel` with lightweight `LLMClient`
  - Direct integration with OpenAI/Anthropic SDKs
  - Remove framework overhead for cleaner, more controllable architecture


### Human-in-the-Loop

- [ ] **Approval Gates**
  - Request human approval for sensitive operations
  - Configurable approval rules
  - Async approval workflow
  - *Help wanted: Workflow design, UX*

- [ ] **Intervention System**
  - Pause execution for human input
  - Modify agent plans mid-execution
  - Override tool calls
  - *Help wanted: Interactive systems design*

### Advanced Features

- [ ] **Multi-Agent Coordination**
  - Agent-to-agent communication
  - Hierarchical task delegation
  - Shared knowledge bases
  - *Help wanted: Distributed systems, multi-agent architectures*

- [ ] **Learning from Feedback**
  - Capture human corrections
  - Fine-tune prompts based on feedback
  - Preference learning
  - *Help wanted: ML/RL experience*

### Advanced Capabilities

- [ ] **Multimodal Support**
  - Image understanding
  - Document processing
  - Audio/video handling
  - *Help wanted: Multimodal AI experience*

- [ ] **Code Generation & Execution**
  - Safe code execution sandboxes
  - Jupyter kernel integration
  - Code analysis tools
  - *Help wanted: Sandboxing, security*

### Observability

- [ ] **Monitoring & Metrics**
  - Prometheus/Grafana integration
  - Custom metrics and dashboards
  - Alert systems
  - *Help wanted: Observability, SRE*

## Community Contribution Areas

### 🟢 Good for Beginners

1. **Documentation**
   - Tutorial writing
   - API documentation
   - Example notebooks
   - Video tutorials

2. **Testing**
   - Increase test coverage
   - Add edge case tests
   - Integration tests
   - Performance benchmarks

3. **Tools**
   - Add new built-in tools
   - Improve existing tools
   - Tool documentation

4. **Examples**
   - Real-world use cases
   - Integration guides
   - Best practices

### 🟡 Intermediate

1. **Core Features**
   - Memory system enhancements
   - Tool registry improvements
   - Error handling

2. **API Improvements**
   - FastAPI endpoints
   - WebSocket support
   - Authentication

3. **CLI Enhancements**
   - Interactive modes
   - Better output formatting
   - Configuration management

### 🔴 Advanced

1. **Architecture**
   - Multi-agent systems
   - Distributed execution
   - Performance optimization

2. **LLM Integration**
   - New provider support
   - Custom model adapters
   - Prompt optimization

3. **Security**
   - Sandboxing
   - Permission systems
   - Vulnerability audits

## How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions on how to contribute to any of these areas.

### Proposing New Features

Before working on a major feature:

1. Check this roadmap and existing issues
2. Open a GitHub Discussion to gather feedback
3. Create a detailed feature proposal issue
4. Get maintainer approval before significant work
5. Break down into smaller, reviewable PRs

### Feature Requests

Have an idea not on this roadmap? We'd love to hear it!

- Open a [Feature Request](https://github.com/aloysathekge/pori/issues/new?template=feature_request.md)
- Participate in [Discussions](https://github.com/aloysathekge/pori/discussions)
- Join community conversations



To maintain focus, we explicitly don't plan to:

- Build a no-code/low-code interface (there are other projects for that)
- Support every possible LLM provider natively (we will implement key providers directly)
- Become a full workflow orchestration system 
- Implement agent UI frameworks (focus is on backend)

## Versioning Policy

- **Major versions** (1.0, 2.0): Breaking API changes
- **Minor versions** (0.1, 0.2): New features, backward compatible
- **Patch versions** (0.1.1, 0.1.2): Bug fixes only

## Questions?

- **General questions**: [GitHub Discussions](https://github.com/aloysathekge/pori/discussions)
- **Feature proposals**: [Feature Request Template](https://github.com/aloysathekge/pori/issues/new?template=feature_request.md)
- **Roadmap feedback**: Comment on this file via PR or issue

---

*Last updated: 2025-11-11*
*This roadmap is subject to change based on community feedback and priorities.*
