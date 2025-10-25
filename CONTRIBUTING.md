# Contributing to Pori

Thank you for your interest in contributing to Pori! 🎉

Pori is a community-driven project and we welcome contributions of all kinds: bug reports, feature requests, documentation improvements, code contributions, and more.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Ways to Contribute

- 🐛 **Report bugs** - Help us identify issues
- 💡 **Suggest features** - Share ideas for improvements
- 📖 **Improve documentation** - Fix typos, add examples, clarify concepts
- 🔧 **Add new tools** - Extend Pori's capabilities
- 🧪 **Write tests** - Improve code coverage
- 🎨 **Improve UX** - Better CLI output, error messages, etc.
- 📝 **Write tutorials** - Help others learn Pori

### Good First Issues

Look for issues tagged with `good-first-issue` or `help-wanted` labels. These are great entry points for new contributors.

## How to Contribute

### Reporting Bugs

Before submitting a bug report:
1. Check the [existing issues](https://github.com/aloysathekge/pori/issues) to avoid duplicates
2. Ensure you're using the latest version of Pori
3. Collect relevant information (error messages, logs, environment details)

Use our [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) when creating an issue.

### Requesting Features

We love hearing new ideas! Before submitting:
1. Check if the feature has already been requested
2. Consider if it fits Pori's scope and philosophy
3. Provide clear use cases and examples

Use our [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).

### Asking Questions

For questions about usage:
- Check the [documentation](README.md) first
- Search [existing discussions](https://github.com/aloysathekge/pori/discussions)
- Create a new discussion (not an issue) for general questions

## Development Setup

### Prerequisites

- Python 3.8.1 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Git

### Local Setup

```powershell
# Clone the repository
git clone https://github.com/aloysathekge/pori.git
cd pori

# Create virtual environment
uv venv
.venv\Scripts\activate  # On Windows
# source .venv/bin/activate  # On Unix/macOS

# Install dependencies
uv pip install -r requirements.txt

# Install development dependencies
uv pip install -e ".[test]"

# Set up environment variables
copy .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Project Structure

```
pori/
├── pori/                  # Main package
│   ├── agent.py          # Core agent implementation
│   ├── orchestrator.py   # Task management
│   ├── memory.py         # Memory system
│   ├── tools.py          # Tool registry
│   ├── evaluation.py     # Task evaluation
│   ├── tools_builtin/    # Built-in tools
│   ├── api/              # FastAPI server
│   └── utils/            # Utilities
├── tests/                # Test suite
├── examples/             # Usage examples
└── Dev_docs/             # Development documentation
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical documentation.

## Development Workflow

### Branching Strategy

- `main` - Stable production branch
- `develop` - Development branch for integrating features
- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `docs/*` - Documentation updates

### Making Changes

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, focused commits
   - Follow our code standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run all tests
   pytest
   
   # Run specific test categories
   pytest -m unit
   pytest -m integration
   
   # Check code coverage
   pytest --cov=pori --cov-report=html
   ```

4. **Lint and format**
   ```bash
   # Format code
   black pori/ tests/
   isort pori/ tests/
   
   # Type checking
   mypy pori/
   ```

## Code Standards

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://github.com/psf/black) for formatting (line length: 88)
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use type hints where appropriate

### Code Quality

- **Functions**: Keep functions focused and under 50 lines when possible
- **Classes**: Follow Single Responsibility Principle
- **Naming**: Use descriptive names (`calculate_fibonacci` not `calc_fib`)
- **Comments**: Explain *why*, not *what* (code should be self-documenting)
- **Documentation**: Use docstrings for public APIs

### Example Code Style

```python
from typing import Dict, Any, List
from pydantic import BaseModel, Field

class ToolParams(BaseModel):
    """Parameters for the example tool."""
    
    query: str = Field(..., description="The search query")
    max_results: int = Field(5, ge=1, le=100, description="Maximum results")

def example_tool(params: ToolParams, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the example tool.
    
    Args:
        params: Validated tool parameters
        context: Execution context with memory and state
        
    Returns:
        Result dictionary with success status and data
    """
    try:
        # Implementation here
        result = perform_search(params.query, params.max_results)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

## Testing

### Test Structure

- **Unit tests**: Test individual functions/classes in isolation
- **Integration tests**: Test interactions between components
- **End-to-end tests**: Test complete workflows

### Writing Tests

```python
import pytest
from pori import Agent, AgentSettings
from pori.tools import ToolRegistry

@pytest.mark.unit
def test_tool_registration():
    """Test that tools can be registered properly."""
    registry = ToolRegistry()
    # Test implementation
    assert len(registry.tools) == 0

@pytest.mark.integration
async def test_agent_execution():
    """Test that agent can execute a simple task."""
    # Test implementation
    pass
```

### Test Markers

- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Component integration tests
- `@pytest.mark.memory` - Memory system tests
- `@pytest.mark.tools` - Tool system tests
- `@pytest.mark.agent` - Agent functionality tests

### Running Tests

```bash
# All tests
pytest

# Specific markers
pytest -m unit
pytest -m "not integration"

# With coverage
pytest --cov=pori --cov-report=term-missing

# Verbose output
pytest -v -s
```

## Pull Request Process

### Before Submitting

- ✅ Tests pass locally
- ✅ Code is formatted (Black, isort)
- ✅ Type checking passes (mypy)
- ✅ Documentation is updated
- ✅ Commit messages are clear
- ✅ Branch is up to date with main

### PR Guidelines

1. **Title**: Clear, descriptive title
   - Good: "Add semantic search tool for memory recall"
   - Bad: "Fix bug"

2. **Description**: Use the PR template
   - What changes were made?
   - Why were they necessary?
   - How were they tested?
   - Any breaking changes?

3. **Size**: Keep PRs focused and reasonably sized
   - Large changes should be discussed in an issue first
   - Consider splitting into multiple PRs

4. **Reviews**: 
   - Address review comments promptly
   - Mark conversations as resolved when done
   - Don't force-push after review starts (unless requested)

### PR Checklist

```markdown
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Code formatted (Black, isort)
- [ ] Type hints added where appropriate
- [ ] No breaking changes (or clearly documented)
- [ ] Linked to relevant issue(s)
```

### After Submission

- CI checks must pass
- At least one maintainer approval required
- Address any requested changes
- Maintainers will merge when ready

## Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, general discussion
- **Pull Requests**: Code contributions

### Getting Help

If you're stuck or need guidance:
1. Check the documentation
2. Search existing issues/discussions
3. Ask in GitHub Discussions
4. Tag maintainers if urgent

### Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` (auto-generated)
- Release notes
- Project README

## Additional Resources

- [Architecture Documentation](ARCHITECTURE.md)
- [Development Roadmap](ROADMAP.md)
<!-- - [API Documentation](https://pori.readthedocs.io) *-->

---

Thank you for contributing to Pori! Every contribution, no matter how small, helps make this project better. 🙏
