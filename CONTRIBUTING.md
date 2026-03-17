# Contributing to Pori

This guide is intentionally short. If you want to contribute, the main loop is:

1. Create a branch
2. Make changes
3. Run formatting + tests
4. Open a PR

## Code of Conduct

Read `CODE_OF_CONDUCT.md` before contributing.

## Quick start (development)

Requirements:

- Python 3.8.1+
- `uv`
- Git

Setup:

```powershell
git clone https://github.com/aloysathekge/pori.git
cd pori

uv sync --extra test

copy config.example.yaml config.yaml
# Create .env with ANTHROPIC_API_KEY and/or OPENAI_API_KEY
```

## Common commands

Format:

```powershell
uv run black pori/ tests/
uv run isort pori/ tests/
```

Checks (what CI runs):

```powershell
uv run black --check pori/ tests/
uv run isort --check-only pori/ tests/
uv run mypy pori/ --ignore-missing-imports
uv run pytest tests/ -v
```

## Issues and PRs

Issues:

- Bugs: https://github.com/aloysathekge/pori/issues
- Features: open an issue with a concrete use case + examples

Pull requests:

- Keep PRs small and focused
- Include tests when behavior changes
- Ensure the commands above pass before pushing

## Useful links

- `README.md`
- `ROADMAP.md`
