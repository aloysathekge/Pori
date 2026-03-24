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

```bash
git clone https://github.com/aloysathekge/pori.git
cd pori

uv sync --extra test

# Install pre-commit hooks (required — auto-formats code on each commit)
pip install pre-commit
pre-commit install

cp config.example.yaml config.yaml
# Create .env with ANTHROPIC_API_KEY and/or OPENAI_API_KEY
```

## Code formatting

This project uses [black](https://github.com/psf/black) and [isort](https://github.com/PyCQA/isort) for code formatting. Both run automatically via pre-commit hooks on every commit, so you generally don't need to think about formatting.

If you need to run them manually:

```bash
uv run black pori/ tests/
uv run isort pori/ tests/
```

CI checks (what runs on every PR):

```bash
uv run black --check pori/ tests/
uv run isort --check-only pori/ tests/
uv run mypy pori/ --ignore-missing-imports
uv run pytest tests/ -v
```

All checks must pass before a PR can be merged.

## Issues and PRs

Issues:

- Bugs: https://github.com/aloysathekge/pori/issues
- Features: open an issue with a concrete use case + examples

Pull requests:

- Keep PRs small and focused
- Include tests when behavior changes
- Pre-commit hooks handle formatting, but CI will catch anything that slips through

## Useful links

- `README.md`
- `ROADMAP.md`
