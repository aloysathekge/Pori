# Current State

## Active Task

Two sandbox correctness fixes on `fix/sandbox-working-dir-and-artifact-tracking`
(committed). Both surfaced from the same failing agent task ("write a txt file,
then a program that zips it"), which the agent could not complete because of
sandbox tooling gaps — not reasoning.

### 1. bash `working_dir` was ignored

`BashParams.working_dir` was declared and advertised to the model but never
read; `LocalSandbox.execute_command` ran `subprocess.run` with no `cwd`, so
every command executed in pori's launch directory (repo root) instead of the
thread sandbox dir. Relative-path programs and bare script names could not find
their sibling files.

Fix: `execute_command(command, cwd=...)` through the `Sandbox` interface;
`bash_tool` resolves `working_dir` (virtual → real via `replace_virtual_path`)
and defaults to the thread workspace when omitted — never the host cwd.

### 2. bash-produced files were not tracked as artifacts

Artifact tracking was tool-self-declaration based (`write_file` /
`sandbox_write_file` only). A file created by a program run under `bash` (e.g.
the produced `.zip`) was invisible — and could never be captured that way, since
a shell command does not announce which files it wrote.

Fix: the sandbox now *observes* the filesystem. `bash_tool` snapshots the thread
dirs before/after the command (bounded at 5000 files) and reports created/
modified files under `files_written` as virtual paths. New `to_virtual_path()`
(inverse of `replace_virtual_path`) is used for display. `agent.py` and
`main.py` artifact collectors now read `files_written`; the existing
`write_file`/`sandbox_write_file` path is untouched.

## Verification

- `uv run pytest tests/ -q`: 337 passed (was 332; +5 new sandbox/CLI tests,
  including a regression that asserts the produced `.zip` is reported).
- black, isort, and `mypy pori/sandbox pori/agent.py pori/main.py
  --ignore-missing-imports` all clean on touched files.

## Remaining Risks / Follow-ups

- Filesystem observation is scoped to the **local** sandbox and its thread dirs
  (the correct boundary). A container-backed sandbox would need its own diff
  mechanism to report `files_written`.
- Snapshot detection is size+mtime based; a same-size, same-mtime rewrite would
  not be flagged (negligible in practice).
- `debug.log` is tracked in git and rewritten on every run (dirties the tree and
  once blocked a `git reset`). Consider gitignoring it. Separately, the local
  sandbox runs shell commands with `shell=True`; defaulting cwd to the workspace
  reduces but does not eliminate the "commands touch the real repo" concern.

## Next Session Should Start With

Open a PR for `fix/sandbox-working-dir-and-artifact-tracking` (base `main`).
The receipt-backed verification V1 was reverted earlier this session by user
request (cost); do not reintroduce a costly verification gate.
