# Sandbox Implementation Plan

How to integrate DeerFlow-style sandboxing into your own agentic framework. Use this as a step-by-step plan; adapt to your stack (LangGraph, custom loop, etc.).

---

## What’s done vs what’s not (current status)

- **Done:** Sandbox interface, provider, virtual paths, path resolution, and local sandbox. The **bash** tool is implemented and wired. **Agent integration:** the agent passes `thread_id` and `sandbox_base_dir` in tool context; the orchestrator accepts `sandbox_base_dir` and passes it to the agent. **Config:** `sandbox.enabled` and `sandbox.base_dir` in config; main sets the provider and base dir when enabled. Per-task dirs are created on first bash use. So the **local sandbox path is complete** for the plan.
- **Not done (optional):** (1) More sandbox tools (e.g. sandbox-backed `read_file` / `write_file` / `list_dir` using the Sandbox API and virtual paths). (2) Container-based sandbox (Step 8). (3) Sandbox section in CONTRIBUTING or architecture docs.

---

## What the new code does (plain English)

New files live under **`pori/sandbox/`**. Here’s what each one is for.

### 1. `pori/sandbox/base.py`

- **Sandbox**  
  An abstract “box” that can run a shell command and do file read/write/list. The agent never talks to “the real machine” directly; it talks to a Sandbox. Later you can have a “local” box (this machine) or a “container” box (e.g. Docker).
- **SandboxProvider**  
  Something that gives you a Sandbox when you ask for one, keyed by “thread” (conversation/session). Same thread → same box so the conversation keeps the same files and cwd.
- **get_sandbox_provider() / set_sandbox_provider()**  
  One global “who is the provider?” so the rest of the app can say “give me the sandbox for this thread” without passing the provider everywhere.

### 2. `pori/sandbox/path_resolution.py`

- **Virtual paths**  
  The agent is told to use fixed path names like `/mnt/user-data/workspace` and `/mnt/user-data/outputs` instead of real paths like `C:\Users\...`. That way prompts and tools don’t depend on the actual OS or machine.
- **ThreadData**  
  For one thread we have three real folders: workspace (scratch), uploads, outputs. `ThreadData` holds those three paths. `get_thread_data(thread_id, base_dir)` creates the folder layout for that thread and returns a `ThreadData`.
- **replace_virtual_path / replace_virtual_paths_in_command**  
  When the sandbox is “local” (this machine), we have to turn the agent’s virtual path into a real path before running a command or opening a file. These functions do that: e.g. `/mnt/user-data/workspace/foo.txt` → `C:\...\threads\abc123\user-data\workspace\foo.txt`.

### 3. `pori/sandbox/local.py`

- **LocalSandbox**  
  The concrete “box” that runs on this machine: runs commands with `subprocess`, reads/writes files with normal Python. It can optionally map extra virtual paths (e.g. `/mnt/skills`) to real paths.
- **LocalSandboxProvider**  
  The provider that always gives you that one local sandbox. `acquire(thread_id)` returns the id `"local"`; `get("local")` returns the LocalSandbox. So right now every thread shares the same host sandbox; the per-thread isolation is only via different **folders** (workspace/uploads/outputs) that path_resolution maps for each thread.

### 4. `pori/sandbox/sandbox_tools.py`

- **bash tool**  
  A single tool the agent can call to run a shell command. When you call it:
  - It gets the current SandboxProvider and acquires a sandbox (using `thread_id` from context if present).
  - If the sandbox is local and we have thread data, it rewrites any `/mnt/user-data/...` paths in the command to the real paths for that thread.
  - Then it runs the command in that sandbox and returns the output.

So: **the plumbing is in place and wired.** The agent passes `thread_id` and `sandbox_base_dir` in tool context; main sets the sandbox provider and base dir from config when `sandbox.enabled` is true.

---

## Quick reference: what’s implemented where

| Step in plan | Done? | Where |
|--------------|-------|--------|
| 1. Sandbox interface | Yes | `pori/sandbox/base.py` |
| 2. SandboxProvider | Yes | `pori/sandbox/base.py` |
| 3. Virtual paths + thread data | Yes | `pori/sandbox/path_resolution.py` |
| 4. Path resolution | Yes | `pori/sandbox/path_resolution.py` |
| 5. Local sandbox | Yes | `pori/sandbox/local.py` |
| 6. Tools wired to sandbox | Yes (bash) | `pori/sandbox/sandbox_tools.py`; agent context has thread_id + sandbox_base_dir |
| 7. Agent loop integration | Yes | Agent + orchestrator + main pass thread_id and sandbox_base_dir; sandbox_tools registered in main |
| 8. Container sandbox | No (optional) | — |
| 9. Config + docs | Config yes; docs no | `pori/config.py` (SandboxConfig), `config.yaml`; no CONTRIBUTING/architecture section yet |

---

## 1. Define the Sandbox Interface

**Goal:** One abstraction for “an execution environment” that can run commands and do file I/O.

**Steps:**

1. **Create a `Sandbox` abstract class** with at least:
   - `execute_command(command: str) -> str` — run a shell command, return combined stdout/stderr.
   - `read_file(path: str) -> str` — read file as text.
   - `write_file(path: str, content: str, append: bool = False) -> None` — write or append text.
   - `list_dir(path: str, max_depth: int = 2) -> list[str]` — list paths (e.g. dirs with trailing `/`).

2. **Optional:** Add `update_file(path: str, content: bytes) -> None` if you need binary writes.

3. **Design choice:** Keep the interface small. Extra capabilities (e.g. network) can be added later or via a different abstraction.

**Deliverable:** One module (e.g. `sandbox/base.py`) with the `Sandbox` ABC and docstrings.

---

## 2. Define the Sandbox Provider

**Goal:** Manage sandbox lifecycle per conversation/thread so each run gets an isolated environment.

**Steps:**

1. **Create a `SandboxProvider` abstract class** with:
   - `acquire(thread_id: str | None = None) -> str` — create or reuse a sandbox for that thread; return a stable `sandbox_id`.
   - `get(sandbox_id: str) -> Sandbox | None` — return the `Sandbox` instance for that id.
   - `release(sandbox_id: str) -> None` — tear down the sandbox (e.g. stop container, free resources).

2. **Decide ownership:** One global provider instance (singleton) per process, or inject it into your agent/graph. Configuration (e.g. `sandbox.use: LocalSandboxProvider`) should drive which implementation is used.

3. **Thread–sandbox mapping:** For the same `thread_id`, `acquire` should return the same `sandbox_id` so the same conversation reuses the same environment across turns.

**Deliverable:** `SandboxProvider` ABC and a way to get/set the active provider (e.g. `get_sandbox_provider()`, `set_sandbox_provider()`).

---

## 3. Introduce Virtual Paths and Thread Data

**Goal:** The agent always sees the same paths; your framework maps them to per-thread (or per-session) directories.

**Steps:**

1. **Pick a virtual prefix**, e.g. `/mnt/user-data`.

2. **Define three logical areas:**
   - `/mnt/user-data/workspace` — scratch/temporary work.
   - `/mnt/user-data/uploads` — user-uploaded files.
   - `/mnt/user-data/outputs` — final deliverables.

3. **Define “thread data”** (or “session data”): a small struct with three paths on the host (or inside the container):
   - `workspace_path`, `uploads_path`, `outputs_path`.

4. **Create the physical layout** for each thread, e.g.:
   - `{base_dir}/threads/{thread_id}/user-data/workspace`
   - `{base_dir}/threads/{thread_id}/user-data/uploads`
   - `{base_dir}/threads/{thread_id}/user-data/outputs`

5. **Document this layout in your system prompt** so the model uses these paths and does not rely on host-specific paths.

**Deliverable:** A config or helper that, given `thread_id`, returns the three paths and ensures the directories exist (e.g. “thread data middleware” or “thread context” step that runs before the agent).

---

## 4. Implement Path Resolution (for Local / Host Sandbox)

**Goal:** When the sandbox runs on the host (no real container), tools must rewrite virtual paths to real paths before calling the sandbox.

**Steps:**

1. **Implement `replace_virtual_path(path, thread_data, prefix)`:**  
   If `path` starts with the prefix (e.g. `/mnt/user-data`), map the first segment (`workspace`, `uploads`, or `outputs`) to the corresponding path in `thread_data` and return the full real path; otherwise return `path` unchanged.

2. **Implement `replace_virtual_paths_in_command(command, thread_data, prefix)`:**  
   Use a regex (or similar) to find all occurrences of the virtual prefix followed by a path in the command string, and replace each with the result of `replace_virtual_path`. This lets the agent pass paths in bash commands and still hit the right dirs.

3. **Use resolution only when needed:**  
   If your sandbox is a real container that already mounts the same layout (e.g. `/mnt/user-data/...` inside the container), you do **not** need to rewrite paths inside the sandbox—only when the “sandbox” is the host (local provider).

4. **Optional:** If you have a shared read-only directory (e.g. skills), add a second mapping (e.g. `/mnt/skills` → host path) and apply it in the same way for local execution.

**Deliverable:** A small `paths` (or `path_resolution`) module with the two functions and clear rules for when they are used (e.g. only when `sandbox_id == "local"` or when using `LocalSandboxProvider`).

---

## 5. Implement the Local Sandbox (Host Execution)

**Goal:** One concrete `Sandbox` that runs commands and file ops on the host, with optional path mappings for “container-style” paths like `/mnt/skills`.

**Steps:**

1. **Implement `LocalSandbox(Sandbox)`:**  
   - `execute_command`: run the command with `subprocess.run(..., shell=True)` (or preferred shell). Apply path mappings so that any “container” paths in the command (e.g. `/mnt/skills/...`) are rewritten to host paths before execution. After execution, optionally rewrite host paths in the output back to virtual paths so the agent sees consistent paths in messages.  
   - `read_file` / `write_file` / `list_dir`: resolve paths via your path mappings (and, if you pass thread paths into the sandbox, via thread_data in the tool layer). Use normal Python file I/O and a small `list_dir` helper (depth limit, ignore patterns).

2. **Path mappings in LocalSandbox:**  
   Use a dict, e.g. `{ "/mnt/skills": "/abs/path/to/skills" }`. Thread-specific paths (`/mnt/user-data/workspace|uploads|outputs`) are **not** stored in the sandbox; they are resolved in the **tools** (step 6) and the tool passes the already-resolved path to the sandbox. So the local sandbox only needs mappings for shared, non-thread-specific paths.

3. **Implement `LocalSandboxProvider(SandboxProvider)`:**  
   - For simplicity, use a single shared `LocalSandbox` (singleton): `acquire` returns a fixed id (e.g. `"local"`), `get` returns that instance, `release` is a no-op.  
   - Optional: pass `path_mappings` into the provider (e.g. from config) and into `LocalSandbox`.

**Deliverable:** `LocalSandbox` and `LocalSandboxProvider` in a `sandbox/local.py` (or equivalent), so you can run and test without any container.

---

## 6. Wire Tools to the Sandbox and Thread Data

**Goal:** Every sandbox-capable tool gets the right sandbox and, for local execution, the right paths.

**Steps:**

1. **Extend your runtime/context** so that tools receive:
   - `thread_id`,
   - `thread_data` (the three paths),
   - (optionally) `sandbox_id` if already acquired,
   - and a way to get the `SandboxProvider` (e.g. from config or global).

2. **Implement “ensure sandbox”:**  
   When a tool runs, check if the runtime already has a `sandbox_id` and the provider has that sandbox. If not, call `provider.acquire(thread_id)`, store `sandbox_id` in the runtime/state, then get the sandbox. This gives lazy acquisition on first tool use.

3. **Implement “ensure thread dirs”:**  
   For the local sandbox only, on first tool use ensure the three thread directories exist (e.g. `os.makedirs(..., exist_ok=True)`). Skip this for container sandboxes where the container already has the layout.

4. **Implement each tool** (e.g. `bash`, `ls`, `read_file`, `write_file`, `str_replace`):
   - Call ensure-sandbox and ensure-thread-dirs.
   - If the sandbox is local, resolve path/command with `replace_virtual_path` / `replace_virtual_paths_in_command` and pass the resolved path/command to the sandbox.
   - Call the corresponding sandbox method and return the result (or a clear error string) to the agent.

**Deliverable:** Sandbox tools that work with your agent’s tool-calling mechanism and that use virtual paths in tool descriptions and examples so the model uses `/mnt/user-data/...`.

---

## 7. Integrate with Your Agent Loop

**Goal:** Before the agent runs, thread data is set; the agent can call sandbox tools; state is consistent across turns.

**Steps:**

1. **Run a “thread data” step** at the start of each run (or at the start of each turn if you create threads on the fly): given `thread_id`, compute the three paths, create dirs if needed, and put `thread_data` into the state/context that tools will read.

2. **Do not acquire the sandbox in this step** if you use lazy acquisition; otherwise call `provider.acquire(thread_id)` here and store `sandbox_id` in state.

3. **Ensure your agent/tool layer** has access to the same state/context (including `thread_id`, `thread_data`, and after first tool use `sandbox_id`). If you use middleware, order it so thread-data runs before sandbox and before the model.

4. **Optional:** Add a “sandbox” middleware that only acquires the sandbox eagerly (no lazy) if you prefer to fail fast before the first LLM call.

**Deliverable:** A clear place in your framework where thread data is set and where sandbox_id is set (lazy or eager), and tools that receive this state.

---

## 8. (Optional) Add a Container-Based Sandbox

**Goal:** Run agent code in a real container for isolation; same virtual paths, no path rewriting inside the container.

**Steps:**

1. **Implement a container `Sandbox`** (e.g. `ContainerSandbox` or `AioSandbox`): it holds a reference to the container (or its API URL). `execute_command` and file methods call the container’s API (e.g. HTTP) or SDK. Paths passed to this sandbox are already the virtual paths the agent sees; the container should have `/mnt/user-data/workspace`, `uploads`, `outputs` (and optionally `/mnt/skills`) mounted from the host.

2. **Implement a container `SandboxProvider`:**  
   In `acquire(thread_id)`: compute thread paths on the host, start a container (or get one from a provisioner), mount the three thread dirs at `/mnt/user-data/workspace`, `/mnt/user-data/uploads`, `/mnt/user-data/outputs`, and optionally mount skills at `/mnt/skills`. Store a mapping `thread_id → sandbox_id` (and optionally persist it for multi-process). Return `sandbox_id`. In `get(sandbox_id)` return the corresponding `Sandbox` (e.g. HTTP client). In `release(sandbox_id)` stop the container and clean up.

3. **Tools:** Do **not** rewrite paths for this sandbox; the agent’s paths are valid inside the container. Only ensure the correct sandbox is acquired and that thread dirs exist on the host before starting the container.

**Deliverable:** A second provider and sandbox implementation so you can switch between “local” and “container” via config.

---

## 9. Configuration and Documentation

**Goal:** One place to configure sandbox behavior; docs so others can add providers or tools.

**Steps:**

1. **Config:** Add a section (e.g. `sandbox`) with at least:
   - Which provider to use (e.g. class or module path).
   - Base directory for thread data.
   - Virtual path prefix (if you ever want to change it).
   - For local: optional path mappings (e.g. skills). For container: image, ports, mounts, env, timeouts.

2. **Document in your repo:**  
   - The Sandbox and SandboxProvider contracts.  
   - The virtual path layout and where thread dirs live.  
   - When path resolution is applied (local vs container).  
   - How to add a new provider or a new sandbox tool.

**Deliverable:** Config schema and a short “Sandbox” section in your architecture or contributor docs.

---

## 10. Checklist Summary

- [x] **Sandbox interface** — `execute_command`, `read_file`, `write_file`, `list_dir` (and optionally `update_file`).
- [x] **SandboxProvider** — `acquire(thread_id)`, `get(sandbox_id)`, `release(sandbox_id)`; single active provider (e.g. singleton or injected).
- [x] **Virtual paths** — One prefix and three areas (workspace, uploads, outputs); document in system prompt.
- [x] **Thread data** — Per-thread dirs created and paths stored in state/context.
- [x] **Path resolution** — `replace_virtual_path` and `replace_virtual_paths_in_command`; use only for local sandbox.
- [x] **Lazy init** — First tool call acquires sandbox and stores `sandbox_id` in state (or eager in middleware).
- [x] **Local implementation** — `LocalSandbox` + `LocalSandboxProvider`; path mappings only for shared dirs; thread paths resolved in tools.
- [x] **Tools** — Bash tool uses ensure-sandbox, ensure-dirs (if local), path resolution (if local); agent context passed. (Optional: add sandbox-backed read_file/write_file/list_dir.)
- [x] **Agent integration** — Thread-data created on first tool use; context includes thread_id and sandbox_base_dir; provider set from config in main.
- [ ] **Optional container** — Second provider + sandbox; mount same layout; no path rewriting in container.
- [x] **Config** — Sandbox section in config (enabled, base_dir). [ ] **Docs** — Short “Sandbox” section in CONTRIBUTING or architecture (optional).

Use this plan in order: get the interface and provider in place, then virtual paths and thread data, then local sandbox and tools, then agent wiring, then optional container and config/docs.
