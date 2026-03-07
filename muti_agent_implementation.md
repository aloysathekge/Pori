## Multi-Agent “Team” Architecture for an Agentic Framework

This document is a **design + implementation guide** for adding a **Team (multi‑agent) abstraction** to your own agentic framework, inspired by Agno’s architecture but written in a framework‑agnostic way.

The goal is:

- To model a **Team** as a first‑class object that:
  - Orchestrates **multiple Agents** (and nested Teams),
  - Decides **who works on what**,
  - Aggregates and returns a **final answer**,
  - Plays nicely with **DB, history, memory, knowledge, tools**, etc.

You can treat this as a set of **reference notes + code skeletons** that you can adapt to your own codebase.

---

## 1. Core Concepts and Mental Model

### 1.1. Single Agent vs Team (Multi-Agent)

- **Agent**:
  - A single conversational/functional unit.
  - Has a model, instructions, tools, knowledge, memory, DB, etc.
  - Handles a request end‑to‑end: build context → call model/tools → respond.

- **Team**:
  - A **coordinator** agent that:
    - Has its own **model** (the “manager” or “conductor”).
    - Has a list of **members**: agents or other teams.
    - Uses its model to:
      - Decide **which member(s)** should work on a request.
      - Decide **what input** each member gets.
      - Possibly **iteratively refine** and combine member outputs.

You can think of a Team as:

> A higher‑level agent whose **primary tool** is “call one or more member agents and combine their results”.

### 1.2. Requirements for a Team Abstraction

Your Team implementation should:

- Support **multiple members** (`Agent` or `Team`).
- Have its own:
  - **Model** (coordinator LLM),
  - **Instructions** (how to coordinate),
  - **State / history / memory** (for team‑level conversations),
  - **Tools** (including tools for delegating tasks).
- Provide **modes** of operation, e.g.:
  - Route to a **single best member**.
  - Broadcast to **all members** and combine.
  - Perform **task‑oriented delegation** across several steps.
- Integrate with your existing:
  - **DB layer** (for sessions, runs, messages),
  - **Knowledge / RAG** (shared across team),
  - **Memory** (long‑term user knowledge),
  - **Events / tracing** (for debugging and metrics).

---

## 2. Minimal Interfaces for Agents and Teams

First, define **interfaces / base protocols**. Your concrete classes can be dataclasses, Pydantic models, or simple classes, but the **behavioral contract** should be clear.

### 2.1. Base Agent Interface

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union, Awaitable, Callable


class LLMModel(Protocol):
    """Minimal interface your LLM wrapper should expose."""

    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        ...

    async def agenerate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        ...


class Tool(Protocol):
    """Minimal interface for tools."""

    name: str
    description: str

    def call(self, **kwargs: Any) -> Any:
        ...

    async def acall(self, **kwargs: Any) -> Any:
        ...


class Database(Protocol):
    """Minimal DB abstraction for sessions / messages / runs."""

    def store_run(self, payload: Dict[str, Any]) -> None:
        ...

    async def astore_run(self, payload: Dict[str, Any]) -> None:
        ...

    def load_history(self, session_id: str, limit: int) -> List[Dict[str, Any]]:
        ...


@dataclass
class Agent:
    """Single agent abstraction."""

    name: str
    model: LLMModel
    instructions: str = "You are a helpful assistant."
    tools: List[Tool] = field(default_factory=list)
    db: Optional[Database] = None

    add_history_to_context: bool = True
    num_history_runs: int = 3

    def run(self, input_text: str, session_id: Optional[str] = None) -> str:
        """Synchronous single-agent run."""
        # 1) Build messages
        messages: List[Dict[str, str]] = [{"role": "system", "content": self.instructions}]

        if self.add_history_to_context and session_id and self.db:
            history = self.db.load_history(session_id=session_id, limit=self.num_history_runs)
            messages.extend(history)

        messages.append({"role": "user", "content": input_text})

        # 2) Call model
        raw = self.model.generate(messages=messages, tools=self._serialize_tools())

        # 3) TODO: handle tool calls if your model returns them

        output_text: str = raw.get("content", "")

        # 4) Store run
        if self.db and session_id:
            self.db.store_run(
                {
                    "session_id": session_id,
                    "agent_name": self.name,
                    "input": input_text,
                    "output": output_text,
                }
            )

        return output_text

    async def arun(self, input_text: str, session_id: Optional[str] = None) -> str:
        """Async single-agent run."""
        messages: List[Dict[str, str]] = [{"role": "system", "content": self.instructions}]

        if self.add_history_to_context and session_id and self.db:
            history = self.db.load_history(session_id=session_id, limit=self.num_history_runs)
            messages.extend(history)

        messages.append({"role": "user", "content": input_text})

        raw = await self.model.agenerate(messages=messages, tools=self._serialize_tools())
        output_text: str = raw.get("content", "")

        if self.db and session_id:
            await self.db.astore_run(
                {
                    "session_id": session_id,
                    "agent_name": self.name,
                    "input": input_text,
                    "output": output_text,
                }
            )

        return output_text

    def _serialize_tools(self) -> Optional[List[Dict[str, Any]]]:
        """Serialize tools for the model, if needed."""
        if not self.tools:
            return None
        return [
            {
                "name": t.name,
                "description": t.description,
                # Add schema if your LLM needs JSON schema here
            }
            for t in self.tools
        ]
```

These details can be expanded depending on your existing framework, but this gives a **concrete starting point**.

---

## 3. Team Abstraction: High-Level Design

### 3.1. What a Team Needs to Know

Your `Team` class needs:

- **Members**:
  - List of `Agent` or `Team` objects (or a factory function).
- **Coordinator model**:
  - An `LLMModel` instance used to decide:
    - Which member(s) should act,
    - In what order,
    - With what input.
- **Team‑level config**:
  - `instructions`: how to coordinate (team’s system prompt).
  - `mode`: routing strategy (single‑route, broadcast, task‑delegation, etc.).
  - `db`: optional team‑level DB (for its own history).
  - `add_history_to_context`, `num_history_runs`: same idea as an Agent, but at the **team level**.

### 3.2. Example Team Dataclass Skeleton

```python
from enum import Enum


class TeamMode(str, Enum):
    ROUTER = "router"        # route request to a single best member
    BROADCAST = "broadcast"  # send to all members and combine
    DELEGATE = "delegate"    # do multi-step task delegation


@dataclass
class Team:
    """Multi-agent Team abstraction."""

    name: str
    model: LLMModel
    members: List[Union[Agent, "Team"]]

    instructions: str = "You are a coordinator of specialized agents. Decide who should handle the task and combine their results."
    mode: TeamMode = TeamMode.ROUTER

    db: Optional[Database] = None
    add_history_to_context: bool = True
    num_history_runs: int = 3

    def run(self, input_text: str, session_id: Optional[str] = None) -> str:
        """Synchronously handle a request using team coordination."""
        if self.mode == TeamMode.ROUTER:
            return self._run_router(input_text, session_id)
        elif self.mode == TeamMode.BROADCAST:
            return self._run_broadcast(input_text, session_id)
        elif self.mode == TeamMode.DELEGATE:
            return self._run_delegate(input_text, session_id)
        raise ValueError(f"Unsupported team mode: {self.mode}")

    async def arun(self, input_text: str, session_id: Optional[str] = None) -> str:
        """Async variant of run."""
        if self.mode == TeamMode.ROUTER:
            return await self._arun_router(input_text, session_id)
        elif self.mode == TeamMode.BROADCAST:
            return await self._arun_broadcast(input_text, session_id)
        elif self.mode == TeamMode.DELEGATE:
            return await self._arun_delegate(input_text, session_id)
        raise ValueError(f"Unsupported team mode: {self.mode}")
```

The `mode` determines which internal strategy is used. Each strategy is implemented separately.

---

## 4. Implementing the Router Mode

### 4.1. Router Mode Concept

**Router mode** means:

- The team’s coordinator model reads the **user request** and the **list of available members**.
- It picks **one best member** to handle the request.
- That member is invoked like a normal agent, then the team may:
  - Return the member’s answer directly, or
  - Wrap it in some meta‑summary.

### 4.2. Prompting the Coordinator for Routing

You need to pass the **member descriptions** to the team model.

Example: build a description string for members:

```python
def _describe_members(self) -> str:
    descriptions = []
    for i, member in enumerate(self.members):
        role = getattr(member, "name", f"member_{i}")
        descriptions.append(f"- {role}: specializes in {getattr(member, 'instructions', 'general tasks')}")
    return "\n".join(descriptions)
```

### 4.3. Implementation: `_run_router`

```python
    def _run_router(self, input_text: str, session_id: Optional[str]) -> str:
        """Router mode: pick a single best member and call it."""
        # 1) Build coordinator messages
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.instructions},
        ]

        if self.add_history_to_context and session_id and self.db:
            history = self.db.load_history(session_id=session_id, limit=self.num_history_runs)
            messages.extend(history)

        member_descriptions = self._describe_members()
        routing_prompt = (
            "You are managing a team of experts. "
            "Given the user request, choose the single best member to handle it.\n\n"
            "Available members:\n"
            f"{member_descriptions}\n\n"
            "Respond with JSON: {\"member_name\": \"name\", \"reason\": \"why\"}."
        )

        messages.append({"role": "user", "content": routing_prompt + "\n\nUser request:\n" + input_text})

        # 2) Call coordinator model
        raw = self.model.generate(messages=messages)
        routing_text = raw.get("content", "")

        # 3) Parse JSON response (you should add error handling)
        import json

        try:
            routing = json.loads(routing_text)
            member_name = routing.get("member_name")
        except Exception:
            # Fallback: default to first member
            member_name = getattr(self.members[0], "name", None)

        # 4) Find member by name
        member = self._find_member_by_name(member_name) or self.members[0]

        # 5) Call the chosen member
        member_output = member.run(input_text=input_text, session_id=session_id)

        # 6) Optionally store a team-level run
        if self.db and session_id:
            self.db.store_run(
                {
                    "session_id": session_id,
                    "team_name": self.name,
                    "mode": "router",
                    "routed_to": getattr(member, "name", None),
                    "input": input_text,
                    "output": member_output,
                }
            )

        return member_output

    def _find_member_by_name(self, name: Optional[str]) -> Optional[Union[Agent, "Team"]]:
        if not name:
            return None
        for m in self.members:
            if getattr(m, "name", None) == name:
                return m
        return None
```

### 4.4. Async Variant

```python
    async def _arun_router(self, input_text: str, session_id: Optional[str]) -> str:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.instructions},
        ]

        if self.add_history_to_context and session_id and self.db:
            history = self.db.load_history(session_id=session_id, limit=self.num_history_runs)
            messages.extend(history)

        member_descriptions = self._describe_members()
        routing_prompt = (
            "You are managing a team of experts. "
            "Given the user request, choose the single best member to handle it.\n\n"
            "Available members:\n"
            f"{member_descriptions}\n\n"
            "Respond with JSON: {\"member_name\": \"name\", \"reason\": \"why\"}."
        )

        messages.append({"role": "user", "content": routing_prompt + "\n\nUser request:\n" + input_text})

        raw = await self.model.agenerate(messages=messages)
        routing_text = raw.get("content", "")

        import json

        try:
            routing = json.loads(routing_text)
            member_name = routing.get("member_name")
        except Exception:
            member_name = getattr(self.members[0], "name", None)

        member = self._find_member_by_name(member_name) or self.members[0]
        member_output = await member.arun(input_text=input_text, session_id=session_id)

        if self.db and session_id:
            await self.db.astore_run(
                {
                    "session_id": session_id,
                    "team_name": self.name,
                    "mode": "router",
                    "routed_to": getattr(member, "name", None),
                    "input": input_text,
                    "output": member_output,
                }
            )

        return member_output
```

---

## 5. Implementing Broadcast Mode

### 5.1. Broadcast Mode Concept

**Broadcast mode** means:

- Every member receives the **same input**.
- They all run **independently**.
- The team then **aggregates** the results:
  - Summarize consensus,
  - Merge answers,
  - Or pick the best.

### 5.2. Synchronous Broadcast Implementation

```python
    def _run_broadcast(self, input_text: str, session_id: Optional[str]) -> str:
        """Send the same request to all members and combine their outputs."""
        # 1) Run all members
        member_outputs = []
        for member in self.members:
            output = member.run(input_text=input_text, session_id=session_id)
            member_outputs.append(
                {
                    "name": getattr(member, "name", "unknown"),
                    "output": output,
                }
            )

        # 2) Ask the coordinator model to summarize / combine
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.instructions},
            {
                "role": "user",
                "content": (
                    "You received answers from multiple experts. "
                    "Combine them into a single high-quality answer.\n\n"
                    f"User request:\n{input_text}\n\n"
                    f"Expert answers:\n{member_outputs}"
                ),
            },
        ]

        raw = self.model.generate(messages=messages)
        combined_output = raw.get("content", "")

        if self.db and session_id:
            self.db.store_run(
                {
                    "session_id": session_id,
                    "team_name": self.name,
                    "mode": "broadcast",
                    "member_outputs": member_outputs,
                    "combined_output": combined_output,
                }
            )

        return combined_output
```

### 5.3. Async Broadcast (Optional Parallelism)

```python
    async def _arun_broadcast(self, input_text: str, session_id: Optional[str]) -> str:
        """Async broadcast; you can choose to run members in parallel."""
        import asyncio

        async def run_member(m: Union[Agent, "Team"]) -> Dict[str, str]:
            out = await m.arun(input_text=input_text, session_id=session_id)
            return {"name": getattr(m, "name", "unknown"), "output": out}

        member_outputs = await asyncio.gather(*(run_member(m) for m in self.members))

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.instructions},
            {
                "role": "user",
                "content": (
                    "You received answers from multiple experts. "
                    "Combine them into a single high-quality answer.\n\n"
                    f"User request:\n{input_text}\n\n"
                    f"Expert answers:\n{member_outputs}"
                ),
            },
        ]

        raw = await self.model.agenerate(messages=messages)
        combined_output = raw.get("content", "")

        if self.db and session_id:
            await self.db.astore_run(
                {
                    "session_id": session_id,
                    "team_name": self.name,
                    "mode": "broadcast",
                    "member_outputs": member_outputs,
                    "combined_output": combined_output,
                }
            )

        return combined_output
```

---

## 6. Implementing Delegate (Task-Oriented) Mode

### 6.1. Delegate Mode Concept

**Delegate mode** is more involved:

- The coordinator creates a **plan** with tasks.
- Each task is assigned to specific member(s).
- Members are called sequentially or iteratively.
- The coordinator aggregates results and may **refine** the plan on the fly.

This mode resembles a **mini-workflow engine**, but the orchestration decisions are made **by the LLM**, not by fixed code.

### 6.2. High-Level Flow

1. Coordinator model receives:
   - User request.
   - Member list.
2. It returns a **plan structure**, e.g. JSON list of steps:
   - `[{ "step": 1, "task": "...", "assignee": "Researcher" }, ...]`
3. The Team iterates:
   - For each step:
     - Find assignee(s),
     - Call them with appropriate input (task + context),
     - Collect and store their outputs.
4. After all steps:
   - Coordinator model generates final answer summarizing all work.

### 6.3. Example Delegate Implementation (Simplified)

```python
    def _run_delegate(self, input_text: str, session_id: Optional[str]) -> str:
        """LLM-generated multi-step plan with delegation to members."""
        # 1) Ask coordinator for a plan
        member_descriptions = self._describe_members()
        planning_prompt = (
            "You are a project manager for a team of AI experts. "
            "Given the user request and the list of members, create a step-by-step plan. "
            "Each step must include a 'task' and an 'assignee' (member name).\n\n"
            "Members:\n"
            f"{member_descriptions}\n\n"
            "User request:\n"
            f"{input_text}\n\n"
            "Respond with JSON: {\"steps\": [{\"step\": 1, \"task\": \"...\", \"assignee\": \"member_name\"}, ...]}."
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": planning_prompt},
        ]

        raw_plan = self.model.generate(messages=messages)
        plan_text = raw_plan.get("content", "")

        import json

        try:
            plan = json.loads(plan_text)
            steps = plan.get("steps", [])
        except Exception:
            steps = []

        step_outputs = []

        # 2) Execute each step by delegating to members
        for step in steps:
            assignee_name = step.get("assignee")
            task = step.get("task", "")
            member = self._find_member_by_name(assignee_name) or self.members[0]

            # Build task-specific input, including original user request
            member_input = f"Original user request: {input_text}\n\nYour task: {task}"
            out = member.run(input_text=member_input, session_id=session_id)

            step_outputs.append(
                {
                    "step": step.get("step"),
                    "assignee": getattr(member, "name", assignee_name),
                    "task": task,
                    "output": out,
                }
            )

        # 3) Ask coordinator to summarize all steps into final answer
        summary_prompt = (
            "You coordinated a multi-step plan executed by experts. "
            "Summarize all steps and provide a final answer to the user.\n\n"
            f"User request:\n{input_text}\n\n"
            f"Execution trace:\n{step_outputs}"
        )

        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": summary_prompt},
        ]

        raw_summary = self.model.generate(messages=messages)
        final_output = raw_summary.get("content", "")

        if self.db and session_id:
            self.db.store_run(
                {
                    "session_id": session_id,
                    "team_name": self.name,
                    "mode": "delegate",
                    "plan": steps,
                    "step_outputs": step_outputs,
                    "final_output": final_output,
                }
            )

        return final_output
```

Async version would mirror this using `await` calls.

---

## 7. Persistence, History, and DB Strategy

### 7.1. Shared vs Per-Entity DB

You have two main approaches:

- **Single shared DB**:
  - One DB instance used by all agents and teams.
  - Simpler deployment, centralized data.
- **Per‑entity DB**:
  - Some agents/teams use custom DB backends (e.g., specialized metrics store).
  - More complex but can be powerful.

**Recommendation** for most frameworks:

- Use a **single Postgres (or similar) DB**.
- Give each table a clear purpose:
  - `sessions`, `messages`, `runs`, `memories`, `knowledge_docs`, etc.
- Agents and Teams should share that DB unless you have a strong reason not to.

### 7.2. Team and Agent History

Decide:

- Do you want **team-level history** to include:
  - Only user↔team messages, or
  - Also summaries of member interactions?
- Do you want **member-level history** to persist:
  - Only user↔member messages, or
  - Also meta-context about being called from a team?

Typical pattern:

- **Team history**: user↔team, plus a short log of “which member handled what”.
- **Member history**: from the member’s perspective, treat each delegation as a regular user request (with the team as the “user”).

Implementation detail:

- Use **different `agent_name` / `team_name` fields** in DB rows to separate logs.

---

## 8. Integration with a Higher-Level Runtime (Optional)

If your framework has an **“OS” or runtime layer** (like `AgentOS` in Agno) that:

- Hosts multiple agents/teams/workflows,
- Exposes HTTP APIs,
- Manages DB/knowledge auto-discovery,

Then ensure:

- **IDs are unique** across entities of the same type (agents, teams, workflows).
- The OS can:
  - Autodiscover Agents and Teams from configuration.
  - Inject a **shared DB** into them if they didn’t specify one.
  - Autodiscover and validate **Knowledge** instances (no duplicate `(name, db, table)`).

This keeps your multi-agent system:

- **Composable** (teams of agents, teams of teams),
- **Configurable** (YAML/JSON or Python),
- **Scalable** (stateless API workers using shared DB + vector DB).

---

## 9. Best Practices and Design Notes

1. **Create once, reuse**:
   - Instantiate Agents and Teams **once at startup**, not per request.
   - This is critical for performance and for features like history/memory.

2. **Keep roles clear**:
   - Give each member a **strong role** via `name` and `instructions`.
   - Avoid making all members generic; specialization helps the coordinator reason better.

3. **Start with Router or Broadcast**:
   - Implement `ROUTER` and `BROADCAST` modes first; they are simpler.
   - Only add `DELEGATE` mode after you are comfortable with multi-step reasoning and JSON planning.

4. **Use JSON / structured responses**:
   - For routing and planning, always have the coordinator respond with **JSON schemas**.
   - This makes it easier to parse and robust against variations.

5. **Trace everything**:
   - Log:
     - Which member was chosen, with what reason,
     - What each member returned,
     - How the final answer was formed.
   - This is invaluable for debugging and evaluation.

6. **Uniform interfaces**:
   - Ensure `Agent` and `Team` both expose `run` / `arun` with the same signature.
   - This allows **nesting**: teams can be members of other teams.

7. **Avoid circular delegation**:
   - Design your system so that:
     - A team cannot end up delegating back into itself in a loop.
   - For example, keep a **call depth counter** or list of visited entities in context.

---

## 10. Putting It All Together – Example Usage

```python
# 1) Define base agents
researcher = Agent(
    name="Researcher",
    model=my_llm_model,
    instructions="You are a world-class researcher. Search, synthesize, and explain clearly.",
    db=my_db,
)

writer = Agent(
    name="Writer",
    model=my_llm_model,
    instructions="You are an excellent technical writer. Turn ideas into clear, well-structured text.",
    db=my_db,
)

critic = Agent(
    name="Critic",
    model=my_llm_model,
    instructions="You review and critique answers for correctness, clarity, and completeness.",
    db=my_db,
)

# 2) Define a team
content_team = Team(
    name="Content Team",
    model=my_llm_model,  # coordinator
    members=[researcher, writer, critic],
    mode=TeamMode.DELEGATE,
    db=my_db,
)

# 3) Use the team in your framework
user_request = "Write a detailed article about how multi-agent LLM teams work."
session_id = "user-123-session-1"

answer = content_team.run(user_request, session_id=session_id)
print(answer)
```

This shows how:

- You create **specialized agents**.
- You compose them into a **Team** that uses a **delegate** strategy.
- The coordinator:
  - Plans steps,
  - Delegates,
  - Aggregates,
  - Returns the final result.

You can now treat `Team` as a **first-class citizen** in your own agentic framework, alongside `Agent` and (if you add it later) **Workflow** abstractions.

