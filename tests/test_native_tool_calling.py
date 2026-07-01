"""Phase B native tool-calling: type/schema foundations + native path (B.1, B.2)."""

import asyncio

from pori.agent import Agent, AgentSettings
from pori.llm import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    ToolTurn,
    UserMessage,
)
from pori.memory import AgentMemory
from pori.tools.registry import ToolRegistry
from pori.tools.standard import register_all_tools


def _registry():
    r = ToolRegistry()
    register_all_tools(r)
    return r


# --- B.1: types + schemas ---------------------------------------------------


def test_tool_call_defaults():
    tc = ToolCall(name="write_file")
    assert tc.name == "write_file"
    assert tc.id == ""
    assert tc.arguments == {}


def test_assistant_message_carries_tool_calls():
    msg = AssistantMessage(
        content="Writing the file",
        tool_calls=[ToolCall(id="t1", name="write_file", arguments={"path": "a"})],
    )
    assert msg.role == "assistant"
    assert msg.tool_calls[0].name == "write_file"
    assert AssistantMessage(content="hi").tool_calls == []


def test_tool_result_message():
    msg = ToolResultMessage(tool_call_id="t1", content="wrote a")
    assert msg.role == "tool"
    assert msg.tool_call_id == "t1"


def test_tool_turn_holds_text_and_calls():
    turn = ToolTurn(
        text="Saving the report",
        tool_calls=[ToolCall(name="write_file", arguments={"path": "r.md"})],
    )
    assert turn.text == "Saving the report"
    assert turn.tool_calls[0].arguments["path"] == "r.md"
    assert ToolTurn().tool_calls == []


def test_registry_tool_schemas_shape():
    schemas = _registry().tool_schemas()
    by_name = {s["name"]: s for s in schemas}
    wf = by_name["write_file"]
    assert set(wf) == {"name", "description", "input_schema"}
    assert wf["description"]
    assert wf["input_schema"]["type"] == "object"
    assert "file_path" in wf["input_schema"]["properties"]


# --- B.2: Anthropic native parsing ------------------------------------------


class _Block:
    def __init__(self, type, **kw):
        self.type = type
        for k, v in kw.items():
            setattr(self, k, v)


class _Resp:
    def __init__(self, content, usage=None):
        self.content = content
        self.usage = usage


class _FakeMessages:
    def __init__(self, resp):
        self._resp = resp
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._resp


class _FakeClient:
    def __init__(self, resp):
        self.messages = _FakeMessages(resp)


def test_anthropic_ainvoke_tools_parses_text_and_tool_use():
    from pori.llm.anthropic import ChatAnthropic

    llm = ChatAnthropic(api_key="x", model="claude-test")
    llm._client = _FakeClient(
        _Resp(
            [
                _Block("text", text="Saving the file"),
                _Block(
                    "tool_use", id="tu1", name="write_file", input={"file_path": "a"}
                ),
            ]
        )
    )
    turn = asyncio.run(
        llm.ainvoke_tools(
            [SystemMessage(content="sys"), UserMessage(content="hi")],
            [
                {
                    "name": "write_file",
                    "description": "d",
                    "input_schema": {"type": "object"},
                }
            ],
        )
    )
    assert turn.text == "Saving the file"
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "write_file"
    assert turn.tool_calls[0].arguments == {"file_path": "a"}
    # system + tools were forwarded to the API.
    sent = llm._client.messages.last_kwargs
    assert sent["system"] == "sys"
    assert sent["tools"][0]["name"] == "write_file"


# --- B.2: agent native branch -----------------------------------------------


class _NativeMockLLM:
    def __init__(self, turns):
        self._turns = turns
        self.i = 0
        self.model = "mock"
        self.last_usage = None

    async def ainvoke_tools(self, messages, tools):
        turn = self._turns[min(self.i, len(self._turns) - 1)]
        self.i += 1
        return turn

    def with_structured_output(self, *a, **k):  # pragma: no cover - native skips this
        raise AssertionError("native mode must not use structured output")


def test_native_branch_maps_tool_calls_to_actions():
    llm = _NativeMockLLM(
        [
            ToolTurn(
                text="Writing the file",
                tool_calls=[
                    ToolCall(
                        name="write_file", arguments={"file_path": "a", "content": "x"}
                    )
                ],
            )
        ]
    )
    agent = Agent(
        task="t",
        llm=llm,
        tools_registry=_registry(),
        settings=AgentSettings(max_steps=2),
        memory=AgentMemory(),
    )
    out = asyncio.run(agent.get_next_action())
    assert out.action == [{"write_file": {"file_path": "a", "content": "x"}}]
    assert out.current_state == {"next_goal": "Writing the file"}


def test_native_mode_end_to_end_answer():
    llm = _NativeMockLLM(
        [
            ToolTurn(
                text="Answering the user",
                tool_calls=[
                    ToolCall(
                        name="answer",
                        arguments={"final_answer": "42", "reasoning": "because"},
                    )
                ],
            )
        ]
    )
    memory = AgentMemory()
    agent = Agent(
        task="what is the answer?",
        llm=llm,
        tools_registry=_registry(),
        settings=AgentSettings(max_steps=3),
        memory=memory,
    )
    result = asyncio.run(agent.run())
    assert result["completed"] is True
    assert memory.get_state("final_answer")["final_answer"] == "42"
    # The assistant text became the activity line.
    assert agent.state.current_activity == "Answering the user"


# --- B.3: OpenAI / Fireworks / Google native --------------------------------


class _OAIFunc:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _OAIToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _OAIFunc(name, arguments)


class _OAIMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _OAIChoice:
    def __init__(self, message):
        self.message = message


class _OAIResp:
    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage


class _OAICompletions:
    def __init__(self, resp):
        self._resp = resp
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._resp


class _OAIClient:
    def __init__(self, resp):
        self.chat = type("Chat", (), {"completions": _OAICompletions(resp)})()


def _openai_resp_with_tool_call():
    return _OAIResp(
        [
            _OAIChoice(
                _OAIMessage(
                    content="Saving the file",
                    tool_calls=[_OAIToolCall("c1", "write_file", '{"file_path": "a"}')],
                )
            )
        ]
    )


def test_openai_ainvoke_tools_parses_tool_calls():
    from pori.llm.openai import ChatOpenAI

    llm = ChatOpenAI(api_key="x", model="gpt-test")
    llm._client = _OAIClient(_openai_resp_with_tool_call())
    turn = asyncio.run(
        llm.ainvoke_tools(
            [SystemMessage(content="sys"), UserMessage(content="hi")],
            [
                {
                    "name": "write_file",
                    "description": "d",
                    "input_schema": {"type": "object"},
                }
            ],
        )
    )
    assert turn.text == "Saving the file"
    assert turn.tool_calls[0].name == "write_file"
    assert turn.tool_calls[0].arguments == {"file_path": "a"}
    sent = llm._client.chat.completions.last_kwargs
    assert sent["tools"][0]["function"]["name"] == "write_file"
    assert sent["tool_choice"] == "auto"


def test_fireworks_inherits_native_tool_calling():
    from pori.llm.fireworks import ChatFireworks

    llm = ChatFireworks(api_key="x", model="accounts/fireworks/models/kimi-k2p6")
    llm._client = _OAIClient(_openai_resp_with_tool_call())
    turn = asyncio.run(
        llm.ainvoke_tools(
            [UserMessage(content="hi")],
            [
                {
                    "name": "write_file",
                    "description": "d",
                    "input_schema": {"type": "object"},
                }
            ],
        )
    )
    assert turn.tool_calls[0].name == "write_file"


def test_gemini_schema_sanitizer_strips_unsupported_keys():
    from pori.llm.google import _sanitize_gemini_schema

    cleaned = _sanitize_gemini_schema(
        {
            "title": "X",
            "type": "object",
            "additionalProperties": False,
            "properties": {"a": {"title": "A", "type": "string"}},
        }
    )
    assert "title" not in cleaned
    assert "additionalProperties" not in cleaned
    assert "title" not in cleaned["properties"]["a"]
    assert cleaned["properties"]["a"]["type"] == "string"


# --- B.4: native prompt + config flag ---------------------------------------


def test_prompt_is_native_only():
    agent = Agent(
        task="t",
        llm=_NativeMockLLM([]),
        tools_registry=_registry(),
        settings=AgentSettings(max_steps=2),
        memory=AgentMemory(),
    )
    sm = agent.system_message
    assert "JSON Output Format" not in sm
    assert "{tool_descriptions}" not in sm
    assert "native tool-calling ability" in sm
    assert "Workflow" in sm  # workflow/rules retained


def test_text_only_turn_becomes_answer():
    """A reply with text but no tool calls is treated as the final answer."""
    llm = _NativeMockLLM(
        [ToolTurn(text="The files were deleted; nothing to show.", tool_calls=[])]
    )
    out = asyncio.run(
        Agent(
            task="show me the todo",
            llm=llm,
            tools_registry=_registry(),
            settings=AgentSettings(max_steps=2),
            memory=AgentMemory(),
        ).get_next_action()
    )
    assert out.action == [
        {
            "answer": {
                "final_answer": "The files were deleted; nothing to show.",
                "reasoning": "",
            }
        }
    ]


def test_text_only_turn_completes_run():
    llm = _NativeMockLLM(
        [ToolTurn(text="Nothing to show — they were deleted.", tool_calls=[])]
    )
    memory = AgentMemory()
    result = asyncio.run(
        Agent(
            task="show me the todo",
            llm=llm,
            tools_registry=_registry(),
            settings=AgentSettings(max_steps=3),
            memory=memory,
        ).run()
    )
    assert result["completed"] is True
    assert "deleted" in memory.get_state("final_answer")["final_answer"]


# --- A.1: OpenAI streaming (on_delta) ---------------------------------------


class _StreamDeltaToolCall:
    def __init__(self, index, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class _StreamChunk:
    def __init__(self, content=None, tool_calls=None, usage=None):
        if content is not None or tool_calls is not None:
            delta = type("D", (), {"content": content, "tool_calls": tool_calls})()
            self.choices = [type("C", (), {"delta": delta})()]
        else:
            self.choices = []
        self.usage = usage


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def _gen():
            for c in self._chunks:
                yield c

        return _gen()


class _StreamCompletions:
    def __init__(self, chunks):
        self._chunks = chunks
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeStream(self._chunks)


class _StreamClient:
    def __init__(self, chunks):
        self.chat = type("Chat", (), {"completions": _StreamCompletions(chunks)})()


def test_openai_streaming_forwards_deltas_and_assembles_tool_call():
    from pori.llm.openai import ChatOpenAI

    chunks = [
        _StreamChunk(content="Sav"),
        _StreamChunk(content="ing"),
        _StreamChunk(
            tool_calls=[
                _StreamDeltaToolCall(
                    0, id="c1", name="write_file", arguments='{"file_path":'
                )
            ]
        ),
        _StreamChunk(tool_calls=[_StreamDeltaToolCall(0, arguments='"a"}')]),
        _StreamChunk(
            usage=type(
                "U",
                (),
                {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            )()
        ),
    ]
    llm = ChatOpenAI(api_key="x", model="m")
    llm._client = _StreamClient(chunks)

    from pori.observability import TEXT_DELTA, TOOL_CALL_START

    events: list = []
    turn = asyncio.run(
        llm.ainvoke_tools(
            [SystemMessage(content="s"), UserMessage(content="hi")],
            [
                {
                    "name": "write_file",
                    "description": "d",
                    "input_schema": {"type": "object"},
                }
            ],
            on_event=events.append,
        )
    )

    text = "".join(e.payload["text"] for e in events if e.type == TEXT_DELTA)
    starts = [e for e in events if e.type == TOOL_CALL_START]
    assert text == "Saving"  # text streamed chunk-by-chunk
    # tool announced the instant its name arrived (before args finished)
    assert [s.payload["name"] for s in starts] == ["write_file"]
    assert turn.text == "Saving"
    assert turn.tool_calls[0].arguments == {"file_path": "a"}
    assert llm._client.chat.completions.last_kwargs["stream"] is True
    assert llm.last_usage["total_tokens"] == 7


class _StreamingMockLLM:
    def __init__(self, turns):
        self._turns = turns
        self.i = 0
        self.model = "mock"
        self.last_usage = None

    async def ainvoke_tools(self, messages, tools, on_event=None):
        from pori.observability import TEXT_DELTA, PoriEvent

        turn = self._turns[min(self.i, len(self._turns) - 1)]
        self.i += 1
        if on_event and turn.text:
            for word in turn.text.split(" "):
                on_event(PoriEvent(TEXT_DELTA, {"text": word + " "}))
        return turn

    def with_structured_output(self, *a, **k):  # pragma: no cover
        raise AssertionError("native mode must not use structured output")


def test_agent_forwards_events_when_streaming():
    from pori.observability import TEXT_DELTA

    llm = _StreamingMockLLM(
        [
            ToolTurn(
                text="hello world",
                tool_calls=[
                    ToolCall(
                        name="answer",
                        arguments={"final_answer": "hi", "reasoning": "r"},
                    )
                ],
            )
        ]
    )
    agent = Agent(
        task="t",
        llm=llm,
        tools_registry=_registry(),
        settings=AgentSettings(max_steps=2),
        memory=AgentMemory(),
    )
    events: list = []
    asyncio.run(agent.run(on_event=events.append, stream=True))
    text = "".join(e.payload["text"] for e in events if e.type == TEXT_DELTA).strip()
    assert text == "hello world"


# --- P3: JSONL event sink + lifecycle events --------------------------------


def test_jsonl_event_sink_writes_lines(tmp_path):
    import json as _json

    from pori.observability import (
        TEXT_DELTA,
        TOOL_CALL_START,
        JsonlEventSink,
        PoriEvent,
    )

    path = tmp_path / "events.jsonl"
    sink = JsonlEventSink(str(path))
    sink(PoriEvent(TEXT_DELTA, {"text": "hi"}, step=2))
    sink(PoriEvent(TOOL_CALL_START, {"name": "answer"}))

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert _json.loads(lines[0]) == {
        "type": TEXT_DELTA,
        "payload": {"text": "hi"},
        "step": 2,
    }


def test_agent_emits_lifecycle_events_without_streaming():
    from pori.observability import RUN_END, RUN_START, TOOL_CALL_END

    llm = _NativeMockLLM(
        [
            ToolTurn(
                text="",
                tool_calls=[
                    ToolCall(
                        name="answer",
                        arguments={"final_answer": "hi", "reasoning": "r"},
                    )
                ],
            )
        ]
    )
    agent = Agent(
        task="t",
        llm=llm,
        tools_registry=_registry(),
        settings=AgentSettings(max_steps=2),
        memory=AgentMemory(),
    )
    events: list = []
    # stream defaults to False: the (non-streaming) mock is never asked to stream,
    # yet lifecycle events still flow via the agent's own _emit.
    asyncio.run(agent.run(on_event=events.append))
    types = [e.type for e in events]
    assert types[0] == RUN_START
    assert RUN_END in types
    assert TOOL_CALL_END in types


# --- P4: reasoning-tag scrubber ---------------------------------------------


def _run_scrubber(text, chunk_size):
    from pori.llm.reasoning import StreamingThinkScrubber

    s = StreamingThinkScrubber()
    segs = []
    for i in range(0, len(text), chunk_size):
        segs += s.feed(text[i : i + chunk_size])
    segs += s.flush()
    visible = "".join(t for k, t in segs if k == "text")
    thinking = "".join(t for k, t in segs if k == "thinking")
    return visible, thinking


def test_think_scrubber_splits_across_chunk_boundaries():
    text = "Hi <think>secret plan</think>there!"
    for cs in (1, 2, 3, 5, 8, 100):
        visible, thinking = _run_scrubber(text, cs)
        assert visible == "Hi there!", cs
        assert thinking == "secret plan", cs


def test_think_scrubber_plain_text_untouched():
    visible, thinking = _run_scrubber("just an answer, no tags at all", 4)
    assert visible == "just an answer, no tags at all"
    assert thinking == ""


def test_openai_streaming_tagged_reasoning_splits_thinking():
    from pori.llm.openai import ChatOpenAI
    from pori.observability import TEXT_DELTA, THINKING_DELTA

    chunks = [
        _StreamChunk(content="<think>plan"),
        _StreamChunk(content="ning</think>Here"),
        _StreamChunk(content=" is it."),
    ]
    llm = ChatOpenAI(api_key="x", model="m", reasoning_mode="tagged")
    llm._client = _StreamClient(chunks)
    events: list = []
    turn = asyncio.run(
        llm.ainvoke_tools(
            [UserMessage(content="hi")],
            [
                {
                    "name": "answer",
                    "description": "d",
                    "input_schema": {"type": "object"},
                }
            ],
            on_event=events.append,
        )
    )
    thinking = "".join(e.payload["text"] for e in events if e.type == THINKING_DELTA)
    text = "".join(e.payload["text"] for e in events if e.type == TEXT_DELTA)
    assert thinking == "planning"
    assert text == "Here is it."
    # reasoning is excluded from the answer text
    assert turn.text == "Here is it."
