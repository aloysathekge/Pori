import json

import pytest
from pori_cloud.streaming import stream_agent_execution

pytestmark = pytest.mark.asyncio


class _Memory:
    def get_final_answer(self):
        return {"final_answer": "Done", "reasoning": "Complete"}


class _PlanItem:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


class _PlanStore:
    def __init__(self, items):
        self._items = [_PlanItem(i) for i in items]

    def items(self):
        return tuple(self._items)


class _Agent:
    def __init__(self, plan):
        self.memory = _Memory()
        self.plan_store = _PlanStore(plan)


class _Orchestrator:
    def __init__(self, plan):
        self.agents = {}
        self._plan = plan

    async def execute_task(self, **kwargs):
        agent = _Agent(self._plan)
        return {
            "success": True,
            "steps_taken": 1,
            "agent": agent,
            "plan": self._plan,
            "result": {"metrics": None, "plan": self._plan},
            "trace": {},
        }


class _Settings:
    max_steps = 3


async def test_stream_final_message_includes_plan():
    plan = [{"id": "1", "content": "answer the user", "status": "completed"}]
    events = []
    async for event in stream_agent_execution(
        _Orchestrator(plan),
        task="test",
        settings=_Settings(),
    ):
        events.append(event)

    message_event = next(
        event for event in events if event.startswith("event: message\n")
    )
    message_data = json.loads(message_event.split("data: ", 1)[1].split("\n", 1)[0])
    assert message_data["plan"] == plan
    assert "runtime_events" not in message_data
