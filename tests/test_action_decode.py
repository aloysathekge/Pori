"""Decoding a model-emitted JSON action envelope (multi-provider robustness)."""

from pori.utils.action_decode import decode_action_envelope, looks_like_action_envelope

ENVELOPE = (
    '{"current_state": {"evaluation_previous_goal": "start", '
    '"memory": "user name is Aloy", "next_goal": "store it"}, '
    '"action": [{"core_memory_append": {"label": "human", "content": "Name: Aloy"}}]}'
)


class TestDecode:
    def test_decodes_action_envelope(self):
        actions = decode_action_envelope(ENVELOPE)
        assert actions == [
            {"core_memory_append": {"label": "human", "content": "Name: Aloy"}}
        ]

    def test_decodes_with_markdown_fence(self):
        actions = decode_action_envelope(f"```json\n{ENVELOPE}\n```")
        assert actions is not None and actions[0].get("core_memory_append")

    def test_decodes_answer_action(self):
        text = (
            '{"action": [{"answer": {"final_answer": "Hi Aloy!", "reasoning": "x"}}]}'
        )
        actions = decode_action_envelope(text)
        assert actions == [{"answer": {"final_answer": "Hi Aloy!", "reasoning": "x"}}]

    def test_multiple_actions(self):
        text = '{"action": [{"a": {"x": 1}}, {"b": {"y": 2}}]}'
        assert decode_action_envelope(text) == [{"a": {"x": 1}}, {"b": {"y": 2}}]

    def test_prose_is_not_an_envelope(self):
        assert decode_action_envelope("Nice to meet you, Aloy!") is None

    def test_prose_mentioning_json_is_safe(self):
        # A normal reply that merely talks about JSON must not be parsed.
        assert (
            decode_action_envelope('Here is an example: {"action": "do stuff"}')
            is None  # not a bare JSON object
        )

    def test_action_not_a_list_rejected(self):
        assert decode_action_envelope('{"action": "answer"}') is None

    def test_malformed_json_rejected(self):
        assert decode_action_envelope('{"action": [oops]}') is None

    def test_bad_item_shape_rejected(self):
        assert decode_action_envelope('{"action": [{"a": 1, "b": 2}]}') is None
        assert decode_action_envelope('{"action": [{"a": "not-a-dict"}]}') is None

    def test_empty_and_none(self):
        assert decode_action_envelope("") is None
        assert decode_action_envelope("   ") is None


class TestLooksLike:
    def test_true_for_envelope(self):
        assert looks_like_action_envelope(ENVELOPE) is True
        assert looks_like_action_envelope(f"```json\n{ENVELOPE}\n```") is True

    def test_false_for_prose(self):
        assert looks_like_action_envelope("Nice to meet you!") is False
        assert looks_like_action_envelope("") is False
