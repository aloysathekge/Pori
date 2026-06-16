"""Tests for the safe calculate tool (no eval())."""

import pytest

from pori.tools.standard.math_tools import (
    CalculateParams,
    calculate_tool,
    evaluate_expression,
)

pytestmark = pytest.mark.tools


def _calc(expression: str):
    return calculate_tool(CalculateParams(expression=expression), {})


class TestEvaluateExpression:
    def test_basic_arithmetic(self):
        assert evaluate_expression("1 + 2 * 3") == 7
        assert evaluate_expression("(1 + 2) * 3") == 9
        assert evaluate_expression("10 / 4") == 2.5
        assert evaluate_expression("10 // 3") == 3
        assert evaluate_expression("10 % 3") == 1
        assert evaluate_expression("2 ** 8") == 256

    def test_unary_operators(self):
        assert evaluate_expression("-5") == -5
        assert evaluate_expression("-(3 + 2)") == -5
        assert evaluate_expression("+7") == 7

    @pytest.mark.parametrize(
        "expr",
        [
            "__import__('os').system('echo pwned')",
            "open('/etc/passwd').read()",
            "1 if True else 2",
            "[x for x in range(3)]",
            "len('abc')",
            "lambda: 1",
            "True",
        ],
    )
    def test_rejects_non_arithmetic(self, expr):
        with pytest.raises((ValueError, SyntaxError)):
            evaluate_expression(expr)

    def test_rejects_huge_exponent(self):
        with pytest.raises(ValueError):
            evaluate_expression("10 ** 100000")


class TestCalculateTool:
    def test_returns_numeric_result(self):
        assert _calc("2 + 2") == 4

    def test_returns_error_dict_on_unsafe_input(self):
        result = _calc("__import__('os').system('echo hi')")
        assert isinstance(result, dict)
        assert "error" in result

    def test_returns_error_dict_on_syntax_error(self):
        result = _calc("2 +")
        assert isinstance(result, dict)
        assert "error" in result
