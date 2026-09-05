"""Comparing two numbers and reporting the outcome as a boolean."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import NUMBER

#: How close two numbers have to be for ``==`` to hold. A float carried through arithmetic
#: rarely lands on a round value, so an exact test on one is a test that almost never passes.
TOLERANCE = 1e-9

#: The tests, in the order the widget offers them, mapped to what each one asks.
COMPARISONS = {
    "a > b": lambda a, b: a > b,
    "a >= b": lambda a, b: a >= b,
    "a < b": lambda a, b: a < b,
    "a <= b": lambda a, b: a <= b,
    "a == b": lambda a, b: abs(a - b) <= TOLERANCE,
    "a != b": lambda a, b: abs(a - b) > TOLERANCE,
}


class LogicCompareNumbers(io.ComfyNode):
    """Emit whether a comparison of two numbers holds."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLogicCompareNumbers",
            display_name="Logic Compare Numbers",
            search_aliases=[
                "WASLogicCompareNumbers", "Logic Compare Numbers",
                "compare numbers",
                "greater than",
                "less than",
                "equals",
                "number condition",
                "number to boolean",
                "loop condition",
            ],
            category="WAS Suite/Logic/Boolean",
            description=(
                "Compare two numbers and report whether the test holds as a true or false "
                "value. Wire the result into While Loop Close to end a loop on a count, or into "
                "any node taking a boolean."
            ),
            inputs=[
                io.MultiType.Input(
                    io.Float.Input("number_a", default=0.0, min=-1e18, max=1e18, step=0.01),
                    [io.Float, io.Int, NUMBER],
                    tooltip=(
                        "The number on the left of the test; FLOAT, INT or NUMBER. Type it, or "
                        "wire it from a counter or a measurement."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("number_b", default=0.0, min=-1e18, max=1e18, step=0.01),
                    [io.Float, io.Int, NUMBER],
                    tooltip=(
                        "The number on the right of the test; FLOAT, INT or NUMBER. Type it, or "
                        "wire it from a counter or a measurement."
                    ),
                ),
                io.Combo.Input(
                    "comparison",
                    list(COMPARISONS),
                    tooltip=(
                        "Which test to apply; COMBO. 'a' is number_a and 'b' is number_b, so "
                        "'a > b' is true while number_a is the larger of the two."
                    ),
                ),
            ],
            outputs=[
                io.Boolean.Output(
                    display_name="boolean",
                    tooltip=(
                        "Whether the test holds; BOOLEAN. Equality allows a millionth of a "
                        "millionth either way, so a computed float still matches."
                    ),
                ),
                io.String.Output(
                    display_name="comparison_text",
                    tooltip=(
                        "The test and its outcome in words; STRING, such as '3 > 2 is true'. For "
                        "a readout or a filename."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, number_a=0.0, number_b=0.0, comparison="a > b") -> io.NodeOutput:
        """Apply the chosen comparison.

        Raises:
            ValueError: ``comparison`` names a test that does not exist, or an input holds
                something that is not a number.
        """
        test = COMPARISONS.get(str(comparison))
        if test is None:
            raise ValueError(
                f"Logic Compare Numbers does not have a comparison called "
                f"'{comparison}'. Choose one of {', '.join(COMPARISONS)}."
            )
        left = _as_number(number_a, "number_a")
        right = _as_number(number_b, "number_b")
        outcome = bool(test(left, right))
        spelled = str(comparison).replace("a", _spell(left), 1)
        spelled = spelled.replace("b", _spell(right), 1)
        return io.NodeOutput(outcome, f"{spelled} is {'true' if outcome else 'false'}")


def _as_number(value, name: str) -> float:
    """``value`` as a float.

    Args:
        value: Whatever arrived on the socket.
        name: The socket's name, for the message.

    Returns:
        The value as a float.

    Raises:
        ValueError: The value is not a number.
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(
            f"Logic Compare Numbers cannot compare {name}, which holds "
            f"{type(value).__name__}. Wire a number into it."
        ) from None


def _spell(value: float) -> str:
    """``value`` without a trailing ``.0``, so a whole number reads as one."""
    return str(int(value)) if float(value).is_integer() else str(value)
