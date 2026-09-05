"""Work out a written expression over a set of numbers."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import NUMBER
from ...modules.number.expression import ON_ERROR, ExpressionError, as_int, evaluate

logger = log.get_logger("nodes.number")

#: The names a formula may use, one per numeric input.
SLOTS = tuple("abcdefghijklmnopqrstuvwx")


class NumberExpression(io.ComfyNode):
    """Evaluate an arithmetic expression over its numeric inputs."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNumberExpression",
            display_name="Number Expression",
            search_aliases=[
                "WASNumberExpression",
                "Number Expression",
                "math",
                "formula",
                "calculator",
                "clamp",
                "round",
                "floor",
                "ceil",
                "abs",
                "min",
                "max",
                "sqrt",
                "lerp",
            ],
            category="WAS Suite/Number/Operations",
            description=(
                "Work out a whole formula over up to 24 numbers in one node, such as "
                "`(a * b) / 2 + c`, `clamp(a, 0, 1)` or `round(a / b, 2)`. The functions are "
                "min, max, abs, round, floor, ceil, sqrt, clamp(v, lo, hi), lerp(a, b, t), "
                "sign, log, log2, log10, exp, sin, cos, tan, atan2, hypot, degrees and "
                "radians, with pi, e and tau as constants. Comparisons and `and`, `or` work "
                "too, so `a if a > b else b` picks the larger and the boolean output carries "
                "the answer. Only arithmetic is read: a name, an attribute or a call that is "
                "not on the list is refused by name before anything runs. The box takes "
                "several lines, joined into one, and `#` starts a comment."
            ),
            inputs=[
                io.String.Input(
                    "expression",
                    default="a + b",
                    multiline=True,
                    placeholder="Eg: (a * b) / 2 + c",
                    tooltip=(
                        "The formula, over `a` to `x`. Eg: `(a * b) / 2 + c`. Functions: "
                        "min max abs round floor ceil sqrt clamp lerp sign log log2 log10 exp "
                        "sin cos tan atan2 hypot degrees radians, plus pi, e and tau. `a > b` "
                        "comes out as 1 or 0; `#` starts a comment."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "a",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `a` stands for. Type it here or wire one in. Unconnected "
                        "slots use the widget, and a slot the expression never names is "
                        "ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "b",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `b` stands for. Type it here or wire one in. `a / b` with "
                        "b at 0 stops the run unless on_error is set to zero."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "c",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `c` stands for. Type it here or wire one in. Handy as the "
                        "offset in `(a * b) / 2 + c`."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "d",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `d` stands for. Type it here or wire one in. The fourth "
                        "value, free for a limit such as `clamp(a, c, d)`."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "e",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `e` stands for. Type it here or wire one in. An "
                        "unconnected slot uses its widget, and a slot the expression never "
                        "names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "f",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `f` stands for. Type it here or wire one in. An "
                        "unconnected slot uses its widget, and a slot the expression never "
                        "names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "g",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `g` stands for. Type it here or wire one in. An "
                        "unconnected slot uses its widget, and a slot the expression never "
                        "names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "h",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `h` stands for. Type it here or wire one in. An "
                        "unconnected slot uses its widget, and a slot the expression never "
                        "names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "i",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `i` stands for. Type it here or wire one in. An "
                        "unconnected slot uses its widget, and a slot the expression never "
                        "names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "j",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `j` stands for. Type it here or wire one in. An "
                        "unconnected slot uses its widget, and a slot the expression never "
                        "names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "k",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `k` stands for. Type it here or wire one in. An "
                        "unconnected slot uses its widget, and a slot the expression never "
                        "names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "l",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `l` stands for. Type it here or wire one in. An "
                        "unconnected slot uses its widget, and a slot the expression never "
                        "names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "m",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `m` stands for. Type it here or wire one in. An "
                        "unconnected slot uses its widget, and a slot the expression never "
                        "names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "n",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `n` stands for. Type it here or wire one in. An "
                        "unconnected slot uses its widget, and a slot the expression never "
                        "names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "o",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `o` stands for. Type it here or wire one in. An "
                        "unconnected slot uses its widget, and a slot the expression never "
                        "names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "p",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `p` stands for. Type it here or wire one in. An "
                        "unconnected slot uses its widget, and a slot the expression never "
                        "names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "q",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `q` stands for. Type it here or wire one in. An "
                        "unconnected slot uses its widget, and a slot the expression never "
                        "names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "r",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `r` stands for. Type it here or wire one in. An "
                        "unconnected slot uses its widget, and a slot the expression never "
                        "names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "s",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `s` stands for. Type it here or wire one in. An "
                        "unconnected slot uses its widget, and a slot the expression never "
                        "names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "t",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `t` stands for. Type it here or wire one in. An "
                        "unconnected slot uses its widget, and a slot the expression never "
                        "names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "u",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `u` stands for. Type it here or wire one in. An "
                        "unconnected slot uses its widget, and a slot the expression never "
                        "names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "v",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `v` stands for. Type it here or wire one in. An "
                        "unconnected slot uses its widget, and a slot the expression never "
                        "names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "w",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `w` stands for. Type it here or wire one in. An "
                        "unconnected slot uses its widget, and a slot the expression never "
                        "names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "x",
                        default=0.0,
                        min=-18446744073709551615,
                        max=18446744073709551615,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, io.Int],
                    optional=True,
                    tooltip=(
                        "The number `x` stands for. Type it here or wire one in. An "
                        "unconnected slot uses its widget, and a slot the expression never "
                        "names is ignored."
                    ),
                ),
                io.Int.Input(
                    "decimals",
                    default=6,
                    min=0,
                    max=15,
                    step=1,
                    optional=True,
                    tooltip=(
                        "Decimal places a fractional answer is rounded to. 6 = 0.333333, "
                        "2 = 0.33, 0 = whole, so 3.7 comes out 4.0. It also clears the trailing "
                        "0.0000000001 that decimal arithmetic leaves behind. A whole answer is "
                        "untouched."
                    ),
                ),
                io.Combo.Input(
                    "on_error",
                    options=list(ON_ERROR),
                    default="error",
                    optional=True,
                    tooltip=(
                        "What a refused or impossible expression does. `error` = stop the run "
                        "and name the cause, `zero` = log it and answer 0. Pick `zero` where a "
                        "division by zero is expected on some frames of a batch."
                    ),
                ),
            ],
            outputs=[
                NUMBER.Output(
                    display_name="number",
                    tooltip=(
                        "The answer on the NUMBER wire, whole where it came out whole. A "
                        "comparison answers 1 or 0."
                    ),
                ),
                io.Float.Output(
                    display_name="float",
                    tooltip="The same answer as a decimal, so 7 leaves here as 7.0.",
                ),
                io.Int.Output(
                    display_name="int",
                    tooltip=(
                        "The same answer with its fraction cut off rather than rounded, so 3.9 "
                        "leaves here as 3. Held to the range a whole-number socket carries."
                    ),
                ),
                io.Boolean.Output(
                    display_name="boolean",
                    tooltip=(
                        "false when the answer is 0, true for anything else. Wire it to a "
                        "switch to branch on `a > b`."
                    ),
                ),
                io.String.Output(
                    display_name="text",
                    tooltip=(
                        "The answer written out, as `4.5` or `7`. Feed it to a filename prefix "
                        "or a text join."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        expression="a + b",
        decimals=6,
        on_error="error",
        **extra,
    ) -> io.NodeOutput:
        """Work out the expression and answer it in five forms.

        Args:
            expression: The formula, over ``a`` to ``x``.
            decimals: Decimal places a fractional answer is rounded to.
            on_error: ``error`` to stop the run, ``zero`` to answer 0.
            extra: The numeric slots ``a`` to ``x``, whatever the expression names.

        Returns:
            The answer as a number, a float, an int, a boolean and text.

        Raises:
            ExpressionError: The expression was refused or could not be worked out, and
                on_error is ``error``.
        """
        try:
            value = evaluate(expression, {name: extra.get(name, 0.0) for name in SLOTS})
        except ExpressionError as refused:
            if on_error != "zero":
                raise
            logger.warning("Number Expression answered 0 instead: %s", refused)
            value = 0

        if isinstance(value, bool):
            value = int(value)
        elif isinstance(value, float):
            value = round(value, int(decimals))

        return io.NodeOutput(value, float(value), as_int(value), value != 0, str(value))
