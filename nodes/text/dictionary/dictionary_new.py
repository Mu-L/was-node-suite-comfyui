"""Build a dictionary from key and value pairs typed in or wired in."""

from __future__ import annotations

import ast
from collections.abc import Mapping

from comfy_api.latest import io

from ....modules.compat.types import DICT, LIST

#: How a typed value becomes a list rather than a string. ``LIST_LITERAL`` reads a value
#: written the way Python writes a list; ``PER_LINE`` treats every line as an item; ``OFF``
#: stores every value as the text it was typed as.
LIST_LITERAL = "list literal"
PER_LINE = "one per line"
OFF = "off"
LIST_MODES = [LIST_LITERAL, PER_LINE, OFF]

#: Openers a list literal may start with, so text that cannot be one is never parsed.
LITERAL_OPENERS = "[("

#: What the appended pairs say. They differ from each other only by number, and from the
#: first pair in that the first one explains the node.
MORE_KEY_HINT = "key name"
MORE_VALUE_HINT = "value"
MORE_KEY_TIP = "Key name, such as `subject` or `style`; STRING. Empty keys are skipped."
MORE_VALUE_TIP = (
    "Value for the key beside it; STRING, LIST or DICT literal, or LIST or DICT by connection."
)


#: Key and value pairs the node declares.
PAIRS = 24


class DictionaryNew(io.ComfyNode):
    """Assemble a dictionary from up to eight key and value pairs."""

    # A LIST or a DICT wired into a value is stored as it arrives whatever list_values says,
    # which is what lets one entry hold the several alternatives Text Find and Replace by
    # Dictionary draws from.

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Dictionary New",
            display_name="Text Dictionary New",
            search_aliases=[
                "Text Dictionary New", "new dictionary", "dict create", "wildcards",
            ],
            category="WAS Suite/Text/Dictionary",
            description=(
                "Build a DICT from up to 24 key/value pairs. A value can be a STRING, or a LIST "
                "of alternatives for Text Find and Replace by Dictionary to draw from. Empty "
                "keys are skipped."
            ),
            inputs=[
                io.String.Input(
                    "key_1",
                    default="",
                    multiline=False,
                    placeholder="Eg: animal",
                    tooltip=(
                        "Key name for value_1; STRING. Text Find and Replace by Dictionary "
                        "swaps __key__ in a prompt for the value. Eg: animal"
                    ),
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_1",
                        default="",
                        multiline=True,
                        placeholder="Eg: a tabby cat",
                    ),
                    [io.String, LIST, DICT],
                    tooltip=(
                        "Value for key_1; STRING, LIST or DICT literal, or LIST or DICT by "
                        "connection. A connection ignores list_values. Eg: a tabby cat"
                    ),
                ),
                io.String.Input(
                    "key_2", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_2", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.String.Input(
                    "key_3", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_3", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.String.Input(
                    "key_4", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_4", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.String.Input(
                    "key_5", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_5", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.String.Input(
                    "key_6", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_6", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.String.Input(
                    "key_7", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_7", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.String.Input(
                    "key_8", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_8", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.String.Input(
                    "key_9", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_9", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.String.Input(
                    "key_10", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_10", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.String.Input(
                    "key_11", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_11", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.String.Input(
                    "key_12", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_12", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.String.Input(
                    "key_13", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_13", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.String.Input(
                    "key_14", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_14", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.String.Input(
                    "key_15", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_15", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.String.Input(
                    "key_16", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_16", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.String.Input(
                    "key_17", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_17", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.String.Input(
                    "key_18", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_18", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.String.Input(
                    "key_19", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_19", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.String.Input(
                    "key_20", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_20", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.String.Input(
                    "key_21", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_21", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.String.Input(
                    "key_22", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_22", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.String.Input(
                    "key_23", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_23", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.String.Input(
                    "key_24", default="", multiline=False, optional=True,
                    placeholder=MORE_KEY_HINT, tooltip=MORE_KEY_TIP,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_24", default="", multiline=True, optional=True,
                        placeholder=MORE_VALUE_HINT,
                    ),
                    [io.String, LIST, DICT], optional=True, tooltip=MORE_VALUE_TIP,
                ),
                io.Combo.Input(
                    "list_values",
                    options=LIST_MODES,
                    default=LIST_LITERAL,
                    optional=True,
                    tooltip=(
                        "How a typed value becomes a LIST. `list literal`: reads "
                        "['a cat', 'a wolf'], anything else stays STRING. `one per line`: "
                        "each line is an item, one line stays STRING. `off`: always STRING. "
                        "Connections ignore this."
                    ),
                ),
            ],
            outputs=[
                DICT.Output(
                    tooltip=(
                        "The DICT, for Text Find and Replace by Dictionary, Text Dictionary "
                        "Get and the other DICT nodes. Duplicate keys keep the last value."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, list_values=LIST_LITERAL, **extra) -> io.NodeOutput:
        pairs = tuple(
            (extra.get(f"key_{slot}", ""), extra.get(f"value_{slot}", ""))
            for slot in range(1, PAIRS + 1)
        )
        dictionary: dict = {}
        for key, value in pairs:
            # A key is a name to look the entry up by, so an empty one names nothing and the
            # pair is dropped rather than stored under the empty string.
            if key is None or str(key) == "":
                continue
            dictionary[str(key)] = _value(value, list_values)
        return io.NodeOutput(dictionary)


def _value(raw, mode: str):
    """One entry's value, as a list where it holds alternatives and as writing otherwise.

    Args:
        raw: Whatever arrived on the value input: text from its box, or a list or mapping
            from a link.
        mode: One of :data:`LIST_MODES`.

    Returns:
        A list, a dict, or a string.
    """
    # A link carrying a list or a mapping already is what it is, and re-reading it as text
    # would flatten a structure somebody built on purpose.
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, (list, tuple)):
        return list(raw)

    text = "" if raw is None else str(raw)
    if mode == PER_LINE:
        lines = [line for line in (part.strip() for part in text.split("\n")) if line]
        # One line is one value rather than a list of one: a dictionary of single answers is
        # the common case, and a node drawing from it treats the two differently.
        return lines if len(lines) > 1 else text
    if mode == LIST_LITERAL:
        return _literal(text)
    return text


def _literal(text: str):
    """A list written the way Python writes one, or the text unchanged.

    Args:
        text: The value as typed.

    Returns:
        A list where the text is a list or tuple literal, otherwise the text itself.
    """
    stripped = text.strip()
    # Checked before parsing so ordinary writing is never handed to the parser, and so a
    # value that happens to be a bare number or a quoted word stays the writing it is.
    if stripped[:1] not in LITERAL_OPENERS:
        return text
    try:
        parsed = ast.literal_eval(stripped)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        # Not a literal after all, so it is writing that begins with a bracket.
        return text
    return list(parsed) if isinstance(parsed, (list, tuple)) else text
