"""Route one of any number of values onward, chosen by number."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import NUMBER
from ....modules.logic.switch_index import MAX_SLOTS, OUT_OF_RANGE, SLOT_NAMES, resolve

#: Tooltip on every slot. One wording for all of them: the letter is already on the socket.
SLOT_TIP = (
    "One candidate for the index. Any type. The first connection fixes the type; an "
    "unconnected slot is not counted, so index 0 is the first slot actually wired."
)


class AnyIndexSwitch(io.ComfyNode):
    """Select one of a list of inputs of any type by its number."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        match = io.MatchType.Template("any_index_switch")
        return io.Schema(
            node_id="WASAnyIndexSwitch",
            display_name="Any Index Switch",
            search_aliases=[
                "WASAnyIndexSwitch",
                "Any Index Switch",
                "index switch",
                "select by index",
                "multi switch",
                "route",
            ],
            category="WAS Suite/Logic/Switch",
            description=(
                "Pass one of any number of values on, chosen by a number rather than a "
                "boolean, whatever type they are. Wire a Number Counter or a loop's index "
                "in to step through the inputs one per run. Only the chosen input is "
                "evaluated, so the work behind the rest is skipped."
            ),
            inputs=[
                io.MultiType.Input(
                    io.Int.Input("index", default=0, min=-99999999, max=99999999, step=1),
                    [io.Int, NUMBER, io.Float],
                    tooltip=(
                        "Slot to pass on, from 0. Negative counts from the end: -1 = last. "
                        "A decimal is truncated: 2.7 = 2."
                    ),
                ),
                io.Combo.Input(
                    "out_of_range",
                    options=list(OUT_OF_RANGE),
                    default="wrap",
                    tooltip=(
                        "Index outside 0..count-1. With 3 slots and index 4: `wrap` = slot "
                        "1, `clamp` = slot 2, `error` stops the prompt."
                    ),
                ),
                io.MatchType.Input(
                    "input_a", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_b", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_c", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_d", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_e", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_f", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_g", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_h", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_i", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_j", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_k", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_l", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_m", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_n", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_o", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_p", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_q", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_r", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_s", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_t", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_u", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_v", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_w", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_x", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_y", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
                io.MatchType.Input(
                    "input_z", template=match, lazy=True, optional=True,
                    tooltip=SLOT_TIP,
                ),
            ],
            outputs=[
                io.MatchType.Output(
                    template=match,
                    display_name="output",
                    tooltip="The selected input, typed to whatever was connected.",
                ),
                io.Int.Output(
                    display_name="resolved_index",
                    tooltip="Slot actually read, from 0, after wrap or clamp.",
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="Connected slots. The index runs 0..count-1.",
                ),
            ],
        )

    @classmethod
    def wired(cls, slots: dict) -> list[str]:
        """The slot names the graph connected, in the order they are drawn.

        Args:
            slots: Every slot keyword the node was called with.

        Returns:
            The names present in ``slots``, ordered against the declaration.
        """
        return [name for name in SLOT_NAMES if name in slots]

    @classmethod
    def check_lazy_status(cls, index=0, out_of_range="wrap", **slots) -> list[str]:
        """Ask only for the slot the index is about to choose.

        Args:
            index: Which slot the run will select.
            out_of_range: What an index past either end does.
            slots: The lazy slots, as far as they have been evaluated.

        Returns:
            The slot still needed, or an empty list once it has arrived.
        """
        names = cls.wired(slots)
        if not names:
            return []
        try:
            position = resolve(index, len(names), out_of_range, "Any Index Switch")
        except ValueError:
            return []
        chosen = names[position]
        return [] if slots.get(chosen) is not None else [chosen]

    @classmethod
    def execute(cls, index=0, out_of_range="wrap", **slots) -> io.NodeOutput:
        """Answer the input the index chooses.

        Args:
            index: Which connected slot to pass on.
            out_of_range: What an index past either end does.
            slots: The connected slots.

        Returns:
            The chosen value, the slot it came from, and how many are connected.

        Raises:
            ValueError: Nothing is connected, or the index is refused.
        """
        # Every wired slot, whether or not it was evaluated: a lazy slot the index did not
        # choose arrives as None, and counting only the ones holding a value would make the
        # count fall to one and every index resolve to zero.
        names = cls.wired(slots)
        if not names:
            raise ValueError(
                "Any Index Switch has nothing connected. Connect at least one input."
            )
        position = resolve(index, len(names), out_of_range, "Any Index Switch")
        return io.NodeOutput(slots[names[position]], position, len(names))
