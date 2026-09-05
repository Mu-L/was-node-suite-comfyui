"""Route the first connected input onward, whatever type it is."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.logic.switch_index import SLOT_NAMES

#: Tooltip on every slot. One wording for all of them: the letter is already on the socket.
SLOT_TIP = (
    "One candidate branch. Any type, and the first connection fixes the type for the rest. "
    "Slots are tried in order, input_a first: with input_a muted or unwired, input_b "
    "answers. Only the branch that answers is evaluated."
)


class AnyFirstSwitch(io.ComfyNode):
    """Pass the first connected input on, whatever type it is."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        match = io.MatchType.Template("any_first_switch")
        return io.Schema(
            node_id="WASAnyFirstSwitch",
            display_name="Any Switch (First Connected)",
            search_aliases=[
                "WASAnyFirstSwitch",
                "Any Switch (First Connected)",
                "first connected",
                "fallback switch",
                "mute switch",
                "bypass switch",
                "route",
            ],
            category="WAS Suite/Logic/Switch",
            description=(
                "Pass the first connected input on, whatever type it is. Slots are tried in "
                "order, input_a to input_z, and the earliest one still connected is the one "
                "that leaves, so muting a branch falls through to the next without any "
                "rewiring. Only that branch is evaluated, so the work behind the rest is "
                "skipped."
            ),
            inputs=[
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
                    tooltip="The branch that answered, typed to whatever was connected.",
                ),
                io.Int.Output(
                    display_name="resolved_index",
                    tooltip=(
                        "Which slot answered, counting the slots on the node from 0: "
                        "input_a = 0, input_c = 2. With input_a and input_b muted, this "
                        "reads 2."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip=(
                        "How many slots are still connected. Muting the node feeding a "
                        "slot drops it by one: 3 wired with 1 muted reads 2."
                    ),
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
    def check_lazy_status(cls, **slots) -> list[str]:
        """Ask for the earliest connected slot until it holds a value.

        Args:
            slots: The lazy slots, as far as they have been evaluated.

        Returns:
            The slot still to evaluate, or an empty list once its value has arrived.
        """
        # A wired slot is present here whether or not it was evaluated, holding None until
        # it is, so the earliest one is asked for alone and the rest stay unevaluated. Only
        # the earliest slot is read: a later slot whose feeding node is already in the run's
        # cache arrives holding a value nobody asked for, and treating that as the answer
        # would route the later branch while the earliest one was never evaluated.
        names = cls.wired(slots)
        if not names or slots[names[0]] is not None:
            return []
        return [names[0]]

    @classmethod
    def execute(cls, **slots) -> io.NodeOutput:
        """Answer the earliest connected slot.

        Args:
            slots: The connected slots.

        Returns:
            The value found, the slot it came from, and how many slots are connected.

        Raises:
            ValueError: Nothing is connected, or the earliest slot carried no value.
        """
        names = cls.wired(slots)
        if not names:
            raise ValueError(
                "Any Switch (First Connected) has nothing connected. Connect at least one "
                "input, and check that the node feeding it is not muted, since muting a "
                "node disconnects the slot it fed"
            )
        # The earliest slot is the one check_lazy_status had evaluated, so it is the only
        # one whose None means the branch answered nothing rather than never having run.
        first = names[0]
        if slots[first] is None:
            raise ValueError(
                f"Any Switch (First Connected) read {first} and it answered nothing. Wire "
                f"{first} to something that carries a value, or mute the node feeding it so "
                f"the switch falls through to the next slot"
            )
        return io.NodeOutput(slots[first], SLOT_NAMES.index(first), len(names))
