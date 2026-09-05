"""List the graph's groups with a mute or bypass switch on each."""

from __future__ import annotations

from comfy_api.latest import io


class FastGroups(io.ComfyNode):
    """A panel of the graph's groups, each switching the nodes inside it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASFastGroups",
            display_name="Fast Groups",
            search_aliases=[
                'WASFastGroups',
                "Fast Groups",
                "group mute",
                "group bypass",
                "mute groups",
                "bypass groups",
                "group switcher",
            ],
            category="WAS Suite/Utilities",
            description=(
                "List every group in the graph, each with a switch that mutes or bypasses "
                "every node inside it. Nothing has to be selected first, and the switches "
                "are the graph's own mute and bypass states, so they survive a save, an "
                "undo and a copy. The node reads nothing and answers nothing."
            ),
            inputs=[],
            outputs=[],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        """Answer nothing.

        Returns:
            An empty result. The switches act on the graph in the browser, so a run has
            nothing left to do.
        """
        return io.NodeOutput()
