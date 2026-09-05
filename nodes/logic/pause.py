"""Hold a run still until it is resumed from the graph."""

from __future__ import annotations

from comfy_api.latest import io


class Pause(io.ComfyNode):
    """Stop a run at this node and wait for Resume, passing its input on unchanged."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.MatchType.Template("pause_passthrough")
        return io.Schema(
            node_id="WASPause",
            display_name="Pause",
            search_aliases=[
                "WASPause",
                "Pause",
                "halt",
                "wait",
                "breakpoint",
                "step through",
            ],
            category="WAS Suite/Logic",
            is_output_node=True,
            description=(
                "Stop a run at this node and wait for Resume on the node itself. Everything "
                "above it has already run and stays cached, so change a widget while it waits "
                "and queue again: only the changed node and what depends on it run a second "
                "time. Whatever is connected passes through untouched."
            ),
            inputs=[
                io.MatchType.Input(
                    "value",
                    template=template,
                    optional=True,
                    tooltip=(
                        "Anything to hold and pass on: IMAGE, LATENT, MODEL, STRING. Leave "
                        "it unconnected to stop the run without carrying anything."
                    ),
                ),
                io.String.Input(
                    "message",
                    default="",
                    tooltip=(
                        "Text drawn beside Resume: `check the mask before sampling`. Empty "
                        "draws the node name alone."
                    ),
                ),
                io.Float.Input(
                    "timeout",
                    default=600.0,
                    min=0.0,
                    max=86400.0,
                    step=10.0,
                    tooltip=(
                        "Seconds to wait before carrying on by itself. 600 is 10 minutes, "
                        "0 waits with no limit. The queue holds still the whole time."
                    ),
                ),
            ],
            outputs=[
                io.MatchType.Output(
                    template=template,
                    display_name="value",
                    tooltip="What arrived, unchanged.",
                ),
                io.String.Output(
                    display_name="outcome",
                    tooltip="How the wait ended: resumed, timed out.",
                ),
                io.Boolean.Output(
                    display_name="resumed",
                    tooltip="true where Resume was pressed, false where the wait ran out.",
                ),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        """Never cached, so a queued run always stops here.

        Returns:
            A value that never matches the last one.
        """
        return float("NaN")

    @classmethod
    def execute(cls, value=None, message="", timeout=600.0) -> io.NodeOutput:
        """Hold the run, then pass the input on.

        Args:
            value: Anything to hold and pass on.
            message: Text drawn beside Resume.
            timeout: Seconds to wait before carrying on.

        Returns:
            What arrived, how the wait ended, and whether Resume was pressed.
        """
        from ...modules.interface.pause import wait_for_resume

        from ...modules.interface.pause import RESUMED

        outcome, _ = wait_for_resume(
            str(cls.hidden.unique_id), timeout=float(timeout), message=message
        )
        return io.NodeOutput(value, outcome, outcome == RESUMED)
