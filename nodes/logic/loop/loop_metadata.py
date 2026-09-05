"""Reading a loop's metadata dictionary back apart."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import DICT


class LoopMetadata(io.ComfyNode):
    """Put each field of a loop's metadata on its own socket."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLoopMetadata",
            display_name="Loop Metadata",
            search_aliases=[
                "WASLoopMetadata", "Loop Metadata",
                "loop info",
                "iteration count",
                "current iteration",
                "stopped reason",
                "accumulated count",
                "for loop metadata",
                "while loop metadata",
            ],
            category="WAS Suite/Logic/Loop",
            description=(
                "Read a loop's metadata output apart into separate values: which iteration is "
                "running, how many finished, how many frames were collected, and why the loop "
                "stopped. Works with both the For and While pairs, from either end."
            ),
            inputs=[
                DICT.Input(
                    "metadata",
                    tooltip=(
                        "The metadata output of a loop's Open or Close node; DICT. An Open node "
                        "describes the iteration about to run, a Close node the finished loop."
                    ),
                ),
            ],
            outputs=[
                io.Int.Output(
                    display_name="current_iteration",
                    tooltip="Which iteration this is; INT, counting from 1.",
                ),
                io.Int.Output(
                    display_name="index",
                    tooltip=(
                        "The loop's counter; INT. Counts from start_index on a For Loop and "
                        "from 0 on a While Loop, for reading a position in a list."
                    ),
                ),
                io.Int.Output(
                    display_name="iterations_completed",
                    tooltip="How many iterations have finished; INT.",
                ),
                io.Int.Output(
                    display_name="limit",
                    tooltip=(
                        "What the loop is counting towards; INT. The iteration count, the frame "
                        "target, or the safety limit, whichever the loop is set to."
                    ),
                ),
                io.Int.Output(
                    display_name="accumulated_count",
                    tooltip=(
                        "Frames collected so far; INT, read from the first slot holding frames. "
                        "0 while accumulate is off."
                    ),
                ),
                io.String.Output(
                    display_name="accumulated_as",
                    tooltip=(
                        "How the values left the loop; STRING. 'final' for the last value "
                        "alone, 'batch' for one joined batch, 'list' for every value."
                    ),
                ),
                io.String.Output(
                    display_name="mode",
                    tooltip=(
                        "What ends the loop; STRING. 'iterations', 'total_frames', or "
                        "'condition' for a While Loop."
                    ),
                ),
                io.String.Output(
                    display_name="stopped_reason",
                    tooltip=(
                        "Why the loop stopped; STRING. Never empty: it reads 'Still running' "
                        "with the iteration while the loop is going, and 'Not started' before "
                        "anything has run."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, metadata=None) -> io.NodeOutput:
        from ....modules.logic import loop_meta

        return io.NodeOutput(
            int(loop_meta.read(metadata, "current_iteration")),
            int(loop_meta.read(metadata, "index")),
            int(loop_meta.read(metadata, "iterations_completed")),
            int(loop_meta.read(metadata, "limit")),
            int(loop_meta.read(metadata, "accumulated_count")),
            str(loop_meta.read(metadata, "accumulated_as")),
            str(loop_meta.read(metadata, "mode")),
            str(loop_meta.read(metadata, "stopped_reason")),
        )
