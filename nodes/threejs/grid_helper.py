"""A reference grid on the ground plane."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_OBJECT
from ...modules.threejs.spec import create_spec

REQUIRES = "threejs"


class ThreeGridHelper(io.ComfyNode):
    """Build a grid helper descriptor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeGridHelper",
            display_name="Three Grid Helper",
            search_aliases=[
                "WASThreeGridHelper",
                "Three Grid Helper",
                "grid",
                "floor",
                "helper",
            ],
            category="WAS Suite/Three",
            description=(
                "A flat grid on the ground plane, giving scale and a horizon to judge the "
                "camera against. It takes no light and casts no shadow, so it never changes "
                "how the scene is lit. Wire it into a group beside the meshes; it is a guide "
                "rather than part of the model."
            ),
            inputs=[
                io.Float.Input(
                    "size",
                    default=10.0,
                    min=0.001,
                    max=1000000.0,
                    step=0.1,
                    tooltip="How far the grid reaches, edge to edge, in scene units. 10.0 suits a unit-sized subject.",
                ),
                io.Int.Input(
                    "divisions",
                    default=10,
                    min=1,
                    max=10000,
                    tooltip="How many cells across. 10 with a size of 10.0 makes each cell one unit.",
                ),
                io.String.Input(
                    "center_color",
                    default="#888888",
                    multiline=False,
                    tooltip="Colour of the two lines through the origin, as hexadecimal. #888888 is mid grey.",
                ),
                io.String.Input(
                    "grid_color",
                    default="#444444",
                    multiline=False,
                    tooltip="Colour of the other lines, as hexadecimal. #444444 sits back from the centre lines.",
                ),
            ],
            outputs=[
                THREE_OBJECT.Output(
                    display_name="grid",
                    tooltip="The grid, for Three Group or the root socket on Three Scene.",
                ),
            ],
        )

    @classmethod
    def execute(cls, size, divisions, center_color, grid_color) -> io.NodeOutput:
        """Describe the grid."""
        return io.NodeOutput(
            create_spec(
                "object",
                "GridHelper",
                params={
                    "size": float(size),
                    "divisions": int(divisions),
                    "centerColor": center_color,
                    "gridColor": grid_color,
                },
            )
        )
