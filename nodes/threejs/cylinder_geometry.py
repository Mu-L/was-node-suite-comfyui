"""A cylinder or cone for a Three.js mesh."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_GEOMETRY
from ...modules.threejs.spec import create_spec

REQUIRES = "threejs"


class ThreeCylinderGeometry(io.ComfyNode):
    """Build a cylinder geometry descriptor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeCylinderGeometry",
            display_name="Three Cylinder Geometry",
            search_aliases=[
                "WASThreeCylinderGeometry",
                "Three Cylinder Geometry",
                "cylinder",
                "cone",
                "tube",
            ],
            category="WAS Suite/Three",
            description=(
                "A cylinder standing on Y, and a cone or a truncated cone when the two radii "
                "differ. Setting the top radius to 0.0 gives a point, so a cone is this node "
                "with one number changed. Open ended leaves the flat caps off, which is what "
                "makes a pipe rather than a solid."
            ),
            inputs=[
                io.Float.Input(
                    "radius_top",
                    default=1.0,
                    min=0.0,
                    max=100000.0,
                    step=0.01,
                    tooltip="Radius at the top. 0.0 closes it to a point, which makes a cone.",
                ),
                io.Float.Input(
                    "radius_bottom",
                    default=1.0,
                    min=0.0,
                    max=100000.0,
                    step=0.01,
                    tooltip="Radius at the base. Matching radius_top at 1.0 gives a straight cylinder.",
                ),
                io.Float.Input(
                    "height",
                    default=1.0,
                    min=0.0001,
                    max=100000.0,
                    step=0.01,
                    tooltip="Height along Y in scene units. 1.0 is as tall as a unit cube.",
                ),
                io.Int.Input(
                    "radial_segments",
                    default=32,
                    min=3,
                    max=2048,
                    tooltip="Divisions around the axis. 32 looks round, 6 gives a hexagonal prism.",
                ),
                io.Int.Input(
                    "height_segments",
                    default=1,
                    min=1,
                    max=2048,
                    tooltip="Divisions along the height. 1 is enough unless a shader has to bend it.",
                ),
                io.Boolean.Input(
                    "open_ended",
                    default=False,
                    tooltip="`true` leaves the flat caps off, making a pipe; `false` closes both ends.",
                ),
            ],
            outputs=[
                THREE_GEOMETRY.Output(
                    display_name="geometry",
                    tooltip="The cylinder shape, for the geometry socket on Three Mesh.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, radius_top, radius_bottom, height, radial_segments, height_segments, open_ended
    ) -> io.NodeOutput:
        """Describe the cylinder."""
        return io.NodeOutput(
            create_spec(
                "geometry",
                "CylinderGeometry",
                params={
                    "args": [
                        float(radius_top),
                        float(radius_bottom),
                        float(height),
                        int(radial_segments),
                        int(height_segments),
                        bool(open_ended),
                    ]
                },
            )
        )
