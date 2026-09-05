"""A ring for a Three.js mesh."""

from __future__ import annotations

import math

from comfy_api.latest import io

from ...modules.compat.types import THREE_GEOMETRY
from ...modules.threejs.spec import create_spec

REQUIRES = "threejs"


class ThreeTorusGeometry(io.ComfyNode):
    """Build a torus geometry descriptor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeTorusGeometry",
            display_name="Three Torus Geometry",
            search_aliases=[
                "WASThreeTorusGeometry",
                "Three Torus Geometry",
                "torus",
                "ring",
                "donut",
            ],
            category="WAS Suite/Three",
            description=(
                "A ring lying in the XY plane. Radius is the distance from the centre to the "
                "middle of the tube and tube is the tube's own thickness, so the outer edge "
                "sits at radius plus tube. A shorter arc leaves an open horseshoe rather than "
                "a closed ring."
            ),
            inputs=[
                io.Float.Input(
                    "radius",
                    default=1.0,
                    min=0.0001,
                    max=100000.0,
                    step=0.01,
                    tooltip="Centre to the middle of the tube, in scene units. 1.0 with a tube of 0.4 reads as a donut.",
                ),
                io.Float.Input(
                    "tube",
                    default=0.4,
                    min=0.0001,
                    max=100000.0,
                    step=0.01,
                    tooltip="Thickness of the tube itself. 0.4 is a fat donut, 0.05 a wire hoop.",
                ),
                io.Int.Input(
                    "radial_segments",
                    default=16,
                    min=3,
                    max=2048,
                    tooltip="Divisions around the tube's own cross section. 16 looks round, 3 gives a triangular tube.",
                ),
                io.Int.Input(
                    "tubular_segments",
                    default=100,
                    min=3,
                    max=8192,
                    tooltip="Divisions around the ring. 100 looks smooth, 6 gives a hexagonal ring.",
                ),
                io.Float.Input(
                    "arc",
                    default=360.0,
                    min=0.01,
                    max=360.0,
                    step=0.1,
                    tooltip="How far the ring sweeps, in degrees. 360.0 closes it, 180.0 gives a horseshoe.",
                ),
            ],
            outputs=[
                THREE_GEOMETRY.Output(
                    display_name="geometry",
                    tooltip="The ring shape, for the geometry socket on Three Mesh.",
                ),
            ],
        )

    @classmethod
    def execute(cls, radius, tube, radial_segments, tubular_segments, arc) -> io.NodeOutput:
        """Describe the ring."""
        return io.NodeOutput(
            create_spec(
                "geometry",
                "TorusGeometry",
                params={
                    "args": [
                        float(radius),
                        float(tube),
                        int(radial_segments),
                        int(tubular_segments),
                        math.radians(float(arc)),
                    ]
                },
            )
        )
