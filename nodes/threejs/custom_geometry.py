"""A shape built by hand-written JavaScript."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_GEOMETRY
from ...modules.threejs.spec import create_spec

REQUIRES = "threejs"

DEFAULT_BODY = "return new THREE.TorusKnotGeometry(1, 0.3, 128, 24);"


class ThreeCustomGeometry(io.ComfyNode):
    """Build a geometry from a JavaScript body."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeCustomGeometry",
            display_name="Three Custom Geometry",
            search_aliases=[
                "WASThreeCustomGeometry",
                "Three Custom Geometry",
                "custom geometry",
                "javascript",
                "torus knot",
            ],
            category="WAS Suite/Three",
            description=(
                "Reach any Three.js geometry class the pack has no node for, by returning one "
                "from a short JavaScript body. `THREE` is in scope, so a torus knot, a lathe, a "
                "tube or a hand-built BufferGeometry are all one line away. The code runs in "
                "your browser when the viewer loads, with the same reach as any frontend "
                "extension, so only run a workflow carrying custom JavaScript if you trust "
                "where it came from."
            ),
            inputs=[
                io.String.Input(
                    "javascript",
                    default=DEFAULT_BODY,
                    multiline=True,
                    tooltip=(
                        "A body returning a geometry, as "
                        "`return new THREE.LatheGeometry(points);`. `THREE` is in scope."
                    ),
                ),
            ],
            outputs=[
                THREE_GEOMETRY.Output(
                    display_name="geometry",
                    tooltip="The shape the code returned, for the geometry socket on Three Mesh.",
                ),
            ],
        )

    @classmethod
    def execute(cls, javascript) -> io.NodeOutput:
        """Carry the code to the browser."""
        return io.NodeOutput(
            create_spec("geometry", "CustomGeometry", params={"javascript": javascript})
        )
