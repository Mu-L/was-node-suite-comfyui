"""One shape drawn with one surface."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_GEOMETRY, THREE_MATERIAL, THREE_OBJECT
from ...modules.threejs.spec import create_spec, require_spec

REQUIRES = "threejs"


class ThreeMesh(io.ComfyNode):
    """Pair a geometry with a material."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeMesh",
            display_name="Three Mesh",
            search_aliases=["WASThreeMesh", "Three Mesh", "mesh", "object"],
            category="WAS Suite/Three",
            description=(
                "Draw one geometry with one material. This is the object that appears in the "
                "scene: wire it into Three Group, or straight into Three Scene. A geometry and "
                "a material can each feed several meshes, and the browser builds one copy of "
                "each, so a hundred meshes sharing a material cost one material."
            ),
            inputs=[
                THREE_GEOMETRY.Input(
                    "geometry",
                    tooltip="The shape to draw, from any of the Three geometry nodes.",
                ),
                THREE_MATERIAL.Input(
                    "material",
                    tooltip="The surface to draw it with, from any of the Three material nodes.",
                ),
                io.String.Input(
                    "name",
                    default="Mesh",
                    multiline=False,
                    tooltip=(
                        "Label carried into the scene graph, such as 'floor' or 'hero'. Custom "
                        "code finds an object by it."
                    ),
                ),
                io.Boolean.Input(
                    "cast_shadow",
                    default=True,
                    tooltip=(
                        "`true` throws a shadow, `false` does not. Needs shadows on in Three App and on "
                        "the light too."
                    ),
                ),
                io.Boolean.Input(
                    "receive_shadow",
                    default=True,
                    tooltip=(
                        "`true` lets shadows land on this mesh; `false` keeps it unshadowed, as for a "
                        "glowing sign or a skybox."
                    ),
                ),
            ],
            outputs=[
                THREE_OBJECT.Output(
                    display_name="object",
                    tooltip="The mesh, for Three Group, Three Transform Object or Three Scene.",
                ),
            ],
        )

    @classmethod
    def execute(cls, geometry, material, name, cast_shadow, receive_shadow) -> io.NodeOutput:
        """Describe the mesh.

        Raises:
            ValueError: An input is not a descriptor of the kind its socket takes.
        """
        require_spec(geometry, "geometry")
        require_spec(material, "material")
        return io.NodeOutput(
            create_spec(
                "object",
                "Mesh",
                params={
                    "name": name,
                    "castShadow": bool(cast_shadow),
                    "receiveShadow": bool(receive_shadow),
                },
                deps={"geometry": geometry, "material": material},
            )
        )
