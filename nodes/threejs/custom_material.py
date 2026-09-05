"""A surface built by hand-written JavaScript."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_MATERIAL, THREE_TEXTURE
from ...modules.threejs.spec import compact_deps, create_spec

REQUIRES = "threejs"

DEFAULT_BODY = (
    'return new THREE.MeshStandardMaterial({color: "#ffffff", roughness: 0.35, metalness: 0.1});'
)

TEXTURE_TOOLTIP = (
    "A texture reachable in the body as `texture1` through `texture4`, by the slot it fills."
)


class ThreeCustomMaterial(io.ComfyNode):
    """Build a material from a JavaScript body."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeCustomMaterial",
            display_name="Three Custom Material",
            search_aliases=[
                "WASThreeCustomMaterial",
                "Three Custom Material",
                "custom material",
                "javascript",
                "toon",
            ],
            category="WAS Suite/Three",
            description=(
                "Reach any Three.js material class the pack has no node for, by returning one "
                "from a short JavaScript body. `THREE` is in scope, and any texture wired in "
                "arrives as `texture1` through `texture4`, so a toon, matcap, lambert or "
                "depth material is one line away. The code runs in your browser when the "
                "viewer loads, with the same reach as any frontend extension, so only run a "
                "workflow carrying custom JavaScript if you trust where it came from."
            ),
            inputs=[
                io.String.Input(
                    "javascript",
                    default=DEFAULT_BODY,
                    multiline=True,
                    tooltip=(
                        "A body returning a material, as "
                        "`return new THREE.MeshToonMaterial({map: texture1});`."
                    ),
                ),
                THREE_TEXTURE.Input("texture1", optional=True, tooltip=TEXTURE_TOOLTIP),
                THREE_TEXTURE.Input("texture2", optional=True, tooltip=TEXTURE_TOOLTIP),
                THREE_TEXTURE.Input("texture3", optional=True, tooltip=TEXTURE_TOOLTIP),
                THREE_TEXTURE.Input("texture4", optional=True, tooltip=TEXTURE_TOOLTIP),
            ],
            outputs=[
                THREE_MATERIAL.Output(
                    display_name="material",
                    tooltip="The surface the code returned, for the material socket on Three Mesh.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, javascript, texture1=None, texture2=None, texture3=None, texture4=None
    ) -> io.NodeOutput:
        """Carry the code and its textures to the browser."""
        return io.NodeOutput(
            create_spec(
                "material",
                "CustomMaterial",
                params={"javascript": javascript},
                deps=compact_deps(
                    texture1=texture1,
                    texture2=texture2,
                    texture3=texture3,
                    texture4=texture4,
                ),
            )
        )
