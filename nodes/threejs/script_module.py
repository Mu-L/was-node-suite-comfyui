"""Named Three.js resources built once by hand-written JavaScript."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_MODULE
from ...modules.threejs.spec import create_spec

REQUIRES = "threejs"

DEFAULT_BODY = """return {
    gold: new THREE.MeshStandardMaterial({color: "#d8b24a", metalness: 0.8, roughness: 0.2}),
    ring: new THREE.TorusGeometry(1, 0.2, 24, 96)
};"""


class ThreeScriptModule(io.ComfyNode):
    """Build named resources from a JavaScript body."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeScriptModule",
            display_name="Three Script Module",
            search_aliases=[
                "WASThreeScriptModule",
                "Three Script Module",
                "script module",
                "javascript",
                "exports",
            ],
            category="WAS Suite/Three",
            description=(
                "Build several named resources in one place and hand them out through Three "
                "Import Material, Three Import Geometry and Three Import Object. The body "
                "returns an object whose keys are the names, and it runs once per viewer load, "
                "so a palette of materials shared across many meshes is built once rather than "
                "per node. The code runs in your browser, with the same reach as any frontend "
                "extension, so only run a workflow carrying custom JavaScript if you trust "
                "where it came from."
            ),
            inputs=[
                io.String.Input(
                    "module_name",
                    default="custom",
                    multiline=False,
                    tooltip="Label for this module, such as 'palette'. Two modules may not share a name.",
                ),
                io.String.Input(
                    "javascript",
                    default=DEFAULT_BODY,
                    multiline=True,
                    tooltip=(
                        "A body returning named resources, as "
                        "`return {gold: new THREE.MeshStandardMaterial({})};`."
                    ),
                ),
            ],
            outputs=[
                THREE_MODULE.Output(
                    display_name="module",
                    tooltip="The named resources, for any of the Three Import nodes.",
                ),
            ],
        )

    @classmethod
    def execute(cls, module_name, javascript) -> io.NodeOutput:
        """Carry the code to the browser."""
        return io.NodeOutput(
            create_spec(
                "module",
                "ScriptModule",
                params={"name": module_name, "javascript": javascript},
            )
        )
