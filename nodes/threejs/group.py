"""Several objects gathered under one parent."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_OBJECT
from ...modules.threejs.spec import create_spec, require_spec

REQUIRES = "threejs"

#: How many object slots the node offers.
SLOTS = 8

SLOT_TOOLTIP = (
    "A mesh, light, helper or another group to gather. Slots may be filled in any order."
)


class ThreeGroup(io.ComfyNode):
    """Gather objects under one parent."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeGroup",
            display_name="Three Group",
            search_aliases=["WASThreeGroup", "Three Group", "group", "parent", "collect"],
            category="WAS Suite/Three",
            description=(
                "Gather up to eight objects under one parent, so a scene can hold more than the "
                "single object its root socket takes. Groups nest, so wiring a group into "
                "another slot gets past eight. Moving the group moves everything in it, which "
                "is how a set of meshes is posed as one."
            ),
            inputs=[
                io.String.Input(
                    "name",
                    default="Group",
                    multiline=False,
                    tooltip=(
                        "Label carried into the scene graph, such as 'set' or 'props'. Custom "
                        "code finds a group by it."
                    ),
                ),
                io.Boolean.Input(
                    "visible",
                    default=True,
                    tooltip="`true` draws the group and everything in it; `false` hides all of it at once.",
                ),
                THREE_OBJECT.Input("object_1", optional=True, tooltip=SLOT_TOOLTIP),
                THREE_OBJECT.Input("object_2", optional=True, tooltip=SLOT_TOOLTIP),
                THREE_OBJECT.Input("object_3", optional=True, tooltip=SLOT_TOOLTIP),
                THREE_OBJECT.Input("object_4", optional=True, tooltip=SLOT_TOOLTIP),
                THREE_OBJECT.Input("object_5", optional=True, tooltip=SLOT_TOOLTIP),
                THREE_OBJECT.Input("object_6", optional=True, tooltip=SLOT_TOOLTIP),
                THREE_OBJECT.Input("object_7", optional=True, tooltip=SLOT_TOOLTIP),
                THREE_OBJECT.Input("object_8", optional=True, tooltip=SLOT_TOOLTIP),
            ],
            outputs=[
                THREE_OBJECT.Output(
                    display_name="group",
                    tooltip="The parent holding everything wired in, for Three Scene or another group.",
                ),
            ],
        )

    @classmethod
    def execute(cls, name, visible, **slots) -> io.NodeOutput:
        """Gather whatever was wired in.

        Raises:
            ValueError: A filled slot is not an object descriptor.
        """
        children = []
        for index in range(1, SLOTS + 1):
            value = slots.get(f"object_{index}")
            if value is not None:
                children.append(require_spec(value, "object"))
        return io.NodeOutput(
            create_spec(
                "object",
                "Group",
                params={
                    "name": name,
                    "visible": bool(visible),
                    "position": [0.0, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0],
                },
                children=children,
            )
        )
