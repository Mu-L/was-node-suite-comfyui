"""Report what kind of value is on a wire, as text a condition can test."""

from __future__ import annotations

from comfy_api.latest import io


class TypeOf(io.ComfyNode):
    """Name the type of whatever is connected."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASTypeOf",
            display_name="Type Of",
            search_aliases=[
                "WASTypeOf",
                "Type Of",
                "what type",
                "type name",
                "inspect",
                "kind",
            ],
            category="WAS Suite/Logic/Boolean",
            description=(
                "Name what is on a wire, so a graph can branch on it. Answers `IMAGE`, "
                "`MASK`, `LATENT`, `MODEL`, `STRING` and so on, with the batch size and "
                "shape beside it. Feed the name to Compare and the answer to a switch to "
                "handle each kind differently."
            ),
            inputs=[
                io.MatchType.Input(
                    "value",
                    template=io.MatchType.Template("type_of"),
                    tooltip="Anything. The wire is read, not changed.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="type_name",
                    tooltip=(
                        "The socket type in capitals: `IMAGE`, `MASK`, `LATENT`, `MODEL`, "
                        "`STRING`, `INT`. Compare it to branch on the kind."
                    ),
                ),
                io.String.Output(
                    display_name="python_type",
                    tooltip="The class behind it, as `Tensor`, `dict`, `str`.",
                ),
                io.String.Output(
                    display_name="shape",
                    tooltip=(
                        "Sizes of a tensor, as `4x512x512x3`, or the entry count of a list "
                        "or dictionary. Empty for a value with neither."
                    ),
                ),
                io.Int.Output(
                    display_name="batch_size",
                    tooltip=(
                        "Frames a batched value carries, from the first axis of an image, "
                        "mask or latent. 1 for a single value and 0 where there is no batch."
                    ),
                ),
                io.Boolean.Output(
                    display_name="is_empty",
                    tooltip=(
                        "true for nothing connected, empty text, an empty list or a tensor "
                        "with no elements."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, value=None) -> io.NodeOutput:
        """Describe the connected value.

        Args:
            value: Whatever is wired in.

        Returns:
            The socket type, the class behind it, its shape, its batch size and whether it
            is empty.
        """
        from ....modules.logic.describe import describe_value

        found = describe_value(value)
        return io.NodeOutput(
            found["type_name"], found["python_type"], found["shape"],
            found["batch_size"], found["is_empty"],
        )
