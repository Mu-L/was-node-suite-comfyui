"""Batch any number of latents into one, on a slot list that grows."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.interface import batch_report
from ...modules.util.slots import connected_in_order

#: Slot ids in the order the sockets are drawn and the latents are joined.
SLOT_NAMES = (
    "latent_a", "latent_b", "latent_c", "latent_d", "latent_e", "latent_f", "latent_g",
    "latent_h", "latent_i", "latent_j", "latent_k", "latent_l", "latent_m", "latent_n",
    "latent_o", "latent_p", "latent_q", "latent_r", "latent_s", "latent_t", "latent_u",
    "latent_v", "latent_w", "latent_x", "latent_y", "latent_z",
)

#: How many slots :data:`SLOT_NAMES` names.
MAX_SLOTS = 26


def check_latent_dimensions(tensors, names) -> None:
    """Reject latents whose sample grids cannot be joined end to end.

    Args:
        tensors: LATENT dicts, in slot order.
        names: Socket id of each entry, used to name the offenders.

    Raises:
        ValueError: Two entries differ in channel count or in grid size.
    """
    shapes: dict[tuple[int, ...], list[str]] = {}
    for name, tensor in zip(names, tensors):
        shape = tuple(int(axis) for axis in tensor["samples"].shape[1:])
        shapes.setdefault(shape, []).append(name)
    if len(shapes) < 2:
        return
    listed = ", ".join(
        f"{'/'.join(sorted(set(slots)))} is {shape[-1]}x{shape[-2]} with {shape[0]} channels"
        for shape, slots in shapes.items()
    )
    raise ValueError(
        f"Latent Batch joins its latents into one batch, so every connected latent must hold "
        f"the same number of channels on the same grid. These do not match: {listed}. Send the "
        f"odd one on separately, or bring it to the same size before batching."
    )


class LatentBatch(io.ComfyNode):
    """Join latents from a slot list that grows a socket as each one is filled."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Latent Batch",
            display_name="Latent Batch (Advanced)",
            search_aliases=[
                "Latent Batch", "batch latents", "combine latents", "dynamic batch",
            ],
            category="WAS Suite/Latent",
            description=(
                "Join any number of latents into one batch, so a single sampler run covers all "
                "of them. A new empty slot appears below the last one filled, up to 26. Every "
                "latent must hold the same number of channels on the same grid, and a slot "
                "holding a batch contributes all of its frames."
            ),
            inputs=[
                io.Latent.Input(
                    "latent_a",
                    optional=True,
                    tooltip=(
                        "First latent or batch. Every connected slot has to hold the same "
                        "number of channels on the same grid, and at least one slot must be "
                        "connected."
                    ),
                ),
                io.Latent.Input(
                    "latent_b",
                    optional=True,
                    tooltip="Latent 2, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Latent.Input(
                    "latent_c",
                    optional=True,
                    tooltip="Latent 3, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Latent.Input(
                    "latent_d",
                    optional=True,
                    tooltip="Latent 4, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Latent.Input(
                    "latent_e",
                    optional=True,
                    tooltip="Latent 5, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Latent.Input(
                    "latent_f",
                    optional=True,
                    tooltip="Latent 6, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Latent.Input(
                    "latent_g",
                    optional=True,
                    tooltip="Latent 7, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Latent.Input(
                    "latent_h",
                    optional=True,
                    tooltip="Latent 8, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Latent.Input(
                    "latent_i",
                    optional=True,
                    tooltip="Latent 9, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Latent.Input(
                    "latent_j",
                    optional=True,
                    tooltip=(
                        "Latent 10, joined on after the one before it. Unconnected is "
                        "skipped."
                    ),
                ),
                io.Latent.Input(
                    "latent_k",
                    optional=True,
                    tooltip=(
                        "Latent 11, joined on after the one before it. Unconnected is "
                        "skipped."
                    ),
                ),
                io.Latent.Input(
                    "latent_l",
                    optional=True,
                    tooltip=(
                        "Latent 12, joined on after the one before it. Unconnected is "
                        "skipped."
                    ),
                ),
                io.Latent.Input(
                    "latent_m",
                    optional=True,
                    tooltip=(
                        "Latent 13, joined on after the one before it. Unconnected is "
                        "skipped."
                    ),
                ),
                io.Latent.Input(
                    "latent_n",
                    optional=True,
                    tooltip=(
                        "Latent 14, joined on after the one before it. Unconnected is "
                        "skipped."
                    ),
                ),
                io.Latent.Input(
                    "latent_o",
                    optional=True,
                    tooltip=(
                        "Latent 15, joined on after the one before it. Unconnected is "
                        "skipped."
                    ),
                ),
                io.Latent.Input(
                    "latent_p",
                    optional=True,
                    tooltip=(
                        "Latent 16, joined on after the one before it. Unconnected is "
                        "skipped."
                    ),
                ),
                io.Latent.Input(
                    "latent_q",
                    optional=True,
                    tooltip=(
                        "Latent 17, joined on after the one before it. Unconnected is "
                        "skipped."
                    ),
                ),
                io.Latent.Input(
                    "latent_r",
                    optional=True,
                    tooltip=(
                        "Latent 18, joined on after the one before it. Unconnected is "
                        "skipped."
                    ),
                ),
                io.Latent.Input(
                    "latent_s",
                    optional=True,
                    tooltip=(
                        "Latent 19, joined on after the one before it. Unconnected is "
                        "skipped."
                    ),
                ),
                io.Latent.Input(
                    "latent_t",
                    optional=True,
                    tooltip=(
                        "Latent 20, joined on after the one before it. Unconnected is "
                        "skipped."
                    ),
                ),
                io.Latent.Input(
                    "latent_u",
                    optional=True,
                    tooltip=(
                        "Latent 21, joined on after the one before it. Unconnected is "
                        "skipped."
                    ),
                ),
                io.Latent.Input(
                    "latent_v",
                    optional=True,
                    tooltip=(
                        "Latent 22, joined on after the one before it. Unconnected is "
                        "skipped."
                    ),
                ),
                io.Latent.Input(
                    "latent_w",
                    optional=True,
                    tooltip=(
                        "Latent 23, joined on after the one before it. Unconnected is "
                        "skipped."
                    ),
                ),
                io.Latent.Input(
                    "latent_x",
                    optional=True,
                    tooltip=(
                        "Latent 24, joined on after the one before it. Unconnected is "
                        "skipped."
                    ),
                ),
                io.Latent.Input(
                    "latent_y",
                    optional=True,
                    tooltip=(
                        "Latent 25, joined on after the one before it. Unconnected is "
                        "skipped."
                    ),
                ),
                io.Latent.Input(
                    "latent_z",
                    optional=True,
                    tooltip="Latent 26, the last slot. Unconnected is skipped.",
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="latent",
                    tooltip=(
                        "One latent holding every connected input end to end, in slot order, "
                        "so a sampler runs them in a single pass."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip=(
                        "How many frames the batch holds, which is the total across the slots "
                        "rather than the number of slots."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, **latent) -> io.NodeOutput:
        """Join every connected slot into one batch.

        Raises:
            ValueError: No slot holds a latent, or two differ in channel count.
        """
        latent_names = connected_in_order(latent, SLOT_NAMES)
        batched_tensors = [latent[name] for name in latent_names]

        if not batched_tensors:
            raise ValueError(
                f"Latent Batch has no latents connected. Connect at least one of the "
                f"{MAX_SLOTS} slots."
            )

        size, mode = batch_report.describe_latents(batched_tensors[0]["samples"])
        try:
            check_latent_dimensions(batched_tensors, latent_names)
        except ValueError as refused:
            # Reported before it is raised, so the node itself says which slot disagreed.
            batch_report.publish(
                frames=sum(int(t["samples"].shape[0]) for t in batched_tensors),
                slots=len(batched_tensors),
                size=size,
                mode=mode,
                memory=sum(batch_report.memory_of(t["samples"]) for t in batched_tensors),
                refused=str(refused),
            )
            raise
        samples_out = {"samples": torch.cat([t["samples"] for t in batched_tensors], dim=0)}
        batch_index = []
        for tensor in batched_tensors:
            batch_index += tensor.get("batch_index", list(range(tensor["samples"].shape[0])))
        samples_out["batch_index"] = batch_index
        batch_report.publish(
            frames=int(samples_out["samples"].shape[0]),
            slots=len(latent_names),
            size=size,
            mode=mode,
            memory=batch_report.memory_of(samples_out["samples"]),
        )
        return io.NodeOutput(samples_out, int(samples_out["samples"].shape[0]))
