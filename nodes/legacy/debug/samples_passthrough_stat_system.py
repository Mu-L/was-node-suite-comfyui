"""Log RAM, VRAM and disk usage while passing a latent through."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import deps
from ....modules.log import get_logger

REQUIRES = "debug"

logger = get_logger("nodes.debug")


class SamplesPassthroughStatSystem(io.ComfyNode):
    """Report system resource usage and return the latent unchanged."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Samples Passthrough (Stat System)",
            display_name="Samples Passthrough (Stat System)",
            search_aliases=["Samples Passthrough (Stat System)", "system stats", "vram", "ram"],
            category="WAS Suite/Debug",
            description=(
                "Deprecated: use ComfyUI's system stats endpoint instead, which reports the "
                "same RAM, VRAM and disk figures. Logs those figures to the console and "
                "passes the latent through unchanged."
            ),
            inputs=[
                io.Latent.Input(
                    "samples",
                    tooltip=(
                        "A latent to pass along. It is not read or altered; it only gives the "
                        "node somewhere to sit in the graph so the figures are logged at that "
                        "point in the run."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="samples",
                    tooltip="The same latent that came in, unchanged.",
                ),
            ],
            is_deprecated=True,
        )

    @classmethod
    def execute(cls, samples) -> io.NodeOutput:
        logger.info("\n%s", "\n".join(cls.system_stats()))
        return io.NodeOutput(samples)

    @classmethod
    def system_stats(cls) -> list[str]:
        """One line each for RAM, VRAM and the disk holding the filesystem root."""
        import torch

        psutil = deps.require("psutil", feature="legacy.debug")

        ram = psutil.virtual_memory()
        ram_used = ram.used / (1024**3)
        ram_total = ram.total / (1024**3)
        ram_stats = f"Used RAM: {ram_used:.2f} GB / Total RAM: {ram_total:.2f} GB"

        if torch.cuda.is_available():
            device = torch.device("cuda")
            vram_used = torch.cuda.memory_allocated(device) / (1024**3)
            vram_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            vram_stats = f"Used VRAM: {vram_used:.2f} GB / Total VRAM: {vram_total:.2f} GB"
        else:
            vram_stats = "Used VRAM: unavailable / Total VRAM: unavailable (no CUDA device)"

        hard_drive = psutil.disk_usage("/")
        used_space = hard_drive.used / (1024**3)
        total_space = hard_drive.total / (1024**3)
        hard_drive_stats = f"Used Space: {used_space:.2f} GB / Total Space: {total_space:.2f} GB"

        return [ram_stats, vram_stats, hard_drive_stats]
