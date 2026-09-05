"""Evening out exposure and colour drift across the frames of a sequence."""

from __future__ import annotations

from comfy_api.latest import io

#: Frames either side the reference may be averaged over. A wider window is what flattens a
#: slower drift, and it is the only control that does: two passes of a narrow window land where
#: one pass of a window twice as wide lands, for twice the work. The padding it adds costs two
#: quantile curves per frame, a few hundred kilobytes at this ceiling, so the bound is on what a
#: sequence can usefully be averaged across rather than on what fits in memory.
MAX_RADIUS = 250


class ImageTemporalEqualize(io.ComfyNode):
    """Remap each frame of a batch onto a temporally averaged version of its own distribution."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageTemporalEqualize",
            display_name="Image Temporal Equalize",
            search_aliases=[
                "WASImageTemporalEqualize", "Image Temporal Equalize",
                "deflicker",
                "flicker",
                "exposure drift",
                "temporal histogram",
                "stabilize exposure",
                "video brightness",
            ],
            category="WAS Suite/Image/Adjustment",
            description=(
                "Even out brightness and colour drift across the frames of a batch, for footage "
                "that flickers or slowly changes exposure. Each frame is remapped onto an "
                "average of the frames around it, so the sequence settles without any single "
                "frame being pushed one way in the shadows and the other way in the highlights."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The sequence to even out; IMAGE. The batch is read in order, so it "
                        "must be the frames of one shot rather than unrelated pictures."
                    ),
                ),
                io.Int.Input(
                    "temporal_radius",
                    default=4,
                    min=0,
                    max=MAX_RADIUS,
                    tooltip=(
                        "Frames either side that each frame is averaged against; INT. Larger "
                        "settles a slower drift and resists a real change in lighting; 0 leaves "
                        "the sequence alone."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How far each frame is moved towards its average; FLOAT, 0 to 1. Below "
                        "1 keeps part of the original variation, for footage where some change "
                        "is meant to be there."
                    ),
                ),
                io.Boolean.Input(
                    "per_channel",
                    default=True,
                    tooltip=(
                        "Correct each colour channel on its own curve; BOOLEAN. On, a colour "
                        "cast that drifts is followed as well as a brightness change. Off, one "
                        "curve from brightness is applied to every channel, which moves no "
                        "colour that was not already moving."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The evened out sequence; IMAGE, the same size and length as it went in.",
                ),
            ],
        )

    @classmethod
    def execute(cls, images, temporal_radius=4, strength=1.0, per_channel=True) -> io.NodeOutput:
        """Even out the sequence.

        Raises:
            ValueError: The input holds no frames.
        """
        from ....modules.image import deflicker, dynamic

        if getattr(images, "ndim", 0) != 4 or int(images.shape[0]) < 1:
            raise ValueError(
                "Image Temporal Equalize needs a batch of images to work across. Connect a "
                "sequence of frames, such as the output of a video loader."
            )
        folded = dynamic.fold(images)
        evened = deflicker.equalize(
            folded.images,
            radius=int(temporal_radius),
            strength=float(strength),
            per_channel=bool(per_channel),
        )
        return io.NodeOutput(dynamic.unfold(evened, folded))
