"""Put what the graph made into an archive, and answer the archive that holds it."""

from __future__ import annotations

import io as stdio

from comfy_api.latest import io

from ...modules import log
from ...modules.archive import container, draft, save
from ...modules.compat.types import DOC, ZIP
from ...modules.convert.tensors import image_planes, mask_images, plane2pil
from ...modules.document import formats

logger = log.get_logger("nodes.archive")

#: Pillow's name for each picture format the widget offers, and the extension it gets.
IMAGE_FORMATS: dict[str, tuple[str, str]] = {
    "PNG": ("PNG", ".png"),
    "JPEG": ("JPEG", ".jpg"),
    "WEBP": ("WEBP", ".webp"),
}

#: How many digits a frame number is padded to when one add writes several pictures.
FRAME_PADDING = 4


class ZipAdd(io.ComfyNode):
    """Append pictures, text or a document to an archive, answering the new archive."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASZipAdd",
            display_name="ZIP Add",
            search_aliases=[
                'WASZipAdd',
                "ZIP Add",
                "zip add",
                "add to zip",
                "archive add",
                "zip images",
                "zip text",
            ],
            category="WAS Suite/Archive",
            description=(
                "Add a picture, mask, string or DOC to an archive held on a wire. A "
                "picture or mask is encoded in image_format, one file per frame; a "
                "string or DOC is written in text_format. Answers a new archive "
                "holding what it was given plus what was added."
            ),
            inputs=[
                io.MultiType.Input(
                    "content",
                    [io.Image, io.Mask, DOC, io.String],
                    tooltip=(
                        "What to add. An IMAGE or MASK batch writes one file per frame, "
                        "numbered; a STRING or a DOC writes one file. This socket takes a "
                        "connection."
                    ),
                ),
                io.String.Input(
                    "name",
                    default="file",
                    multiline=False,
                    tooltip=(
                        "Entry name, without an extension, which the format supplies. A '/' "
                        "makes a folder inside the archive. A batch numbers each frame after "
                        "this, so 'renders/frame' gives 'renders/frame_0001.png'. A name "
                        "already in the archive is numbered apart rather than replaced. Eg: "
                        "renders/frame"
                    ),
                ),
                io.Combo.Input(
                    "image_format",
                    options=list(IMAGE_FORMATS),
                    tooltip=(
                        "How a picture or a mask is encoded. 'PNG' is lossless and keeps "
                        "alpha, 'JPEG' is smaller and has neither, 'WEBP' is smaller than "
                        "PNG and keeps alpha."
                    ),
                ),
                io.Combo.Input(
                    "text_format",
                    options=list(formats.FORMATS),
                    default=formats.MARKUP,
                    tooltip=(
                        "How a string or a DOC is written, and the extension it gets. "
                        "'.wasdoc' loses nothing from a DOC, '.html' writes a whole page, "
                        "and the rest write the text alone."
                    ),
                ),
                io.Combo.Input(
                    "compression",
                    options=list(save.COMPRESSIONS),
                    tooltip=(
                        "'deflate' shrinks entries that compress; 'store' writes them as "
                        "they are, which suits pictures that are compressed already."
                    ),
                ),
                ZIP.Input(
                    "zip",
                    optional=True,
                    tooltip=(
                        "The archive to add to, from Open ZIP or another ZIP Add. Left "
                        "unconnected, the archive starts empty."
                    ),
                ),
            ],
            outputs=[
                ZIP.Output(
                    display_name="zip",
                    tooltip=(
                        "The archive holding everything it held before plus what was added. "
                        "Send it to another ZIP Add, or to Save ZIP to write it."
                    ),
                ),
                io.Int.Output(
                    display_name="entry_count",
                    tooltip="How many files the answered archive holds.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        content=None,
        name="file",
        image_format="PNG",
        text_format=formats.MARKUP,
        compression="deflate",
        zip=None,
    ) -> io.NodeOutput:
        """Encode the content and answer the archive holding it.

        Raises:
            ValueError: ``content`` is None, or is of no kind this writes.
        """
        source = zip if container.is_archive(zip) else None
        additions = cls.encode(
            content, str(name).strip() or "file", image_format, text_format
        )
        built = draft.extended(source, additions, compression)
        logger.info(
            "ZIP Add wrote %d entry(s), and the archive now holds %d",
            len(additions),
            len(built.files),
        )
        return io.NodeOutput(built, len(built.files))

    @classmethod
    def encode(cls, content, name, image_format, text_format) -> list:
        """The entries one piece of content becomes.

        Args:
            content: An IMAGE or MASK tensor, a DOC, or a string.
            name: The entry name without an extension.
            image_format: A key of :data:`IMAGE_FORMATS`.
            text_format: One of :data:`modules.document.formats.FORMATS`.

        Returns:
            The :class:`~modules.archive.draft.Addition` entries, in frame order.

        Raises:
            ValueError: ``content`` is None, or is of no kind this writes.
        """
        if content is None:
            raise ValueError(
                "ZIP Add was given nothing to add. Connect an IMAGE, a MASK, a DOC or a "
                "STRING to 'content'."
            )
        pictures = cls.pictures(content)
        if pictures is not None:
            codec, extension = IMAGE_FORMATS.get(image_format, IMAGE_FORMATS["PNG"])
            return [
                draft.Addition(
                    cls.frame_name(name, index, len(pictures), extension),
                    cls.encoded(picture, codec),
                )
                for index, picture in enumerate(pictures)
            ]
        extension = formats.normalized(text_format)
        return [
            draft.Addition(
                cls.frame_name(name, 0, 1, extension),
                formats.payload(content, extension),
            )
        ]

    @classmethod
    def pictures(cls, content):
        """The frames a picture or mask tensor holds, or None when it is neither."""
        shape = getattr(content, "shape", None)
        if shape is None:
            return None
        if len(shape) == 4:
            return [plane2pil(plane) for plane in image_planes(content)]
        if len(shape) == 3:
            return list(mask_images(content))
        return None

    @classmethod
    def frame_name(cls, name: str, index: int, total: int, extension: str) -> str:
        """One entry's name, numbered only where an add writes more than one."""
        stem = name[: -len(extension)] if name.lower().endswith(extension) else name
        if total <= 1:
            return stem + extension
        return "%s_%0*d%s" % (stem, FRAME_PADDING, index + 1, extension)

    @classmethod
    def encoded(cls, picture, codec: str) -> bytes:
        """One picture's bytes in the named codec."""
        if codec == "JPEG" and picture.mode not in ("RGB", "L"):
            picture = picture.convert("RGB")
        buffer = stdio.BytesIO()
        picture.save(buffer, format=codec)
        return buffer.getvalue()
