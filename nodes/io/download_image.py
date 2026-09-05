"""Fetch an image over HTTP and hand it on as a picture, a mask and its colour profile."""

from __future__ import annotations

from urllib.parse import urlsplit

from comfy_api.latest import io

from ...modules import deps, log
from ...modules.compat.types import WAS_COLOUR_PROFILE
from ...modules.image import colour_profile
from .load_image import decode

REQUIRES = "network"

logger = log.get_logger("nodes.io")

#: Config key of the group this node is gated on. Nothing here runs without it.
FEATURE = "features.network"

#: What an address has to start with to be fetched.
SCHEMES = ("http://", "https://")


def fetch(url: str):
    """Fetch an image over HTTP.

    Args:
        url: An ``http`` or ``https`` address naming an image.

    Returns:
        The decoded PIL image.

    Raises:
        DependencyError: ``requests`` is not installed.
        ValueError: The address is not an ``http`` one, the request failed, or what came
            back does not decode as an image.
    """
    from io import BytesIO

    from PIL import Image

    address = (url or "").strip()
    if not address.lower().startswith(SCHEMES):
        raise ValueError(
            f"`{address}` is not a web address. url takes an http or https address naming an "
            f"image. Use Image Load to read a file that is already on disk"
        )

    requests = deps.require("requests", feature=FEATURE)
    try:
        answered = requests.get(address, timeout=30)
        answered.raise_for_status()
    except requests.exceptions.RequestException as error:
        raise ValueError(
            f"{address} could not be fetched: {error}. Check that the address opens in a "
            f"browser, and that this machine is allowed out to it"
        ) from error

    try:
        return Image.open(BytesIO(answered.content))
    except Exception as error:
        raise ValueError(
            f"{address} answered {len(answered.content)} byte(s) that do not decode as an "
            f"image. An address behind a sign-in page answers this way"
        ) from error


def named(url: str, kind: str) -> str:
    """What to call the picture one address answered with.

    Args:
        url: The address it was fetched from.
        kind: The format it decoded as, as PIL named it.

    Returns:
        The last part of the address's path, with an extension where it had none. An
        address whose path names nothing is called ``download``.
    """
    tail = urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1]
    if not tail:
        return f"download.{kind or 'png'}"
    return tail if "." in tail else f"{tail}.{kind or 'png'}"


class DownloadImage(io.ComfyNode):
    """Fetch an image over HTTP and answer it the way the file loader does."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASDownloadImage",
            display_name="Download Image",
            search_aliases=[
                "WASDownloadImage", "Download Image", "image from url", "fetch image",
                "http image", "web image", "url", "Image Load",
            ],
            category="WAS Suite/IO",
            description=(
                "Fetch an image from an http or https address and hand it on as a picture, a "
                "mask, its name and its colour profile, the same four things Image Load "
                "answers with. A file tagged with a colour profile is converted to sRGB as "
                "it is read, or kept in its own space. This node is in the network group, so "
                "it only appears with features.network on in config.yaml, and it is the only "
                "node in the pack that fetches a picture."
            ),
            inputs=[
                io.String.Input(
                    "url",
                    default="",
                    multiline=False,
                    tooltip=(
                        "The address to fetch, such as "
                        "'https://example.com/photo.jpg'. It is read on every run, since "
                        "nothing on the wire says whether it has changed."
                    ),
                ),
                io.Boolean.Input(
                    "RGBA",
                    default=False,
                    tooltip=(
                        "`off` discards any transparency and hands on a plain colour image, "
                        "which is what samplers and most nodes expect; `on` keeps "
                        "the transparency channel in the image itself. The mask output is "
                        "produced either way."
                    ),
                ),
                io.Boolean.Input(
                    "filename_text_extension",
                    default=True,
                    optional=True,
                    tooltip=(
                        "Whether the filename_text output keeps the extension. On = 'cat.png', "
                        "off = 'cat'. The name is the last part of the "
                        "address."
                    ),
                ),
                io.Combo.Input(
                    "colour_space",
                    options=colour_profile.spaces(),
                    default="sRGB",
                    optional=True,
                    tooltip=(
                        "Which colour space the picture comes out in. \"the file's own\" "
                        "leaves a tagged file exactly as it was written. 'sRGB' is what a "
                        "sampler, a filter and a LUT expect. The rest, such as 'Adobe RGB "
                        "(1998)' and 'Display P3', are for a photograph that goes back out "
                        "in its own space."
                    ),
                ),
                io.Combo.Input(
                    "icc_mode",
                    options=list(colour_profile.MODES),
                    optional=True,
                    tooltip=(
                        "What to do with the space above. 'convert' changes the numbers so "
                        "the colour stays put, which is what a photograph wants. 'assign' "
                        "leaves the numbers alone and says they were in that space all "
                        "along, which is how an untagged file that is really Display P3 is "
                        "put right. Ignored for \"the file's own\"."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip="The picture that was fetched, as a batch of one.",
                ),
                io.Mask.Output(
                    display_name="mask",
                    tooltip=(
                        "The image's transparency as a mask, with the transparent parts "
                        "white and the opaque parts black. An image with no transparency "
                        "gives an empty 64x64 mask."
                    ),
                ),
                io.String.Output(
                    display_name="filename_text",
                    tooltip=(
                        "The last part of the address, for reuse as a caption or a save "
                        "prefix."
                    ),
                ),
                WAS_COLOUR_PROFILE.Output(
                    display_name="profile",
                    tooltip=(
                        "The colour profile the file was tagged with, such as Adobe RGB "
                        "(1998). Wire it into Image Save to write the result back in that "
                        "space rather than in sRGB. Empty for a file carrying no profile."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls, url="", RGBA=False, filename_text_extension=True, colour_space="sRGB",
        icc_mode=colour_profile.CONVERT,
    ):
        """A value that differs every prompt, since an address may answer differently."""
        return float("NaN")

    @classmethod
    def execute(
        cls, url="", RGBA=False, filename_text_extension=True, colour_space="sRGB",
        icc_mode=colour_profile.CONVERT,
    ) -> io.NodeOutput:
        """Fetch the address and answer what came back.

        Raises:
            DependencyError: ``requests`` is not installed.
            ValueError: The address is not an ``http`` one, the request failed, or what came
                back does not decode as an image.
        """
        address = (url or "").strip()
        opened = fetch(address)
        kind = (getattr(opened, "format", "") or "").lower()
        logger.info("fetched %s, %s %s", address, kind or "an image", opened.size)
        return decode(
            opened, kind, address, RGBA, filename_text_extension, colour_space,
            name=named(address, kind), icc_mode=icc_mode,
        )
