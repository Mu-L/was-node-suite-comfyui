"""Select one style out of the style library."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import config, log
from ....modules.prompt import styles
from ....modules.util import sandbox

logger = log.get_logger("text.styles")

#: Offered when the library holds no styles, so the combo still has a value to store.
NO_STYLE = "None"


def style_names() -> list[str]:
    """The style names for a selector combo, read from the library at each call.

    Returns:
        Every name in the library, or ``["None"]`` while it is empty or absent.
    """
    try:
        library = load_library()
    except sandbox.PathNotAllowed as error:
        logger.error("%s", error)
        library = None
    names = list(library.get_prompts()) if library is not None else []
    return names or [NO_STYLE]


def library_path():
    """The style library file named by ``paths.styles``.

    Returns:
        The resolved override, or ``None`` when the key is unset and the library in the
        pack's own config directory is read instead.

    Raises:
        PathNotAllowed: ``paths.styles`` resolves outside every permitted read root.
    """
    configured = config.load_config()["paths"]["styles"]
    return sandbox.resolve_read(configured) if configured else None


def load_library():
    """The style library, or ``None`` when it cannot be read.

    Returns:
        A :class:`~modules.prompt.styles.PromptStyles`, or ``None`` when the library is
        missing or cannot be read.

    Raises:
        PathNotAllowed: ``paths.styles`` resolves outside every permitted read root.
    """
    path = library_path()
    try:
        return styles.open_styles(path)
    except Exception as error:
        logger.error("The style library could not be read (%s).", error)
        return None


class PromptStylesSelector(io.ComfyNode):
    """Emit the positive and negative prompt of one named style."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Prompt Styles Selector",
            display_name="Prompt Styles Selector",
            search_aliases=["Prompt Styles Selector", "style", "a1111 styles"],
            category="WAS Suite/Text/Styles",
            description=(
                "Emit the positive and negative prompt of a style from the style "
                "library. Styles come from styles.json in the config directory, or from "
                "the AUTOMATIC1111 styles.csv named by paths.styles, which has to sit in "
                "a folder this pack may read."
            ),
            inputs=[
                io.Combo.Input(
                    "style",
                    options=style_names(),
                    tooltip=(
                        "Which saved style to emit. Each one is a named pair of a positive "
                        "and a negative prompt. The menu is filled from the style library "
                        "and shows only 'None' while no style has been saved or imported."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="positive_string",
                    tooltip=(
                        "The style's positive prompt, for a positive CLIP Text Encode. Empty "
                        "when the style could not be found."
                    ),
                ),
                io.String.Output(
                    display_name="negative_string",
                    tooltip=(
                        "The style's negative prompt, for a negative CLIP Text Encode. Empty "
                        "when the style has none, or could not be found."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, style) -> io.NodeOutput:
        library = load_library()
        if library is None:
            return io.NodeOutput("", "")

        if style not in library.get_prompts():
            logger.error("Style `%s` was not found in the style library.", style)
            return io.NodeOutput("", "")

        prompt, negative_prompt = library.get_prompt(style)
        return io.NodeOutput(prompt or "", negative_prompt or "")
