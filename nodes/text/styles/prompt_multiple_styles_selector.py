"""Combine styles from the style library."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from .prompt_styles_selector import load_library, style_names

logger = log.get_logger("text.styles")

#: The option meaning "this slot is unused", listed after the library's own names.
NONE = "None"


def style_options() -> list[str]:
    """The library's style names, with the unused option once, at the end.

    Returns:
        Every value a slot may hold, ``None`` last.
    """
    return [*(name for name in style_names() if name != NONE), NONE]



#: Input ids in socket order.
SLOTS = tuple(f"style{index + 1}" for index in range(24))


class PromptMultipleStylesSelector(io.ComfyNode):
    """Concatenate the prompts of the chosen styles."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        offered = style_options()
        return io.Schema(
            node_id="Prompt Multiple Styles Selector",
            display_name="Prompt Multiple Styles Selector",
            search_aliases=["Prompt Multiple Styles Selector", "styles", "a1111 styles"],
            category="WAS Suite/Text/Styles",
            description=(
                "Concatenate the positive and negative prompts of up to four styles from the "
                "style library, separated by spaces. A slot left on None is skipped."
            ),
            inputs=[
                io.Combo.Input(
                    "style1",
                    options=list(offered),
                    default=NONE,
                    tooltip=(
                        "First style to combine, or None to skip it; its prompts come first in "
                        "both outputs. A style named here and missing from the library empties "
                        "both outputs rather than dropping part of the prompt silently."
                    ),
                ),
                io.Combo.Input(
                    "style2",
                    options=list(offered),
                    default=NONE,
                    tooltip="Second style, appended after style1. None skips it.",
                ),
                io.Combo.Input(
                    "style3",
                    options=list(offered),
                    default=NONE,
                    tooltip="Third style, appended after style2. None skips it.",
                ),
                io.Combo.Input(
                    "style4",
                    options=list(offered),
                    default=NONE,
                    tooltip="Fourth style, appended last. None skips it.",
                ),
                io.Combo.Input(
                    "style5",
                    options=list(offered),
                    default=NONE,
                    tooltip="Style 5, concatenated after the one before it. `None` is skipped.",
                ),
                io.Combo.Input(
                    "style6",
                    options=list(offered),
                    default=NONE,
                    tooltip="Style 6, concatenated after the one before it. `None` is skipped.",
                ),
                io.Combo.Input(
                    "style7",
                    options=list(offered),
                    default=NONE,
                    tooltip="Style 7, concatenated after the one before it. `None` is skipped.",
                ),
                io.Combo.Input(
                    "style8",
                    options=list(offered),
                    default=NONE,
                    tooltip="Style 8, concatenated after the one before it. `None` is skipped.",
                ),
                io.Combo.Input(
                    "style9",
                    options=list(offered),
                    default=NONE,
                    tooltip="Style 9, concatenated after the one before it. `None` is skipped.",
                ),
                io.Combo.Input(
                    "style10",
                    options=list(offered),
                    default=NONE,
                    tooltip="Style 10, concatenated after the one before it. `None` is skipped.",
                ),
                io.Combo.Input(
                    "style11",
                    options=list(offered),
                    default=NONE,
                    tooltip="Style 11, concatenated after the one before it. `None` is skipped.",
                ),
                io.Combo.Input(
                    "style12",
                    options=list(offered),
                    default=NONE,
                    tooltip="Style 12, concatenated after the one before it. `None` is skipped.",
                ),
                io.Combo.Input(
                    "style13",
                    options=list(offered),
                    default=NONE,
                    tooltip="Style 13, concatenated after the one before it. `None` is skipped.",
                ),
                io.Combo.Input(
                    "style14",
                    options=list(offered),
                    default=NONE,
                    tooltip="Style 14, concatenated after the one before it. `None` is skipped.",
                ),
                io.Combo.Input(
                    "style15",
                    options=list(offered),
                    default=NONE,
                    tooltip="Style 15, concatenated after the one before it. `None` is skipped.",
                ),
                io.Combo.Input(
                    "style16",
                    options=list(offered),
                    default=NONE,
                    tooltip="Style 16, concatenated after the one before it. `None` is skipped.",
                ),
                io.Combo.Input(
                    "style17",
                    options=list(offered),
                    default=NONE,
                    tooltip="Style 17, concatenated after the one before it. `None` is skipped.",
                ),
                io.Combo.Input(
                    "style18",
                    options=list(offered),
                    default=NONE,
                    tooltip="Style 18, concatenated after the one before it. `None` is skipped.",
                ),
                io.Combo.Input(
                    "style19",
                    options=list(offered),
                    default=NONE,
                    tooltip="Style 19, concatenated after the one before it. `None` is skipped.",
                ),
                io.Combo.Input(
                    "style20",
                    options=list(offered),
                    default=NONE,
                    tooltip="Style 20, concatenated after the one before it. `None` is skipped.",
                ),
                io.Combo.Input(
                    "style21",
                    options=list(offered),
                    default=NONE,
                    tooltip="Style 21, concatenated after the one before it. `None` is skipped.",
                ),
                io.Combo.Input(
                    "style22",
                    options=list(offered),
                    default=NONE,
                    tooltip="Style 22, concatenated after the one before it. `None` is skipped.",
                ),
                io.Combo.Input(
                    "style23",
                    options=list(offered),
                    default=NONE,
                    tooltip="Style 23, concatenated after the one before it. `None` is skipped.",
                ),
                io.Combo.Input(
                    "style24",
                    options=list(offered),
                    default=NONE,
                    tooltip="Style 24, concatenated after the one before it. `None` is skipped.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="positive_string",
                    tooltip=(
                        "The chosen positive prompts joined with spaces, for a positive CLIP "
                        "Text Encode. Empty when a named style is missing from the library."
                    ),
                ),
                io.String.Output(
                    display_name="negative_string",
                    tooltip=(
                        "The chosen negative prompts joined with spaces, for a negative CLIP "
                        "Text Encode. Empty when a named style is missing from the library."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, **extra) -> io.NodeOutput:
        library = load_library()
        if library is None:
            return io.NodeOutput("", "")

        # A slot left on None is dropped before anything is looked up.
        selected = [
            chosen
            for chosen in (extra.get(name, NONE) for name in SLOTS)
            if chosen != NONE
        ]
        known = library.get_prompts()
        for style in selected:
            if style not in known:
                logger.error("Style `%s` was not found in the style library.", style)
                return io.NodeOutput("", "")

        prompt = ""
        negative_prompt = ""
        for style in selected:
            prompt += known[style]["prompt"] + " "
            negative_prompt += known[style]["negative_prompt"] + " "

        return io.NodeOutput(prompt.strip(), negative_prompt.strip())
