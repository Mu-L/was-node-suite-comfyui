"""Convert a hex colour string to hue, saturation and lightness."""

from __future__ import annotations

from comfy_api.latest import io


class HexToHSL(io.ComfyNode):
    """Split ``#RRGGBB`` or ``#RRGGBBAA`` into rounded HSL components."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Hex to HSL",
            display_name="Hex to HSL",
            search_aliases=["Hex to HSL", "colour", "color", "hsl"],
            category="WAS Suite/Utilities",
            description="Convert a hex colour string into its HSL components and CSS string.",
            inputs=[
                io.String.Input(
                    "hex_color",
                    default="#FF0000",
                    tooltip=(
                        "The colour to convert, written as six hex digits for red, green "
                        "and blue, '#FF0000' is pure red, or as eight with a trailing "
                        "pair for opacity. The leading '#' is optional."
                    ),
                ),
                io.Boolean.Input(
                    "include_alpha",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Whether to read the last two hex digits as opacity. Off, alpha is "
                        "reported as 1.0 and the string comes out as 'hsl(...)'; on, and "
                        "given an eight-digit colour, the string comes out as 'hsla(...)'."
                    ),
                ),
            ],
            outputs=[
                io.Int.Output(
                    display_name="hue",
                    tooltip=(
                        "Position on the colour wheel in degrees, 0 to 360: 0 is red, 120 "
                        "green, 240 blue. Grey and white have no hue and report 0."
                    ),
                ),
                io.Int.Output(
                    display_name="saturation",
                    tooltip=(
                        "How strong the colour is, as a percentage: 0 is grey, 100 is fully "
                        "saturated."
                    ),
                ),
                io.Int.Output(
                    display_name="lightness",
                    tooltip=(
                        "How light the colour is, as a percentage: 0 is black, 50 is the "
                        "pure hue, 100 is white."
                    ),
                ),
                io.Float.Output(
                    display_name="alpha",
                    tooltip=(
                        "Opacity from 0.0 to 1.0, rounded to two places. 1.0 unless "
                        "include_alpha is on and the colour carried eight hex digits."
                    ),
                ),
                io.String.Output(
                    display_name="hsl",
                    tooltip=(
                        "The same colour as a CSS string, e.g. 'hsl(0, 100%, 50%)', ready "
                        "for HSL to Hex or any node that takes a colour string."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, hex_color, include_alpha=False) -> io.NodeOutput:
        if hex_color.startswith("#"):
            hex_color = hex_color[1:]

        red = int(hex_color[0:2], 16) / 255.0
        green = int(hex_color[2:4], 16) / 255.0
        blue = int(hex_color[4:6], 16) / 255.0
        alpha = int(hex_color[6:8], 16) / 255.0 if include_alpha and len(hex_color) == 8 else 1.0

        max_val = max(red, green, blue)
        min_val = min(red, green, blue)
        delta = max_val - min_val
        luminance = (max_val + min_val) / 2.0

        if delta == 0:
            hue = 0
            saturation = 0
        else:
            saturation = delta / (1 - abs(2 * luminance - 1))
            if max_val == red:
                hue = ((green - blue) / delta) % 6
            elif max_val == green:
                hue = (blue - red) / delta + 2
            else:
                hue = (red - green) / delta + 4
            hue *= 60
            if hue < 0:
                hue += 360

        luminance = luminance * 100
        saturation = saturation * 100

        if include_alpha:
            hsl_string = (
                f"hsla({round(hue)}, {round(saturation)}%, {round(luminance)}%, "
                f"{round(alpha, 2)})"
            )
        else:
            hsl_string = f"hsl({round(hue)}, {round(saturation)}%, {round(luminance)}%)"

        return io.NodeOutput(
            round(hue), round(saturation), round(luminance), round(alpha, 2), hsl_string
        )
