"""Splitting each image into a grid of tiles, one tile per output."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image import dynamic
from ....modules.convert.tensors import image_planes, stack_images, tensor2pil
from ....modules.interface import size_report

REQUIRES = "extras"

#: Widest and tallest grid offered, so the most tiles is the product of the two. Sixteen wires is
#: already more than a canvas wants; a count past that is what `Image Tiled` answers as a batch.
MAX_COLUMNS = 4
MAX_ROWS = 4
MAX_TILES = MAX_COLUMNS * MAX_ROWS

#: One name per tile, in reading order. Declared in full: ``io.Autogrow`` is input only, the
#: schema always carries every slot, and the frontend draws only the ones in use.
TILE_NAMES = tuple(f"tile_{index + 1}" for index in range(MAX_TILES))


class ImageTileExtractGrid(io.ComfyNode):
    """Cut each image into a grid and send every tile to its own output."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageTileExtractGrid",
            display_name="Image Tile Extract (Grid)",
            search_aliases=[
                "WASImageTileExtractGrid", "Image Tile Extract Grid",
                "tile", "grid", "split", "crop", "dynamic tiles", "tile per output",
            ],
            category="WAS Suite/Image/Transform",
            description=(
                "Cut each picture into a grid of tiles and send each tile to its own output, "
                "reading left to right then top to bottom. Outputs appear as the grid grows. "
                "For many tiles on one wire rather than one each, use Image Tiled."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The pictures to cut up. Every frame of a batch is cut the same way, so "
                        "each output carries the same tile of every frame."
                    ),
                ),
                io.Int.Input(
                    "columns",
                    default=2,
                    min=1,
                    max=MAX_COLUMNS,
                    step=1,
                    tooltip=(
                        "Tiles across. Together with rows this decides how many outputs appear, "
                        f"up to {MAX_TILES}."
                    ),
                ),
                io.Int.Input(
                    "rows",
                    default=2,
                    min=1,
                    max=MAX_ROWS,
                    step=1,
                    tooltip=(
                        "Tiles down. A grid of 2 by 2 gives the four quadrants, which is what "
                        "Image Tile Extract (Quadrants) does with fixed outputs."
                    ),
                ),
                io.Int.Input(
                    "border_width",
                    default=0,
                    min=0,
                    max=100,
                    step=1,
                    tooltip=(
                        "Border in pixels drawn around each tile, in border_color. The tile is "
                        "shrunk to fit inside it, so the output stays the same size. 0 leaves "
                        "the tile at its own resolution with no resampling at all."
                    ),
                ),
                io.String.Input(
                    "border_color",
                    default="#FFFFFF",
                    tooltip=(
                        "Colour of the border, as a hex string such as #FFFFFF for white or "
                        "#000000 for black. The leading # is optional, and an unreadable value "
                        "falls back to white. Ignored when border_width is 0."
                    ),
                ),
            ],
            # Declared last, and written out one by one rather than built in a loop: outputs
            # are indexed by position, so a slot added after this block would shift every tile
            # index out from under a saved prompt.
            outputs=[
                io.Image.Output(
                    display_name="tile_1",
                    tooltip=(
                        "The first tile, counting left to right then top to bottom. Carries "
                        "nothing meaningful when the grid holds fewer tiles than this, so leave "
                        "it unwired."
                    ),
                ),
                io.Image.Output(
                    display_name="tile_2",
                    tooltip=(
                        "The second tile, counting left to right then top to bottom. Carries "
                        "nothing meaningful when the grid holds fewer tiles than this, so leave "
                        "it unwired."
                    ),
                ),
                io.Image.Output(
                    display_name="tile_3",
                    tooltip=(
                        "The third tile, counting left to right then top to bottom. Carries "
                        "nothing meaningful when the grid holds fewer tiles than this, so leave "
                        "it unwired."
                    ),
                ),
                io.Image.Output(
                    display_name="tile_4",
                    tooltip=(
                        "The fourth tile, counting left to right then top to bottom. Carries "
                        "nothing meaningful when the grid holds fewer tiles than this, so leave "
                        "it unwired."
                    ),
                ),
                io.Image.Output(
                    display_name="tile_5",
                    tooltip=(
                        "The fifth tile, counting left to right then top to bottom. Carries "
                        "nothing meaningful when the grid holds fewer tiles than this, so leave "
                        "it unwired."
                    ),
                ),
                io.Image.Output(
                    display_name="tile_6",
                    tooltip=(
                        "The sixth tile, counting left to right then top to bottom. Carries "
                        "nothing meaningful when the grid holds fewer tiles than this, so leave "
                        "it unwired."
                    ),
                ),
                io.Image.Output(
                    display_name="tile_7",
                    tooltip=(
                        "The seventh tile, counting left to right then top to bottom. Carries "
                        "nothing meaningful when the grid holds fewer tiles than this, so leave "
                        "it unwired."
                    ),
                ),
                io.Image.Output(
                    display_name="tile_8",
                    tooltip=(
                        "The eighth tile, counting left to right then top to bottom. Carries "
                        "nothing meaningful when the grid holds fewer tiles than this, so leave "
                        "it unwired."
                    ),
                ),
                io.Image.Output(
                    display_name="tile_9",
                    tooltip=(
                        "The ninth tile, counting left to right then top to bottom. Carries "
                        "nothing meaningful when the grid holds fewer tiles than this, so leave "
                        "it unwired."
                    ),
                ),
                io.Image.Output(
                    display_name="tile_10",
                    tooltip=(
                        "The tenth tile, counting left to right then top to bottom. Carries "
                        "nothing meaningful when the grid holds fewer tiles than this, so leave "
                        "it unwired."
                    ),
                ),
                io.Image.Output(
                    display_name="tile_11",
                    tooltip=(
                        "The eleventh tile, counting left to right then top to bottom. Carries "
                        "nothing meaningful when the grid holds fewer tiles than this, so leave "
                        "it unwired."
                    ),
                ),
                io.Image.Output(
                    display_name="tile_12",
                    tooltip=(
                        "The twelfth tile, counting left to right then top to bottom. Carries "
                        "nothing meaningful when the grid holds fewer tiles than this, so leave "
                        "it unwired."
                    ),
                ),
                io.Image.Output(
                    display_name="tile_13",
                    tooltip=(
                        "The thirteenth tile, counting left to right then top to bottom. Carries "
                        "nothing meaningful when the grid holds fewer tiles than this, so leave "
                        "it unwired."
                    ),
                ),
                io.Image.Output(
                    display_name="tile_14",
                    tooltip=(
                        "The fourteenth tile, counting left to right then top to bottom. Carries "
                        "nothing meaningful when the grid holds fewer tiles than this, so leave "
                        "it unwired."
                    ),
                ),
                io.Image.Output(
                    display_name="tile_15",
                    tooltip=(
                        "The fifteenth tile, counting left to right then top to bottom. Carries "
                        "nothing meaningful when the grid holds fewer tiles than this, so leave "
                        "it unwired."
                    ),
                ),
                io.Image.Output(
                    display_name="tile_16",
                    tooltip=(
                        "The sixteenth tile, counting left to right then top to bottom. Carries "
                        "nothing meaningful when the grid holds fewer tiles than this, so leave "
                        "it unwired."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, images, columns, rows, border_width, border_color) -> io.NodeOutput:
        """Cut every frame into the grid and answer one batch per tile.

        Raises:
            ValueError: The grid asks for more tiles than there are outputs.
        """
        folded = dynamic.fold(images)
        images = folded.images
        from PIL import Image

        from ....modules.image.tiles import parse_hex_color

        wanted = columns * rows
        if wanted > MAX_TILES:
            raise ValueError(
                f"Image Tile Extract (Grid) has {MAX_TILES} outputs, so a grid of {columns} by "
                f"{rows} asking for {wanted} tiles does not fit. Reduce the grid, or use Image "
                f"Tiled, which answers any number of tiles as one batch."
            )

        border_rgb = parse_hex_color(border_color)
        collected: list[list[Image.Image]] = [[] for _ in range(MAX_TILES)]

        for plane in image_planes(images):
            source = tensor2pil(plane)
            width, height = source.size
            # Floor division, then the last column and row take what is left, so the tiles cover
            # the whole picture rather than dropping a strip when a side does not divide evenly.
            tile_w = width // columns
            tile_h = height // rows

            for row in range(rows):
                for column in range(columns):
                    left = column * tile_w
                    top = row * tile_h
                    right = width if column == columns - 1 else left + tile_w
                    bottom = height if row == rows - 1 else top + tile_h
                    tile = source.crop((left, top, right, bottom))
                    if border_width > 0:
                        tile = cls.draw_border(tile, border_width, border_rgb)
                    collected[row * columns + column].append(tile)

        # Every declared output has to answer something, so the slots past the grid repeat the
        # first tile rather than being left empty, which the scheduler cannot index.
        filled = [stack_images(tiles) for tiles in collected[:wanted]]
        frame = size_report.frame_size(images)
        if frame is not None:
            across, down = frame[0] // columns, frame[1] // rows
            # The last column and row take what floor division left over, so the corner
            # tile is the one that can be wider or taller than the rest.
            size_report.publish(
                frame,
                (across, down),
                action="cut into tiles",
                facts={
                    "tiles": f"{wanted} of {columns} by {rows}",
                    "corner": f"{frame[0] - across * (columns - 1)}x"
                              f"{frame[1] - down * (rows - 1)}",
                },
            )
        answered = [dynamic.unfold(tile, folded) for tile in filled]
        return io.NodeOutput(*(answered + [answered[0]] * (MAX_TILES - wanted)))

    @staticmethod
    def draw_border(tile, border_width: int, colour: tuple[int, int, int]):
        """Shrink a tile and set it inside a border, keeping its outer size.

        Args:
            tile: The tile to frame.
            border_width: Border thickness in pixels.
            colour: Border colour as an RGB triple.

        Returns:
            A tile the same size as the one given, its picture inset by the border.
        """
        from PIL import Image

        width, height = tile.size
        inner = tile.resize(
            (max(width - border_width * 2, 1), max(height - border_width * 2, 1)),
            Image.LANCZOS,
        )
        framed = Image.new("RGB", (width, height), colour)
        framed.paste(inner, (border_width, border_width))
        return framed
