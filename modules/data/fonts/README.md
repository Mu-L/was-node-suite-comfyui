# Bundled fonts

Typefaces this pack ships and may redistribute, vendored whole with the licence and
authorship files their upstreams distribute them with. Nothing here is downloaded at
runtime and nothing is fetched from a font service; the files in this directory are what
`Image Draw Text` renders with.

The `font` menu on **Image Draw Text** offers these faces by name. A name that is not one of
them, and not a file found in the user font directory, selects nothing.

## What is here, and under what terms

| Family | Files | Licence | Upstream |
|---|---|---|---|
| DejaVu 2.37 | `dejavu/DejaVuSans.ttf`, `DejaVuSans-Bold.ttf`, `DejaVuSansMono.ttf`, `DejaVuSerif.ttf` | Bitstream Vera + Arev, see `dejavu/LICENSE` | <https://github.com/dejavu-fonts/dejavu-fonts/releases/tag/version_2_37> |
| Liberation 2.1.5 | `liberation/LiberationSans-Regular.ttf`, `LiberationSans-Bold.ttf`, `LiberationSerif-Regular.ttf`, `LiberationMono-Regular.ttf` | SIL Open Font License 1.1, see `liberation/LICENSE` | <https://github.com/liberationfonts/liberation-fonts/releases/tag/2.1.5> |

Both licences permit redistribution, including inside a larger work and commercially, on the
condition that the licence and copyright notice travel with the font files. `LICENSE` and
`AUTHORS` in each directory are those notices. A font moved out of here without them is being
redistributed outside its terms.

Neither licence extends to this pack's own code, and neither is affected by it. The OFL
places one restriction: a Liberation file may not be redistributed under the reserved names,
Liberation, Arimo, Tinos, Cousine, if it has been modified. These files are unmodified, byte
for byte as the upstream release ships them.

## The two families

**DejaVu** is the coverage set. Around 6000 glyphs each, spanning Latin, Greek, Cyrillic,
currency, arrows, and the punctuation prose actually uses, em dashes and curly quotes
among them. It is what a caption falls back to when the text is not known in advance.

**Liberation** is metric-compatible with Arial, Times New Roman and Courier New: same
advance widths, same line breaks, so a layout designed against those fonts composes
identically without shipping fonts that cannot be redistributed. It is the set to reach for
when matching an existing design rather than choosing a new one.

## Adding your own fonts

Not here, put them in `<ComfyUI user dir>/was-node-suite/fonts`, beside `config.yaml`.
This directory belongs to the pack and an update replaces it, taking anything added by
hand with it.

Subfolders are read as well. A face there joins the `font` menu under its own name, and a
name that clashes with another file is qualified with the folder it sits in. A face already
bundled here is not offered twice.

## Adding a font *to the pack*

A font added here is redistributed by this repository. Copy the file in beside its licence,
add it to `FONTS` in `modules/data/paths.py` so it gets a curated name and a fixed place in
the menu, and add a row above. Two things decide whether a font can be here at all:

- the licence permits redistribution inside this repository, and
- the licence text is vendored with it.

A font whose licence cannot be established does not qualify, however freely it is
available. `font.ttf` in the parent directory is Rheiborn Sans Clean, and carries no licence
string, no vendor and no designer in its own name table. `Image Color Palette` and the
channel-scope nodes draw their labels with it. It is not offered in the `font` menu.

Users who need a face that is not here point `font_path` on `Image Draw Text` at their own
file. A font outside ComfyUI's own directories needs its folder added to
`paths.allow_read` in `config.yaml`, which is the same rule every other path in the pack
follows.
