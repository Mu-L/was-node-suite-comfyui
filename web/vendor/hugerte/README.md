# HugeRTE, vendored

Third-party code. Nothing in this directory is written or maintained by WAS Node Suite, and
the files are kept byte for byte as upstream publishes them.

| | |
|---|---|
| Package | `hugerte` |
| Version | 1.0.12, built 2026-06-29 |
| Licence | MIT for HugeRTE itself, see `license.txt`; two of its files also bundle DOMPurify 3.4.11, which is dual licensed `MPL-2.0 OR Apache-2.0`. See [Licensing](#licensing) |
| Upstream | https://github.com/hugerte/hugerte |
| Home page | https://hugerte.org/ |
| Source | the npm package `hugerte@1.0.12` |

HugeRTE is a fork of TinyMCE 6, taken before TinyMCE 7 relicensed to GPLv2+ with a
commercial alternative. HugeRTE's own code is MIT, and it is self-hosted: no API key, no
account and no cloud call. It resolves its own theme, model, icon pack, plugins and skins by
URL at runtime, which is why it is served as real files rather than read as script text the
way Prism, KaTeX and Mermaid are under `web/viewer/`.

## Licensing

Two projects are licensed here, under three licences. MIT and Apache-2.0 are permissive;
MPL-2.0 is copyleft at file level, and the files it reaches are the two named below.
Redistributing the pack with this directory in it needs nothing beyond leaving the licence
texts where they are.

| Covers | Licence | Text |
|---|---|---|
| HugeRTE, every file in this directory, apart from the DOMPurify block inside the two named in the row below | MIT, copyright Ephox Corporation DBA Tiny Technologies, Inc., and the HugeRTE contributors | `license.txt` |
| DOMPurify 3.4.11, compiled into `hugerte.min.mjs` and `themes/silver/theme.min.mjs` | dual licensed `MPL-2.0 OR Apache-2.0`, copyright Dr.-Ing. Mario Heiderich, Cure53 and other contributors | `LICENSE-MPL-2.0.txt`, `LICENSE-APACHE-2.0.txt` |

HugeRTE compiles DOMPurify into those two bundles rather than loading it as a separate file,
and that copy is what `xss_sanitization` runs. It is the code and not only the credit: the
minified bodies carry DOMPurify's `ALLOWED_URI_REGEXP` and its Trusted Types `createPolicy`
calls. Both files open with the upstream notice, "This file bundles the code of DOMPurify,
which is dual-licensed under the Mozilla Public License v2.0 and the Apache License, Version
2.0", and the `@license` line inside each names version 3.4.11. Those two FILES are 920,653 of
the 1,353,824 bytes of HugeRTE kept here, about two thirds, so this is not a footnote. Every byte
count in this section is of the 19 upstream files rather than of the directory, so editing this
README cannot move one of them.

How much of those two files is DOMPurify's, rather than HugeRTE's, is a separate number, and it
is the one a licence audit wants. Each copy runs from that file's `@license DOMPurify 3.4.11`
comment to the end of the sanitiser the comment introduces:

| File | Whole file | The DOMPurify copy in it | Share |
|---|---|---|---|
| `hugerte.min.mjs` | 472,604 bytes | 58,147 bytes, at offsets 184,972 to 243,119 | 12.3% |
| `themes/silver/theme.min.mjs` | 448,049 bytes | 58,144 bytes, at offsets 103,327 to 161,471 | 13.0% |

So 116,291 bytes, about 114 KiB, are under MPL-2.0 or Apache-2.0: 12.6 per cent of those two
files and 8.6 per cent of the 1,353,824 bytes kept here. Everything else in this directory is
MIT. HugeRTE's build keeps DOMPurify's own comments, 125 comment blocks in each copy, which is
why a copy here is more than twice the size of the copy of DOMPurify 3.2.4 inside
`web/viewer/views/markdown_scripts/mermaid.min.txt`: that one keeps its licence comment and
nothing else, and is 22,720 bytes.

The choice is DOMPurify's own, not one made here: `dompurify@3.4.11` declares
`"license": "(MPL-2.0 OR Apache-2.0)"` and the upstream repository ships one text for each.
Satisfying either one is enough; you do not have to satisfy both.

Where the two texts came from, so they can be checked. Neither is modified, and neither
carries a header added here, because a licence reproduced inexactly is not the licence:

| File | Taken from | SHA-256 |
|---|---|---|
| `LICENSE-APACHE-2.0.txt` | `https://raw.githubusercontent.com/cure53/DOMPurify/3.4.11/LICENSE`, which is byte for byte the text published at https://www.apache.org/licenses/LICENSE-2.0.txt | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `LICENSE-MPL-2.0.txt` | `https://raw.githubusercontent.com/cure53/DOMPurify/3.4.11/LICENSE-MPL` | `fab3dd6bdab226f1c08630b1dd917e11fcb4ec5e1e020e2c16f83a0a13863e85` |

The MPL text is taken from DOMPurify rather than from mozilla.org because the two differ in
two characters: what mozilla.org currently serves at `/media/MPL/2.0/index.txt` has dropped a
trailing space on one line and changed the scheme in the Exhibit A URL from `http` to
`https`. The copy that travels with the code being bundled is the one that matters, so that
is the copy kept.

## Serving

`pyproject.toml` sets `[tool.comfy] web = "web"` and `[project] name = "was-node-suite-comfyui"`.
ComfyUI's `nodes.load_custom_node` reads the first key, resolves it against the pack
directory and files the result under the project name in `EXTENSION_WEB_DIRS`; `server.py`
then mounts each entry with `web.static("/extensions/" + name, dir)`. So this directory is
reachable at:

```
/extensions/was-node-suite-comfyui/vendor/hugerte
```

That string is the editor's `base_url`. Every path below is relative to it.

## The `.mjs` extension

Every executable script here carries `.mjs` where upstream ships `.js`. The bytes are
untouched; only the file name differs.

ComfyUI's `/extensions` route globs `**/*.js` under each pack's web directory, and the
frontend passes every result to a dynamic `import()` at page load, treating it as a ComfyUI
extension module. Left as `.js`, the whole editor would be fetched and executed on every
page load by everyone, whether or not a rich text node is on the canvas. Worse, those
imports all start together, so the theme, model, icon and plugin files race the core: each
of them reads the global `hugerte` as its first statement, and whichever arrives first
throws `ReferenceError` into the console.

`.mjs` is a standard JavaScript extension that the glob does not match. Python's
`mimetypes`, which is what aiohttp's static handler answers from, maps it to
`text/javascript`, so both the module import of the core and the classic scripts the loader
adds are served a JavaScript content type.

## Layout

The internal layout is upstream's, because the loader builds URLs from `base_url`:

```
hugerte.min.mjs                                  core
themes/silver/theme.min.mjs                      the only theme upstream ships
models/dom/model.min.mjs                         the only model upstream ships
icons/default/icons.min.mjs                      the default icon pack
plugins/<name>/plugin.min.mjs                    one directory per plugin
skins/ui/oxide-dark/skin.min.css                 toolbar, menus and dialogs
skins/ui/oxide-dark/content.min.css              that skin's rules inside the iframe
skins/ui/oxide-dark/skin.shadowdom.min.css       requested only inside a shadow root
skins/content/default/content.min.css            the document's own base stylesheet
license.txt                                      upstream MIT licence
LICENSE-APACHE-2.0.txt                           one of DOMPurify's two, added here
LICENSE-MPL-2.0.txt                              the other, added here
```

## Configuring the loader

The core and the icon pack are loaded directly; everything else is named by option, because
the loader would otherwise build a `.js` URL that does not exist here.

| What | How it is found |
|---|---|
| Core | a classic `<script src="<base>/hugerte.min.mjs">`, which assigns `window.hugerte`. The bundle is an IIFE compiled for sloppy mode and is not an ES module, so it is not imported |
| Icons | a classic script from `<base>/icons/default/icons.min.mjs` before `init`, which registers the `default` pack and stops the loader fetching it. Its first statement reads the global `hugerte`, so it is loaded only after the core has run |
| Theme | `theme_url: "<base>/themes/silver/theme.min.mjs"` |
| Model | `model_url: "<base>/models/dom/model.min.mjs"` |
| Plugins | `external_plugins: {"<name>": "<base>/plugins/<name>/plugin.min.mjs"}`, which loads each URL and appends its name to `plugins`. The pass that follows skips any name already loaded, so nothing is fetched twice |
| UI skin | `base_url` plus `skin: "oxide-dark"`, resolved to `<base>/skins/ui/oxide-dark/`. The stylesheet names under it are fixed at `.min.css` and do not read `suffix` |
| Content skin | `content_css: "default"` plus `suffix: ".min"`, resolved to `<base>/skins/content/default/content.min.css` |

`base_url` and `suffix` are set both ways, through `window.hugeRTEPreInit` before the core
script runs and again in the `init` options. `EditorManager.setup` reads the first while the
bundle is still running, and the `Editor` constructor reads the second. Left to itself the
manager works them out from the `src` of a script whose file name matches `hugerte.min.js`,
which is a name nothing here carries.

## Where the editor puts its own elements

| Element | Where it lands |
|---|---|
| The editor box | inside the element handed to `target`, replacing it. That element is a DOM widget on a node, so it carries the graph's zoom transform |
| Menus, dropdowns and the toolbar overflow | inside the editor box, so they scale with the node and are clipped by anything above them that clips |
| Dialogs | a sink appended to `document.body`, classed `tox tox-silver-sink tox-hugerte-aux`, outside the node and outside its transform, so they open at page size |
| Skin stylesheets | `<link>` elements in the page's own `<head>` |

## Defaults that rewrite content

HugeRTE filters and rewrites what is set into it, before anything this pack does.

| Option | Default | Effect |
|---|---|---|
| `valid_elements` | unset | unset means the `html5` schema is the allowlist, not that filtering is off. Anything outside it is dropped on the way in and on the way out |
| `verify_html` | `true` | enforces that allowlist. Setting it to `false` widens `valid_elements` to `*[*]`, so nothing is dropped for being unknown to the schema and no attribute for not belonging to its element. It does not widen the schema's lists of which children a parent may hold, which is one of the rewrites below that no option reaches. It also does not reach the parser's own `validate` flag, which `DomParser` is handed as a hardcoded `true` |
| `xss_sanitization` | `true` | runs the bundled DOMPurify over the content. Independent of `verify_html` and has to be turned off separately |
| `convert_urls` | `true` | rewrites `href` and `src` between absolute and relative forms |
| `entity_encoding` | `'named'` | rewrites a character that has a name in its own table of about 250 into that named entity on the way out. A character with no name in the table is written as itself, so this is not every non-ASCII character. `'raw'`, which the pack sets, is the least it will do and still encodes five characters: see the rewrites below |
| `indent` | `true` | reflows the whitespace between block elements |
| `convert_unsafe_embeds` | `true` | rewrites `<object>` and `<embed>` into `<iframe>`, `<img>` or `<video>` |
| `sandbox_iframes` | `true` | adds a `sandbox` attribute to every `<iframe>` in the content |
| `allow_script_urls` | `false` | deletes `javascript:`, `vbscript:`, `mhtml:` and any non-image `data:` from `src`, `href`, `data`, `background`, `action`, `formaction`, `poster` and `xlink:href`. None of that belongs to `xss_sanitization`: it runs with that off, and this is the switch in front of it. It does **not** reach the same filtering inside a `style` attribute, which is one of the rewrites below that no option reaches |
| `inline_styles` | `true` | rewrites `<font>` into a `<span>` carrying the same colour, family and size, and `<strike>` into `<s>`. Both filters are installed together behind this one option, and it is the only switch that reaches the second |
| `convert_fonts_to_spans` | `true` | the first half of that pair. Marked deprecated, along with `inline_styles`, so both are set for the intent to survive whichever the version keeps |
| `remove_trailing_brs` | `true` | drops the last `<br>` of a block, then removes the block that leaves empty or pads it with a non-breaking space |
| `allow_conditional_comments` | `false` | drops the conditional sections Word writes |
| `preserve_cdata` | `false` | drops CDATA sections |
| `allow_unsafe_link_target` | `false` | adds `rel="noopener"` to a link with `target="_blank"` |
| `allow_html_in_named_anchor` | `false` | takes the content of an `<a>` that carries a name or id and no `href` out in front of it, leaving the anchor empty |

`cleanup` is not an option in HugeRTE 1.0.12. It was removed in the TinyMCE 6 line this fork
was taken from, and setting it does nothing. `allow_html_data_urls` and `allow_svg_data_urls`
are real options, and setting them here would be dead config: every place that reads them sits
behind an `allow_script_urls` short-circuit, or inside the DOMPurify branch that
`xss_sanitization: false` never reaches, or inside the style parser that is handed neither
option and so can never see one set.

An editor that is meant to keep whatever HTML it is given, including HTML written by a
model, has to change every row above. `web/interface/rich_text.js` does, `valid_elements`
through `verify_html: false` and the other fourteen by name, and blocks script inside the
frame with `content_security_policy` instead, so nothing the document carries runs in
ComfyUI's own origin.

## Rewrites with no option behind them

Turning every row above off is not the same as a byte-exact round trip, and 1.0.12 does not
offer one. It cannot: the editor parses the document into a browser DOM and writes it back out
of that DOM, which is what a WYSIWYG editor is, and normalising the markup is inherent to that
rather than a filter with a switch behind it. These are the rewrites that reach every document
the editor is given, whatever the options say:

- **A top-level node the schema will not take as a body child is wrapped in a paragraph.**
  `forced_root_block` defaults to `'p'`, and `false` is a value the editor removed rather than
  one it still reads: the option processor rejects it and falls back to `'p'`. `Hello` reads
  back `<p>Hello</p>`, and so does a lone `<iframe>`, `<object>`, `<textarea>`, `<input>`,
  `<math>` or unknown element.
- **An element the schema will not allow where it stands is unwrapped, moved, or given the
  parent it needs.** The parser's `validate` flag is hardcoded on, and `Schema` fills its child
  lists from html5 even under `valid_elements: '*[*]'`. `valid_children` takes no wildcard, so
  the lists cannot be widened. `<td>a</td>` reads back `<p>a</p>`, `<li>a</li>` reads back
  `<ul><li>a</li></ul>`, and a `<table><tr>` gains a `<tbody>`.
- **Five characters are written as entities.** `Writer` calls `Entities.getEncodeFunc`, and
  `'raw'`, the loosest setting, resolves to `encodeRaw`, which maps through `baseEntities`:
  `<`, `>` and `&` in text, and those three plus `"` and a backtick in an attribute value. An
  apostrophe is not in either character class and is left alone. `AT&T` reads back `AT&amp;T`.
- **Every other entity is decoded to the character it names.** `&copy;` reads back `©` and
  `&nbsp;` reads back as a literal U+00A0, because the parse decodes and `raw` does not
  re-encode.
- **A `style` attribute is reparsed and reserialised, and some declarations are dropped.**
  `font-weight:700` reads back `bold`, a colour reads back lowercase, a `url()` is requoted with
  single quotes, and a declaration comes back as `name: value;`. A declaration holding
  `javascript:` or `vbscript:` or `data:image/svg+xml` inside a `url()`, or named `behavior`, or
  holding `expression()` or a CSS comment, is dropped, and a `style` attribute left empty by
  that is dropped with it. `DOMUtils` builds its `Styles` with `url_converter`,
  `url_converter_scope` and `force_hex_color` only, so the `allow_script_urls` and
  `allow_svg_data_urls` guards inside `Styles` read `undefined` and the filters behind them
  always run. This is the pass the serializer's `style` attribute filter calls.
- **An `on*` attribute on a descendant of `<svg>` is deleted.** `shouldKeepAttribute` returns
  `false` for an event handler whenever the namespace scope is not `'html'` and the tag is not
  `svg` itself, before any option is read. `namespaceElements` holds `svg` alone, so `<math>` is
  not treated this way, and the `on*` on the `<svg>` element itself is kept.
- **A boolean attribute is rewritten as `name="name"`, and a schema default is added.**
  `disabled=""` reads back `disabled="disabled"` and `<input name="a">` gains `type="text"`.
- **A document, head or metadata element is dropped.** The editor's content is a body, so
  `<!DOCTYPE>`, `<html>`, `<head>`, `<body>`, `<title>`, `<link>`, `<meta>` and `<style>` do not
  survive it. A whole page set into the editor reads back as its body's children.
- **Whitespace in text collapses.** A tab or a newline becomes a space, a run of spaces becomes
  one, whitespace between blocks is dropped, and the two ends of the document are trimmed.
  Whitespace inside `<pre>` is kept.
- **HTML tag and attribute names come back lowercase and every attribute value comes back
  double quoted**, and a void element loses an XHTML slash: `<BR/>` reads back `<br>`. SVG keeps
  its camel case, so `<linearGradient>` is unchanged.
- **An empty block gains a `<br>`, and a document that is nothing but one empty paragraph comes
  back empty.** `<p></p>` reads back `<p><br></p>`. `trimEmptyContents` matches the whole
  serialised document against one empty forced root block, so `<p>&nbsp;</p>` on its own reads
  back as the empty string. That trim runs outside `remove_trailing_brs`, but it is the only
  thing that does: with `remove_trailing_brs: false` a trailing `<br>`, and a pair of them, are
  kept wherever they were written.

`web/interface/rich_text.js` states this beside the options it sets, and
[`docs/CONFIG.md`](../../../docs/CONFIG.md) states it for users, one row per behaviour, because a
document whose exact bytes matter has to be edited in the node's own `html` widget rather than in
the editor.

## Defaults left alone, and why

The table above is the path a document takes when it is set into the editor and read back out.
Typing and pasting are two other paths, and their rewrites are left at HugeRTE's defaults
because they are what a word processor is expected to do rather than filtering of given markup.

| Option | Default | Effect |
|---|---|---|
| `text_patterns` | a list of Markdown-style patterns | `*italic*`, `**bold**`, and `#` through `######` followed by a space, applied as you type. `false` turns the lot off |
| `paste_remove_styles_if_webkit` | `true` | with `paste_webkit_styles` at its own default of `'none'`, strips every `style` attribute from a paste that did not come from inside the editor |
| `paste_merge_formats` | `true` | merges a pasted format element into an identical one it lands inside |
| `smart_paste` | `true` | a pasted URL becomes a link, or an image when it looks like one |
| `paste_data_images` | `true` | a pasted image is kept, as a data URL, which is what `automatic_uploads: false` in `rich_text.js` relies on |

None of these reaches a document set into the editor from the node's `html` widget, which is
why they are left as they are. Turning them off is a one-line change in `editorOptions` if a
future node wants the editor to be inert on those paths too.

## The trim

The published package is 217 files and 7.86 MiB. What is kept is 19 of those files and 1.29
MiB, plus this README and the two DOMPurify licence texts, which upstream does not ship
inside the package.

### Kept

| Path | Why |
|---|---|
| `hugerte.min.mjs` | the editor |
| `themes/silver/` | the only theme upstream ships, and the toolbar, menubar and dialogs live in it |
| `models/dom/` | the only model upstream ships, and the editor will not start without one |
| `icons/default/` | every toolbar button renders from this pack |
| `plugins/lists/` | bulleted and numbered lists |
| `plugins/advlist/` | list style choices on those two buttons |
| `plugins/link/` | insert, edit and remove hyperlinks |
| `plugins/autolink/` | a typed or pasted URL becomes a link |
| `plugins/image/` | insert an image and edit its source, alt text and size |
| `plugins/table/` | tables, the one document feature the core has no support for |
| `plugins/code/` | a dialog that edits the document's HTML directly |
| `plugins/searchreplace/` | find and replace inside a long document |
| `plugins/wordcount/` | word and character count in the status bar |
| `plugins/charmap/` | symbols that are not on the keyboard |
| `skins/ui/oxide-dark/` | one UI skin, the dark one, because ComfyUI's default palette is dark and the base shows through wherever a theme override is missing |
| `skins/content/default/` | one content skin, the plain one, because it sets no body background or text colour and so leaves the document's colours entirely to the theme |
| `license.txt` | the MIT licence, which must travel with the code |

The two DOMPurify texts are not in the package at all. Upstream keeps them in the DOMPurify
repository, and they are added here because the code they cover is compiled into
`hugerte.min.mjs` and `themes/silver/theme.min.mjs`. See [Licensing](#licensing).

Bold, italic, underline, strikethrough, headings and blocks, alignment, indent, text and
background colour, subscript and superscript, blockquote, horizontal rule, remove format,
undo and redo, cut, copy and paste all come from the core and need no plugin.

### Dropped

| What | Why |
|---|---|
| `hugerte.js`, `plugin.js`, `theme.js`, `model.js`, `icons.js` | unminified duplicates of what is kept |
| `index.js` in every kept directory | ESM entry points for a bundler, and this is loaded by URL |
| `hugerte.d.ts` | TypeScript definitions, 125 KB, and the pack is plain JavaScript |
| `CHANGELOG.md`, upstream `README.md` | history and install notes for the npm package |
| `package.json`, `bower.json`, `composer.json` | package manager manifests |
| 19 plugins | `accordion`, `anchor`, `autoresize`, `autosave`, `codesample`, `directionality`, `emoticons`, `help`, `importcss`, `insertdatetime`, `media`, `nonbreaking`, `pagebreak`, `preview`, `quickbars`, `save`, `template`, `visualblocks`, `visualchars`. None is on the toolbar. `codesample` alone is 48 KB and carries a second copy of Prism, which `web/viewer/` already vendors |
| `plugins/fullscreen` | cannot work here. The editor sits inside a DOM widget whose container ComfyUI positions with a CSS `transform`, and a transform is the containing block for `position: fixed` inside it, so the plugin's full screen box fills the widget it is already the size of. It also puts `overflow: hidden` and `height: 100%` on ComfyUI's own `<html>` and `<body>`, which breaks the page until it is toggled back. The editor's height is a setting instead, and the `code` dialog opens at page size |
| `skins/ui/oxide`, `skins/ui/hugerte-5`, `skins/ui/hugerte-5-dark` | three unused UI skins |
| `skins/content/dark`, `document`, `writer`, `hugerte-5`, `hugerte-5-dark` | five unused content skins |
| `content.inline.min.css` in the kept UI skin | requested only in inline mode, and the editor runs in an iframe |
| unminified `.css` and the `.js` CSS wrappers in every kept skin | the wrappers exist for bundlers that inline the stylesheet |

## Reproducing the trim

```sh
npm pack hugerte@1.0.12
tar -xf hugerte-1.0.12.tgz
```

Copy the paths in the "Kept" table out of the unpacked `package/` directory, renaming each
`.js` to `.mjs`, and keep the directory names. Adding a toolbar button that needs a plugin
means copying one more `plugins/<name>/plugin.min.js` in under the same rule and adding a
row above.

Then fetch DOMPurify's two licence texts, which the package does not carry, and check them
against the hashes in [Licensing](#licensing):

```sh
curl -o LICENSE-APACHE-2.0.txt https://raw.githubusercontent.com/cure53/DOMPurify/3.4.11/LICENSE
curl -o LICENSE-MPL-2.0.txt https://raw.githubusercontent.com/cure53/DOMPurify/3.4.11/LICENSE-MPL
sha256sum LICENSE-APACHE-2.0.txt LICENSE-MPL-2.0.txt
```

Upgrading HugeRTE means reading the `@license` line in the new `hugerte.min.mjs` first. If
the DOMPurify version inside it has moved, re-fetch both texts at that tag and update the
version in this README.
