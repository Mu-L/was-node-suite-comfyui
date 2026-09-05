"""Read a document's metadata onto typed sockets and show it on the node."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ...modules.compat.types import DICT, DOC, LIST
from ...modules.document import container, summary
from ...modules.log import get_logger

logger = get_logger("nodes.document")


class ViewDocMetadata(io.ComfyNode):
    """A document's metadata, one field per socket, and the whole reading as text."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASViewDOCMetadata",
            display_name="View DOC Metadata",
            search_aliases=[
                "WASViewDOCMetadata",
                "View DOC Metadata",
                "document metadata",
                "document info",
                "doc metadata",
                "copyright",
                "word count",
            ],
            category="WAS Suite/Document",
            description=(
                "Read what a document says about itself and put every field on its own "
                "socket: the title, description, author, copyright statement, language and "
                "keywords, when it was created and last changed, what wrote it, the "
                "author's own custom pairs, the word and character counts, and the files "
                "embedded in it. The whole reading is shown on the node as well, and "
                "emitted as one block of text. A field the document does not carry comes "
                "out empty rather than invented, so an older document with no language tag "
                "reports an empty language and not a guessed one. The two counts are worked "
                "out from the document's content every run, so a document whose stored "
                "counts are out of date is reported as its content actually reads."
            ),
            inputs=[
                DOC.Input(
                    "doc",
                    tooltip=(
                        "The document to read, from any node with a DOC output. Nothing is "
                        "opened from disk here: what is reported is what the document on "
                        "this wire carries, so a document loaded from a file reports what "
                        "that file holds and one just built reports what has been set on it "
                        "so far."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="title",
                    tooltip=(
                        "What the document is called. Empty when it has none, which is how "
                        "a document made from plain text starts out. This is the title an "
                        "exported file carries and the one a file manager shows in its "
                        "title column."
                    ),
                ),
                io.String.Output(
                    display_name="description",
                    tooltip=(
                        "The sentence or two saying what the document is. Empty when "
                        "nothing was written. Every document format a DOC can be exported "
                        "to has this field, so it is worth filling in before a save."
                    ),
                ),
                io.String.Output(
                    display_name="author",
                    tooltip=(
                        "Who wrote the document, as free text. Empty when the document "
                        "names nobody."
                    ),
                ),
                io.String.Output(
                    display_name="copyright",
                    tooltip=(
                        "The rights statement, such as '(c) 2026 A. Name, CC BY 4.0'. Free "
                        "text rather than a licence code, because a document may carry "
                        "either. Empty when the document makes no claim, which is worth "
                        "checking before publishing what a workflow produced."
                    ),
                ),
                io.String.Output(
                    display_name="language",
                    tooltip=(
                        "The language tag the document is written in, such as 'en' or "
                        "'pt-BR'. It decides hyphenation and spell checking in an exported "
                        "file and the voice a screen reader picks. Empty when the document "
                        "carries no tag, which is the case for every document written "
                        "before the field was filled in."
                    ),
                ),
                io.String.Output(
                    display_name="keywords",
                    tooltip=(
                        "The keywords joined with commas, which is how the document formats "
                        "and desktop search engines spell the field. Empty when the "
                        "document has none. Use the keywords_list output instead to reach "
                        "one keyword at a time."
                    ),
                ),
                LIST.Output(
                    display_name="keywords_list",
                    tooltip=(
                        "The same keywords as a list, in the order they were given, for "
                        "Text List Get, Text List Length and Text List Slice. A document "
                        "with no keywords gives an empty list, which those nodes report as "
                        "a length of zero rather than failing."
                    ),
                ),
                io.String.Output(
                    display_name="created",
                    tooltip=(
                        "When the document was first made, as UTC in the form "
                        "'2026-01-02T03:04:05Z', which sorts correctly as text. Empty when "
                        "the document carries no stamp: the time it was read is never "
                        "reported as the time it was written."
                    ),
                ),
                io.String.Output(
                    display_name="modified",
                    tooltip=(
                        "When the document's content last changed, in the same form as "
                        "created. Editing a document's text or its embedded files stamps "
                        "this; changing only its metadata does not, because this field "
                        "reports on the content. Empty when the document carries no stamp."
                    ),
                ),
                io.String.Output(
                    display_name="generator",
                    tooltip=(
                        "What produced the document, 'WAS Node Suite' for one this pack "
                        "wrote. Empty for a container built by hand or by another tool that "
                        "did not fill the field in."
                    ),
                ),
                DICT.Output(
                    display_name="custom",
                    tooltip=(
                        "The author's own pairs of text, in the order the document holds "
                        "them, for Text Dictionary Get, Text Dictionary Keys and Dictionary "
                        "to Console. This is where anything the standard fields have no "
                        "room for is carried, and all three export formats keep such pairs. "
                        "An empty dictionary when the document has none."
                    ),
                ),
                io.Int.Output(
                    display_name="word_count",
                    tooltip=(
                        "How many words the document's text holds, with the markup stripped "
                        "first, so bold text inside a word does not split it. Counted from "
                        "the content on this run rather than read from the document, so it "
                        "is right even where the file's own figure is out of date. Zero for "
                        "an empty document."
                    ),
                ),
                io.Int.Output(
                    display_name="character_count",
                    tooltip=(
                        "How many characters that same text holds: spaces between words "
                        "count, each line break counts as one, and the indentation between "
                        "tags counts for nothing. Counted from the content on this run. "
                        "Wire it into a condition node to catch a document that came out "
                        "empty."
                    ),
                ),
                io.Int.Output(
                    display_name="asset_count",
                    tooltip=(
                        "How many files are embedded in the document, such as the pictures "
                        "in it. Zero for a document that carries none, and an entry naming "
                        "a place outside the document is not counted, because it is refused "
                        "when the document is read."
                    ),
                ),
                LIST.Output(
                    display_name="assets",
                    tooltip=(
                        "The names of those files, sorted, each one relative to the "
                        "document's own assets folder and spelled with '/' whatever machine "
                        "the document was written on. An empty list when there are none."
                    ),
                ),
                io.Boolean.Output(
                    display_name="has_metadata",
                    tooltip=(
                        "True when the document says anything about itself: any of title, "
                        "description, author, copyright, language or keywords holds "
                        "something. The timestamps and the generator are not counted, "
                        "because this pack writes all three on every document it saves. "
                        "Wire it into a switch to fill the fields in before an export "
                        "rather than shipping a file that describes nothing."
                    ),
                ),
                io.String.Output(
                    display_name="summary",
                    tooltip=(
                        "The whole reading as one block of text, a field to a line, labelled "
                        "with the name of the socket beside it, with '(not set)' where a "
                        "field is empty. The same text is shown on the node. Send it to Text "
                        "to Console or Save Text File to keep a record of what a run "
                        "produced."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, doc) -> io.NodeOutput:
        document = container.require_document(doc, "doc")
        values = summary.fields(document)
        logger.debug(
            "read the metadata of %r: %d word(s), %d embedded file(s)",
            values.title or "untitled", values.word_count, values.asset_count,
        )
        return io.NodeOutput(*values, ui=ui.PreviewText(values.summary))
