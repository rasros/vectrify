"""Prompt building blocks for SVG generation and editing."""

# What the local operators can and cannot reach, stated to the model so it
# spends its one call on the other half. Mutation nudges numbers, shifts
# colors and stroke widths, and moves elements; crossover grafts subtrees
# between candidates. None of that invents a shape that was never proposed,
# removes one that should not be there, or changes what an existing shape is.
STRUCTURE_FIRST = """\
Your output is a starting point: a local optimizer then spends thousands of \
steps on it, moving and resizing parts and tuning their coordinates and \
colors. What it cannot do is invent a shape you left out, remove a structure \
you invented, or change what a shape fundamentally is.

So spend your effort where only you can:
- Every distinct part of the target is present, and nothing extra is.
- A part can be tiny. A nostril, a pupil's highlight, a dot inside a shape: \
these are parts, not detail, and they are the ones most often dropped. Four \
runs of one drawing put the beak, eye and wing in every time and the nostril \
in none.
- Each part is the right kind of thing: an outline that closes is one closed \
path, not two strokes that nearly meet; a filled region is a fill, not a \
thick stroke.
- Counts are exact. Ten circles means ten, not "about ten".
- The arrangement and proportions read correctly at a glance.

Rough coordinates and approximate colors are fine — they get optimized away. \
Do not spend effort deriving exact values."""


def diff_format_instructions(
    lang: str,
    *,
    unit: str = "lines",
    subject: str | None = None,
) -> str:
    """Instructions telling the model to answer with search/replace blocks.

    *lang* names the language ("SVG"), *unit* what a block contains
    ("lines"/"fragment"), *subject* how to refer to the document being
    edited (defaults to "<lang> code").
    """
    subject = subject or f"{lang} code"
    return f"""\
Respond with two or three ALTERNATIVE attempts at improving the {subject},
separated by a line containing only ===ALTERNATIVE=== . Each attempt is one or
more search/replace blocks, is judged on its own, and only the best is kept --
so make them different ideas rather than one change split up.

Prefer blocks to a full rewrite: a block changes only what it names, where
re-authoring the whole {subject} retypes every part you were not trying to
change.

<<<SEARCH>>>
exact {lang} {unit} to replace (copy verbatim from the current {subject})
<<<REPLACE>>>
improved replacement {unit}
<<<END>>>

Rules:
- Always return at least one block. If nothing looks clearly wrong, take the \
part that matches the target least well and improve that; a reply with no \
block is a discarded call, not a verdict that the drawing is finished.
- The SEARCH text must match the current {subject} character for character, \
though how the whitespace inside it is written does not matter.
- Keep blocks small and focused; only change what needs to change.
- Multiple blocks are allowed.
- An attempt must carry every block its own change needs; sections are applied \
independently of each other.
- If you cannot copy the text to replace exactly, output the complete \
{subject} instead. That is worth more than a block that matches nothing: a \
reply with neither is a discarded call."""
