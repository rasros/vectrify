import xml.etree.ElementTree as ET
from typing import Any

# What the editor's refinement can and cannot reach, stated to the model so it
# spends its call on the other half. Optimize nodes moves, adds and removes a
# path's points and moves whole paths; Fit colours sets flat fills. None of that
# invents a shape that was never drawn, removes one that should not be there,
# or changes what an existing shape is.
STRUCTURE_FIRST = """\
Your output is a starting point that can be refined afterwards: path points \
are moved, added and removed, whole paths are moved, and flat fill colours are \
fitted. What refinement cannot do is invent a shape you left out, remove a \
structure you invented, or change what a shape fundamentally is.

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

Rough path coordinates and approximate colors are fine — they can be refined. \
Do not spend effort deriving exact values for them."""


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


_DIFF_FORMAT_INSTRUCTIONS = diff_format_instructions(
    "SVG", unit="fragment", subject="SVG"
)

# Written for what refinement reaches afterwards: Optimize nodes reshapes paths
# point by point and leaves primitives alone; Fit colours sets flat fills.
MUTABLE_SVG = """\
Write the SVG this way:
- A shape that is exactly a circle, ellipse or rectangle is written as \
`<circle>`, `<ellipse>` or `<rect>`, placed and sized with care: refinement \
reshapes paths only, so a primitive stays as you write it. The best eye any \
run has produced was a white `<circle>` with a smaller black `<circle>` offset \
inside it, where earlier runs fitted two paths and inverted the highlight.
- Everything else is `<path d="...">`, whose points refinement moves, adds \
and removes.
- Coordinates written out directly, already in the viewBox above.
- Colors in `fill` and `stroke` as flat hex values; refinement fits flat fills.
- Each shape its own element, with its own attributes.
- Many small explicit elements rather than one clever construction."""


def build_svg_gen_prompt(
    original_data_url: str,
    iter_index: int,
    svg_prev: str | None = None,
    rasterized_svg_data_url: str | None = None,
    goal: str | None = None,
    canvas: tuple[int, int] = (0, 0),
    source_name: str | None = None,
    invisible: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build LLM prompt for SVG generation/refinement.

    *canvas* pins the viewBox. Left to itself the model copies whatever
    dimensions the prompt image happens to have, so changing the raster size
    would silently change the coordinate space its drawing is written in.
    """
    is_edit = svg_prev is not None
    if not is_edit:
        # A render of nothing tells the model nothing; only an edit shows one.
        rasterized_svg_data_url = None
    view_w, view_h = canvas
    # The file often names the subject, and the model is otherwise working from
    # the picture alone: one model read a connect-the-dots duck as a banana and
    # two moons, named its groups accordingly, and drew a crescent where the
    # eye's highlight belonged. Offered as evidence rather than instruction --
    # plenty of files are called scan_04.png, and the image wins if they
    # disagree.
    subject_line = (
        f"The file is named `{source_name}`. Filenames often name the subject;"
        " weigh it against what you see, and trust the image if they disagree."
        if source_name
        else None
    )

    lines = [
        "Reproduce the target image as SVG code.",
        "- Always include `xmlns='http://www.w3.org/2000/svg'` and"
        f" `viewBox='0 0 {view_w} {view_h}'` on the root <svg> element."
        " Use exactly this viewBox and express every coordinate in it.",
        "- Work out what the picture depicts before drawing it, and name each"
        " <g id='name'> after the part it is: `beak`, `eye`, `wing`. The names"
        " are the record of that reading, and every later edit works from"
        " them, so a part named for what it resembles rather than what it is"
        " gets drawn as that instead. A run whose groups came back as"
        " `large_crescent` and `dark_moon` drew a crescent moon where the"
        " target had an eye with a highlight, and never recovered: nothing"
        " downstream can tell that the subject was misread.",
        "- One group per part, however many strokes it takes: a wing drawn as"
        " a sweep and three feathers is one `wing`, not a `body_outline` and a"
        " `tail_feathers`.",
        "- Wrap related elements in <g id='name'>: the groups are what later"
        " edits select and work on, so they should follow the target's own"
        " parts.",
        "",
        STRUCTURE_FIRST,
        "",
        MUTABLE_SVG,
        "",
        f"Iteration #{iter_index}.",
    ]
    if subject_line:
        lines.insert(1, subject_line)

    if not is_edit:
        lines.append("Output ONLY the raw <svg>...</svg>. No markdown.")
    else:
        lines.append(
            "The render of the current SVG is shown below the target. Make the"
            " change the goal asks for. Where it leaves you a choice, prefer"
            " what refinement cannot reach: parts that are missing, extra, the"
            " wrong kind of thing, or in the wrong place. Leave small value"
            " tweaks to refinement."
        )

    # Named because the model has no other way to know. On screen these are
    # simply absent, and in the markup they look like any other element -- so a
    # drawing can carry a nostril or an eye highlight through a whole run while
    # painting neither. An edit is the natural fix: making an occluded element
    # show needs its draw order and its position changed together, and each
    # search mutation does only one of those.
    if is_edit and invisible:
        lines.append(
            "These elements are in the SVG and paint nothing — hidden behind"
            " something drawn later, or sitting on background of their own"
            " colour. They are absent from the render, so you cannot see the"
            " problem. Give each one a position and a draw order where it shows,"
            " or delete it:"
        )
        lines.extend(f"  {entry}" for entry in invisible)

    if goal:
        lines.extend(["USER GOAL (highest priority):", goal])

    if is_edit:
        lines.extend(
            ["CURRENT SVG CODE TO MODIFY:", svg_prev, _DIFF_FORMAT_INSTRUCTIONS]
        )

    content = [
        {"type": "input_text", "text": "\n".join(lines)},
        {"type": "input_text", "text": "Target Image:"},
        {"type": "input_image", "image_url": original_data_url},
    ]

    if rasterized_svg_data_url:
        content.append({"type": "input_text", "text": "Your Current SVG Render:"})
        content.append({"type": "input_image", "image_url": rasterized_svg_data_url})

    return content


def extract_svg_fragment(raw: str) -> str:
    """Extract <svg> tag from LLM response text."""
    lower = raw.lower()
    end_idx = lower.rfind("</svg>")
    if end_idx != -1:
        start_idx = lower.rfind("<svg", 0, end_idx)
        if start_idx != -1:
            return raw[start_idx : end_idx + 6].strip()

    start_idx = lower.find("<svg")
    if start_idx != -1 and end_idx != -1:
        return raw[start_idx : end_idx + 6].strip()
    return raw.strip()


def is_valid_svg(svg_text: str) -> tuple[bool, str | None]:
    try:
        root = ET.fromstring(svg_text)
        if root.tag.lower().endswith("svg"):
            return True, None
        return False, f"Root tag is not <svg>: got <{root.tag}>"
    except ET.ParseError as e:
        return False, f"XML parse error: {e}"
