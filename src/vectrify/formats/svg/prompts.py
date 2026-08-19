import xml.etree.ElementTree as ET
from typing import Any

from vectrify.formats.prompts import STRUCTURE_FIRST, diff_format_instructions

_DIFF_FORMAT_INSTRUCTIONS = diff_format_instructions(
    "SVG", unit="fragment", subject="SVG"
)

# The attributes named here are the ones the operators mutate (_NUMERIC_ATTRS,
# _COLOR_ATTRS and mutate_path in formats/svg/operations.py). Keep them in step.
MUTABLE_SVG = """\
Write the SVG this way:
- A shape that is a circle, ellipse or rectangle is written as `<circle>`, \
`<ellipse>` or `<rect>`; everything else is `<path d="...">`. A primitive \
cannot be dented -- it only moves and resizes -- so it survives thousands of \
optimizer steps that a hand-fitted path does not. The best eye any run has \
produced was a white `<circle>` with a smaller black `<circle>` offset inside \
it, where earlier runs fitted two paths and inverted the highlight.
- Numbers in attributes: `x`, `y`, `cx`, `cy`, `r`, `rx`, `ry`, `width`, \
`height`, `x1`, `y1`, `x2`, `y2`, `font-size`, `stroke-width`, `opacity`.
- Coordinates written out directly, already in the viewBox above.
- Colors in `fill` and `stroke` as hex; gradient stops in `stop-color`.
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
) -> list[dict[str, Any]]:
    """Build LLM prompt for SVG generation/refinement.

    *canvas* pins the viewBox. Left to itself the model copies whatever
    dimensions the prompt image happens to have, so changing the raster size
    silently changes the coordinate space candidates are written in -- and
    crossover grafts elements between parents without rescaling them.
    """
    is_edit = svg_prev is not None
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
        "- Wrap related elements in <g id='name'>: the groups are what later"
        " edits and crossover graft between candidates, so they should follow"
        " the target's own parts.",
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
            "The render below is already polished shape by shape, so nudging"
            " its values gains nothing. Change what no amount of polishing"
            " reaches: parts that are missing, extra, the wrong kind of thing,"
            " or in the wrong place — reposition and resize whole parts where"
            " the arrangement is off. Output ONLY search/replace diff blocks,"
            " no full file."
        )

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
