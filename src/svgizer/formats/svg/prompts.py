import xml.etree.ElementTree as ET
from typing import Any

_DIFF_FORMAT_INSTRUCTIONS = """\
Respond with one or more search/replace blocks — do NOT output the full file.

<<<SEARCH>>>
exact SVG fragment to replace (copy verbatim from the current SVG)
<<<REPLACE>>>
improved replacement fragment
<<<END>>>

Rules:
- The SEARCH text must match the current SVG exactly (including whitespace).
- Keep blocks small and focused; only change what needs to change.
- Multiple blocks are allowed."""


def build_svg_gen_prompt(
    original_data_url: str,
    iter_index: int,
    svg_prev: str | None = None,
    svg_prev_invalid_msg: str | None = None,
    rasterized_svg_data_url: str | None = None,
    goal: str | None = None,
    diff_data_url: str | None = None,
) -> list[dict[str, Any]]:
    is_edit = svg_prev is not None

    lines = [
        "Convert the target image into SVG code that reproduces it as accurately as possible.",
        "- Wrap related elements in <g id='name'>.",
        f"Iteration #{iter_index}.",
    ]

    if not is_edit:
        lines.append("Output ONLY the raw <svg>...</svg>. No markdown.")
    else:
        lines.append(
            "Output ONLY search/replace diff blocks. "
            "No full file. Do not remove accurate elements."
        )
        if svg_prev_invalid_msg:
            lines.append(
                f"CRITICAL: Previous SVG failed to parse: {svg_prev_invalid_msg}. "
                "Fix syntax."
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

    if diff_data_url:
        content.append(
            {
                "type": "input_text",
                "text": "Difference Map (Target vs Current Render - "
                "bright pixels indicate geometric/color errors):",
            }
        )
        content.append({"type": "input_image", "image_url": diff_data_url})

    return content


def extract_svg_fragment(raw: str) -> str:
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
