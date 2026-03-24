import xml.etree.ElementTree as ET
from typing import Any


def build_svg_gen_prompt(
    original_data_url: str,
    iter_index: int,
    svg_prev: str | None = None,
    svg_prev_invalid_msg: str | None = None,
    rasterized_svg_data_url: str | None = None,
    change_summary: str | None = None,
) -> list[dict[str, Any]]:
    lines = [
        "You are a world-class SVG developer. Convert the input raster into a "
        "CLEAN, optimized SVG.",
        "RULES:",
        "1. Output ONLY the raw <svg>...</svg> code. No markdown, no text.",
        "2. Use SEMANTIC GROUPING: Wrap related paths in <g id='name'>.",
        "3. Fixed Viewport: Use the same width/height/viewBox for all iterations.",
        f"Context: Iteration #{iter_index}.",
    ]

    if svg_prev is None:
        lines.append("First attempt: Create a high-level structural blocking.")
    else:
        lines.append(
            "REFINEMENT TASK: Refine existing groups. Do not delete accurate ones."
        )
        if svg_prev_invalid_msg:
            lines.append(
                f"CRITICAL: Previous SVG failed to parse: {svg_prev_invalid_msg}. "
                "Fix syntax."
            )

    if change_summary:
        lines.extend(["PRIORITY FIXES:", change_summary])

    if svg_prev is not None:
        lines.extend(["CURRENT SVG CODE TO MODIFY:", svg_prev])

    content = [
        {"type": "input_text", "text": "\n".join(lines)},
        {"type": "input_image", "image_url": original_data_url},
    ]
    if rasterized_svg_data_url:
        content.append({"type": "input_image", "image_url": rasterized_svg_data_url})
    return content


def build_summarize_prompt(
    original_data_url: str,
    rasterized_svg_data_url: str | None,
    custom_goal: str | None = None,
) -> list[dict[str, Any]]:
    lines = [
        "Compare the original image (first) to the current SVG render (second).",
        "Provide 3-5 concise, actionable bullet points to improve the likeness.",
    ]

    if custom_goal:
        lines.extend(
            [
                "USER SPECIFIC GOAL:",
                custom_goal,
                "Prioritize this instruction when generating your bullet points.",
            ]
        )

    lines.append("Output ONLY the bullet points.")

    content = [
        {"type": "input_text", "text": "\n".join(lines)},
        {"type": "input_image", "image_url": original_data_url},
    ]
    if rasterized_svg_data_url:
        content.append({"type": "input_image", "image_url": rasterized_svg_data_url})
    return content


def build_crossover_prompt(
    original_data_url: str, svg_a: str, svg_b: str
) -> list[dict[str, Any]]:
    text = (
        "Merge the best elements of Candidate A and Candidate B into a superior SVG.\n"
        "Output ONLY raw <svg> code.\n"
        f"--- CANDIDATE A ---\n{svg_a}\n"
        f"--- CANDIDATE B ---\n{svg_b}"
    )
    return [
        {"type": "input_text", "text": text},
        {"type": "input_image", "image_url": original_data_url},
    ]


def extract_svg_fragment(raw: str) -> str:
    lower = raw.lower()
    start_idx = lower.find("<svg")
    end_idx = lower.rfind("</svg>")
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
