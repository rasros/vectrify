import xml.etree.ElementTree as ET
from typing import Any


def build_svg_gen_prompt(
    original_data_url: str,
    iter_index: int,
    svg_prev: str | None = None,
    svg_prev_invalid_msg: str | None = None,
    rasterized_svg_data_url: str | None = None,
    change_summary: str | None = None,
    diff_data_url: str | None = None,
    force_diverse: bool = False,
) -> list[dict[str, Any]]:
    lines = [
        "You are a world-class SVG developer. Convert the input raster into a CLEAN,"
        " optimized SVG.",
        "RULES:",
        "1. Scratchpad: Think step-by-step in a <plan>...</plan> block about "
        "coordinates and geometry before writing code.",
        "2. Output ONLY the raw <svg>...</svg> code immediately after your "
        "<plan>. No markdown.",
        "3. Use SEMANTIC GROUPING: Wrap related paths in <g id='name'>.",
        "4. Fixed Viewport: Use the same width/height/viewBox for all iterations.",
        f"Context: Iteration #{iter_index}.",
    ]

    if force_diverse:
        lines.append(
            "DIVERSITY SEED: The search has converged. Ignore any prior SVG. "
            "Produce a FRESH structural interpretation using completely different "
            "primitives, geometry, or decomposition than you might otherwise attempt."
        )
    elif svg_prev is None:
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
        lines.extend(["PRIORITY FIXES FROM JUDGE:", change_summary])

    if svg_prev is not None:
        lines.extend(["CURRENT SVG CODE TO MODIFY:", svg_prev])

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


def build_summarize_prompt(
    original_data_url: str,
    rasterized_svg_data_url: str | None,
    custom_goal: str | None = None,
    previous_summary: str | None = None,
) -> list[dict[str, Any]]:
    lines = [
        "You are a technical design critic. Compare the original image (first) "
        "to the current SVG render (second).",
        "Provide 3-5 concise, technical bullet points to improve the likeness.",
        "GUIDELINES:",
        "- Use SVG terminology: refer to 'stroke-width', 'opacity', "
        "'coordinates', or 'viewBox'.",
        "- Categorize feedback: [Geometry/Layout], [Typography/Text], "
        "or [Color/Style].",
        "- Be spatially specific: instead of 'move up', say 'shift the "
        "y-coordinate higher'.",
    ]

    if previous_summary:
        lines.extend(
            [
                "PREVIOUS FEEDBACK GIVEN:",
                previous_summary,
                "Identify if these points were ignored and emphasize "
                "them if still relevant.",
            ]
        )

    if custom_goal:
        lines.extend(
            [
                "USER SPECIFIC GOAL:",
                custom_goal,
                "Prioritize this instruction over all other visual improvements.",
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
    # Use rfind to avoid matching <svg> tags discussed inside the <plan> block
    lower = raw.lower()
    end_idx = lower.rfind("</svg>")
    if end_idx != -1:
        start_idx = lower.rfind("<svg", 0, end_idx)
        if start_idx != -1:
            return raw[start_idx : end_idx + 6].strip()

    # Fallback
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
