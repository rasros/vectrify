from typing import Optional, List
import os
import xml.etree.ElementTree as ET

from openai import OpenAI


MODEL_NAME = os.getenv("SVGIZER_OPENAI_MODEL", "gpt-5.4")


def extract_svg_fragment(raw: str) -> str:
    lower = raw.lower()
    start_idx = lower.find("<svg")
    end_idx = lower.rfind("</svg>")
    if start_idx != -1 and end_idx != -1:
        end_idx += len("</svg>")
        return raw[start_idx:end_idx].strip()
    return raw.strip()


def is_valid_svg(svg_text: str):
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as e:
        return False, f"XML parse error: {e}"
    if root.tag.lower().endswith("svg"):
        return True, None
    return False, f"Root tag is not <svg>: got <{root.tag}>"


def summarize_changes(
    client: OpenAI,
    original_data_url: str,
    iter_index: int,
    rasterized_svg_data_url: Optional[str],
) -> str:
    lines = [
        "You are a Senior SVG QA Engineer. Compare Original (A) and current SVG Render (B).",
        "Your goal: Provide actionable geometric feedback to minimize the perceptual distance.",
        "",
        "### STEP 1: QUADRANT ANALYSIS",
        "Mentally divide the image into a 2x2 grid (Top-Left, Top-Right, Bottom-Left, Bottom-Right).",
        "Identify which quadrant has the most 'broken' geometry or incorrect color fill.",
        "",
        "### STEP 2: GROUP IDENTIFICATION",
        "Look at the SVG code (if provided in context later) or the render.",
        "Reference specific elements by their logical names (e.g., 'the background group', 'the central icon', 'the shadow path').",
        "",
        "### STEP 3: OUTPUT REQUIREMENTS",
        "- Provide 3-5 concise bullet points.",
        "- Be spatially specific (e.g., 'In the Top-Right, the border-radius is too sharp').",
        "- Identify if a group is missing entirely.",
        "- Output ONLY the bullet points. No intro or outro.",
        f"Iteration Context: #{iter_index}.",
    ]

    content: List[dict] = [
        {"type": "input_text", "text": "\n".join(lines)},
        {"type": "input_image", "image_url": original_data_url},
    ]

    if rasterized_svg_data_url:
        content.append({"type": "input_image", "image_url": rasterized_svg_data_url})

    resp = client.responses.create(
        model=MODEL_NAME,
        input=[{"role": "user", "content": content}],
        temperature=0.2,
        text={"format": {"type": "text"}},
    )
    return resp.output_text


def call_openai_for_svg(
    client: OpenAI,
    original_data_url: str,
    iter_index: int,
    temperature: float,
    svg_prev: Optional[str] = None,
    svg_prev_invalid_msg: Optional[str] = None,
    rasterized_svg_data_url: Optional[str] = None,
    change_summary: Optional[str] = None,
    diversity_hint: Optional[str] = None,
) -> str:
    lines = [
        "You are a world-class SVG developer. Convert the input raster into a CLEAN, optimized SVG.",
        "RULES:",
        "1. Output ONLY the raw <svg>...</svg> code. No markdown, no backticks, no text.",
        "2. Use SEMANTIC GROUPING: Wrap related paths in <g id='name'> (e.g., <g id='main_subject'>, <g id='background_shapes'>).",
        "3. Fixed Viewport: Use the same width/height/viewBox for all iterations to prevent drift.",
        "4. No Noise: Do not attempt to vectorize compression artifacts or grain.",
        f"Context: Iteration #{iter_index}.",
    ]

    if diversity_hint:
        lines.append(f"Diversity hint: {diversity_hint}")

    if svg_prev is None:
        lines.append("First attempt: Create a high-level structural blocking of the image using large paths and groups.")
    else:
        lines.append("REFINEMENT TASK:")
        lines.append("Refine the existing SVG groups. Do not delete groups that are already accurate.")

    if svg_prev_invalid_msg:
        lines.append(f"CRITICAL: The previous SVG failed to parse: {svg_prev_invalid_msg}. Fix the syntax immediately.")

    if change_summary:
        lines.append("PRIORITY FIXES (from Vision Critique):")
        lines.append(change_summary)

    if rasterized_svg_data_url:
        lines.append(
            "You are given the original raster and a rasterized rendering of your current SVG. "
            "Use this to improve the likeness as much as possible. "
            "The priority is to fix overall shapes first, then position of items, and then details like arrows, icons and text."
        )

    if svg_prev:
        lines.append("CURRENT SVG CODE TO MODIFY:")
        lines.append(svg_prev)

    content: List[dict] = [
        {"type": "input_text", "text": "\n".join(lines)},
        {"type": "input_image", "image_url": original_data_url},
    ]

    if rasterized_svg_data_url:
        content.append({"type": "input_image", "image_url": rasterized_svg_data_url})

    response = client.responses.create(
        model=MODEL_NAME,
        input=[{"role": "user", "content": content}],
        temperature=temperature,
        text={"format": {"type": "text"}},
    )

    return response.output_text
