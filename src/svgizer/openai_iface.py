from __future__ import annotations

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
        "You are an SVG critique engine. Compare the Original Image (A) and the Current SVG Render (B).",
        "Your goal is to guide the SVG generator to make B look like A.",
        "Identify the 3-5 most distinct GEOMETRIC errors. Be specific about:",
        "1. Missing elements (e.g., 'The blue background circle is missing').",
        "2. Shape mismatch (e.g., 'The main rectangle is too wide', 'The corners should be rounded').",
        "3. Alignment issues (e.g., 'The icon is too far to the left').",
        "4. Color discrepancies (e.g., 'The green is too pale').",
        "Output ONLY the bullet points. Do not include introductory text.",
        f"Iteration #{iter_index}.",
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
        "You convert a raster input image into a clean, valid SVG.",
        "Output ONLY a complete <svg>...</svg> with no commentary or backticks.",
        "Prioritize matching the overall likeness of the image: silhouette, composition, large shapes, major color blocks.",
        "Do NOT reproduce noise or artifacts typical of AI-generated images (random speckles, glitchy edges, smeared details, meaningless text or watermark-like blobs). Clean them up in the SVG.",
        "Aim for structural fidelity and clear vector geometry, not pixel-level accuracy.",
        "Ensure the SVG is valid XML with a single <svg> root.",
        f"Iteration #{iter_index}.",
    ]

    if diversity_hint:
        lines.append(f"Diversity hint (to avoid producing the exact same SVG): {diversity_hint}")

    if svg_prev is None:
        lines.append("This is the first attempt. Produce your best high-level SVG approximation.")
    else:
        lines.append("Refine the previous SVG by addressing the differences.")

    if svg_prev_invalid_msg:
        lines.append(
            f"The previous SVG was INVALID:\n{svg_prev_invalid_msg}\n"
            "Return a corrected, valid SVG."
        )

    if change_summary:
        lines.append(
            "Here is a summary of the MOST IMPORTANT changes needed to improve likeness. Use these as priorities:\n"
            + change_summary
        )

    if rasterized_svg_data_url:
        lines.append(
            "You are given the original raster and a rasterized rendering of your current SVG. "
            "Use this to improve the likeness as much as possible. "
            "The priority is to fix overall shapes first, then position of items, and then details like arrows, icons and text."
        )

    if svg_prev:
        lines.append("Here is the previous SVG to refine:\n" + svg_prev)

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
