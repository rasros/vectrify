_DIFF_FORMAT_INSTRUCTIONS = """\
Respond with one or more search/replace blocks — do NOT output the full file.

<<<SEARCH>>>
exact Typst lines to replace (copy verbatim from the current Typst code)
<<<REPLACE>>>
improved replacement lines
<<<END>>>

Rules:
- The SEARCH text must match the current Typst code exactly (including whitespace).
- Keep blocks small and focused; only change what needs to change.
- Multiple blocks are allowed."""

_TYPST_SYNTAX_RULES = (
    "Typst syntax rules:\n"
    "- ALWAYS start your document with a single page setup that forces auto-sizing:\n"
    "  `#set page(width: auto, height: auto, margin: 0pt)`\n"
    '- Use `#set text(font: "Arial", size: 12pt)` for font settings.\n'
    "- Use standard Typst shapes like `#rect(...)`, `#circle(...)`, `#line(...)`.\n"
    '- Colors can be specified as `rgb("ff0000")` or predefined like `red`.\n'
    "- Use `#place(dx: 10pt, dy: 20pt)[...]` for absolute positioning"
    " to match images.\n"
    "- NEVER use multiple pages; everything must fit on one auto-sized page."
)


def build_typst_gen_prompt(
    image_data_url: str,
    node_index: int,
    typst_prev: str | None,
    rasterized_data_url: str | None,
    goal: str | None,
    diff_data_url: str | None,
) -> list[dict]:
    """Build LLM prompt for Typst generation/refinement."""
    is_edit = typst_prev is not None

    system_text = (
        "Write Typst code that, when rendered, visually matches the target image.\n\n"
        + _TYPST_SYNTAX_RULES
    )

    if not is_edit:
        system_text += "\n- Output ONLY the Typst code block, no explanation"
    else:
        system_text += "\n- Output ONLY search/replace diff blocks, no full file"

    blocks: list[dict] = [{"type": "input_text", "text": system_text}]
    blocks.append({"type": "input_text", "text": "Target image:"})
    blocks.append({"type": "input_image", "image_url": image_data_url})

    if not is_edit:
        seed_text = (
            f"Iteration #{node_index}. Write complete Typst code. "
            "Wrap in ```typst\n...\n```"
        )
        if goal:
            seed_text += f"\nUser goal: {goal}"
        blocks.append({"type": "input_text", "text": seed_text})
    else:
        if rasterized_data_url:
            blocks.append({"type": "input_text", "text": "Current rendered output:"})
            blocks.append({"type": "input_image", "image_url": rasterized_data_url})

        if diff_data_url:
            blocks.append(
                {
                    "type": "input_text",
                    "text": "Difference map (bright = mismatch — focus edits here):",
                }
            )
            blocks.append({"type": "input_image", "image_url": diff_data_url})

        edit_text = (
            f"Iteration #{node_index}. "
            "Improve the Typst code to better match the target. "
            "Focus on alignment, sizes, paddings, and colors.\n"
        )
        if goal:
            edit_text += f"\nUser goal (highest priority): {goal}\n"
        edit_text += (
            f"\nCurrent Typst code:\n```typst\n{typst_prev}\n```\n\n"
            + _DIFF_FORMAT_INSTRUCTIONS
        )
        blocks.append({"type": "input_text", "text": edit_text})

    return blocks
