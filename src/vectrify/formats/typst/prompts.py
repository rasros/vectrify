from vectrify.formats.prompts import build_code_gen_prompt

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
    return build_code_gen_prompt(
        image_data_url=image_data_url,
        node_index=node_index,
        code_prev=typst_prev,
        rasterized_data_url=rasterized_data_url,
        goal=goal,
        diff_data_url=diff_data_url,
        lang="Typst",
        fence="typst",
        syntax_rules=_TYPST_SYNTAX_RULES,
        focus_hint="alignment, sizes, paddings, and colors",
    )
