from vectrify.formats.prompts import build_code_gen_prompt


def _typst_syntax_rules(canvas: tuple[int, int]) -> str:
    """Typst rules with the page pinned to the raster.

    Auto-sizing makes the coordinate space implicit: the page grows to fit
    whatever content it has, so the same `#place(dx:, dy:)` means a different
    position depending on the document's extent. Crossover grafts placements
    between candidates, which changes that extent and silently rescales
    everything. A fixed page is the Typst equivalent of pinning the SVG
    viewBox.
    """
    w, h = canvas
    return (
        "Typst syntax rules:\n"
        "- ALWAYS start your document with this exact page setup:\n"
        f"  `#set page(width: {w}pt, height: {h}pt, margin: 0pt)`\n"
        f"  The canvas is {w}x{h}pt; express every coordinate in it.\n"
        '- Use `#set text(font: "Arial", size: 12pt)` for font settings.\n'
        "- Use standard Typst shapes like `#rect(...)`, `#circle(...)`,"
        " `#line(...)`.\n"
        '- Colors can be specified as `rgb("ff0000")` or predefined like `red`.\n'
        "- Use `#place(dx: 10pt, dy: 20pt)[...]` for absolute positioning"
        " to match images.\n"
        "- NEVER use multiple pages; everything must fit on the one page."
    )


def build_typst_gen_prompt(
    image_data_url: str,
    node_index: int,
    typst_prev: str | None,
    rasterized_data_url: str | None,
    goal: str | None,
    canvas: tuple[int, int] = (0, 0),
) -> list[dict]:
    """Build LLM prompt for Typst generation/refinement."""
    return build_code_gen_prompt(
        image_data_url=image_data_url,
        node_index=node_index,
        code_prev=typst_prev,
        rasterized_data_url=rasterized_data_url,
        goal=goal,
        lang="Typst",
        fence="typst",
        syntax_rules=_typst_syntax_rules(canvas),
        focus_hint="alignment, sizes, paddings, and colors",
    )
