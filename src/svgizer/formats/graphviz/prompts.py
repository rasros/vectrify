_DIFF_FORMAT_INSTRUCTIONS = """\
Respond with one or more search/replace blocks — do NOT output the full file.

<<<SEARCH>>>
exact DOT lines to replace (copy verbatim from the current DOT code)
<<<REPLACE>>>
improved replacement lines
<<<END>>>

Rules:
- The SEARCH text must match the current DOT code exactly (including whitespace).
- Keep blocks small and focused; only change what needs to change.
- Multiple blocks are allowed."""

_DOT_SYNTAX_RULES = (
    "DOT syntax rules:\n"
    "- Use `digraph G { ... }` for directed graphs (with -> edges) or "
    "`graph G { ... }` for undirected (with -- edges)\n"
    "- CRITICAL: -> edges require digraph; "
    "-- edges require graph. Never mix them.\n"
    "- Node declarations: "
    '`NodeName [label="...", shape=box, style=filled, fillcolor=lightblue];`\n'
    '- Edge declarations: `A -> B [label="...", style=dashed];`  (in a digraph)\n'
    '- ALWAYS use plain quoted strings for labels: label="My Node"\n'
    "- NEVER use HTML-style labels like label=<B>text</B> — "
    "they cause parse errors\n"
    "- Graph attributes at top: `rankdir=LR; splines=ortho; nodesep=0.5;`\n"
    "- Layout engines: dot (hierarchical), neato (spring), "
    "fdp (force-directed), circo (radial)\n"
    "- Common shapes: box, ellipse, circle, diamond, parallelogram, "
    "hexagon, doublecircle"
)


def build_dot_gen_prompt(
    image_data_url: str,
    node_index: int,
    dot_prev: str | None,
    rasterized_dot_data_url: str | None,
    goal: str | None,
    diff_data_url: str | None,
) -> list[dict]:
    """Build LLM prompt for DOT graph generation/refinement."""
    is_edit = dot_prev is not None

    system_text = (
        "Write Graphviz DOT code that, when rendered, visually matches the target image.\n\n"
        + _DOT_SYNTAX_RULES
    )

    if not is_edit:
        system_text += "\n- Output ONLY the DOT code block, no explanation"
    else:
        system_text += "\n- Output ONLY search/replace diff blocks, no full file"

    blocks: list[dict] = [{"type": "text", "text": system_text}]
    blocks.append({"type": "text", "text": "Target image:"})
    blocks.append({"type": "image_url", "image_url": {"url": image_data_url}})

    if not is_edit:
        seed_text = f"Iteration #{node_index}. Write complete DOT code. Wrap in ```dot\\n...\\n```"
        if goal:
            seed_text += f"\nUser goal: {goal}"
        blocks.append({"type": "text", "text": seed_text})
    else:
        if rasterized_dot_data_url:
            blocks.append({"type": "text", "text": "Current rendered output:"})
            blocks.append(
                {"type": "image_url", "image_url": {"url": rasterized_dot_data_url}}
            )

        if diff_data_url:
            blocks.append(
                {"type": "text", "text": "Difference map (bright = mismatch):"}
            )
            blocks.append({"type": "image_url", "image_url": {"url": diff_data_url}})

        edit_text = (
            f"Iteration #{node_index}. "
            "Improve the DOT code to better match the target. "
            "Focus on structure, layout, node/edge attributes, and colors.\n"
        )
        if goal:
            edit_text += f"\nUser goal (highest priority): {goal}\n"
        edit_text += (
            f"\nCurrent DOT code:\n```dot\n{dot_prev}\n```\n\n"
            + _DIFF_FORMAT_INSTRUCTIONS
        )
        blocks.append({"type": "text", "text": edit_text})

    return blocks
