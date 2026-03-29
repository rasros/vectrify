def build_dot_gen_prompt(
    image_data_url: str,
    node_index: int,
    dot_prev: str | None,
    rasterized_dot_data_url: str | None,
    change_summary: str | None,
    diff_data_url: str | None,
) -> list[dict]:
    """Build LLM prompt for DOT graph generation/refinement."""
    is_first = dot_prev is None

    system_text = (
        "You are an expert graph visualization engineer. "
        "Your task is to write Graphviz DOT language code that, when rendered, "
        "visually matches a target image as closely as possible.\n\n"
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
        "hexagon, doublecircle\n"
        "- Output ONLY the DOT code block, no explanation\n"
    )

    blocks: list[dict] = [{"type": "text", "text": system_text}]

    blocks.append({"type": "text", "text": "Target image to reproduce as a DOT graph:"})
    blocks.append({"type": "image_url", "image_url": {"url": image_data_url}})

    if is_first:
        blocks.append(
            {
                "type": "text",
                "text": (
                    "This is iteration #1. Analyze the target image carefully:\n"
                    "- What graph structure does it show? "
                    "(hierarchy, network, flow, tree?)\n"
                    "- What are the main nodes and their relationships?\n"
                    "- What visual style is used? "
                    "(colors, shapes, layout direction)\n\n"
                    "Write complete DOT code that reproduces this structure.\n"
                    "Wrap your code in ```dot\\n...\\n```"
                ),
            }
        )
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

        feedback_text = f"Design feedback:\n{change_summary}" if change_summary else ""
        blocks.append(
            {
                "type": "text",
                "text": (
                    f"Iteration #{node_index}. {feedback_text}\n\n"
                    "Current DOT code:\n"
                    f"```dot\n{dot_prev}\n```\n\n"
                    "Improve the DOT code to better match the target. "
                    "Focus on structure, layout, node/edge attributes, and colors. "
                    "Output ONLY the updated DOT code:\n"
                    "```dot\n...\n```"
                ),
            }
        )

    return blocks


def build_dot_summarize_prompt(
    image_data_url: str,
    raster_preview_url: str | None,
    custom_goal: str | None,
    previous_summary: str | None,
) -> list[dict]:
    """Build LLM prompt to critique the current DOT render vs target."""
    blocks: list[dict] = []

    goal_text = custom_goal or "reproduce the target diagram as accurately as possible"
    prev_text = f"\nPrevious feedback: {previous_summary}" if previous_summary else ""

    blocks.append(
        {
            "type": "text",
            "text": (
                f"You are a graph visualization critic. Goal: {goal_text}.{prev_text}\n"
                "Compare the target and current render. Identify 3-5 specific issues:\n"
                "- Missing or wrong nodes/edges\n"
                "- Incorrect layout direction or structure\n"
                "- Wrong node shapes, colors, or styles\n"
                "- Label errors\n"
                "Be specific and technical (e.g. 'add edge A->C', "
                "'change node B shape to diamond')."
            ),
        }
    )
    blocks.append({"type": "text", "text": "Target:"})
    blocks.append({"type": "image_url", "image_url": {"url": image_data_url}})
    if raster_preview_url:
        blocks.append({"type": "text", "text": "Current render:"})
        blocks.append({"type": "image_url", "image_url": {"url": raster_preview_url}})

    return blocks
