from vectrify.formats.prompts import build_code_gen_prompt

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
    return build_code_gen_prompt(
        image_data_url=image_data_url,
        node_index=node_index,
        code_prev=dot_prev,
        rasterized_data_url=rasterized_dot_data_url,
        goal=goal,
        diff_data_url=diff_data_url,
        lang="DOT",
        lang_display="Graphviz DOT",
        fence="dot",
        syntax_rules=_DOT_SYNTAX_RULES,
        focus_hint=(
            "structure, layout, label text, node/edge attributes, and colors"
        ),
    )
