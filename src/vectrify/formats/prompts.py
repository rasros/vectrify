"""Prompt building blocks shared by the format backends."""

# What the local operators can and cannot reach, stated to the model so it
# spends its one call on the other half. Mutation nudges numbers, shifts
# colors and stroke widths, and moves elements; crossover grafts subtrees
# between candidates. None of that invents a shape that was never proposed,
# removes one that should not be there, or changes what an existing shape is.
STRUCTURE_FIRST = """\
Your output is a starting point: a local optimizer then spends thousands of \
steps on it, moving and resizing parts and tuning their coordinates and \
colors. What it cannot do is invent a shape you left out, remove a structure \
you invented, or change what a shape fundamentally is.

So spend your effort where only you can:
- Every distinct part of the target is present, and nothing extra is.
- Each part is the right kind of thing: an outline that closes is one closed \
path, not two strokes that nearly meet; a filled region is a fill, not a \
thick stroke.
- Counts are exact. Ten circles means ten, not "about ten".
- The arrangement and proportions read correctly at a glance.

Rough coordinates and approximate colors are fine — they get optimized away. \
Do not spend effort deriving exact values."""


def diff_format_instructions(
    lang: str,
    *,
    unit: str = "lines",
    subject: str | None = None,
) -> str:
    """Instructions telling the model to answer with search/replace blocks.

    *lang* names the language ("DOT"), *unit* what a block contains
    ("lines"/"fragment"), *subject* how to refer to the document being
    edited (defaults to "<lang> code").
    """
    subject = subject or f"{lang} code"
    return f"""\
Respond with one or more search/replace blocks — do NOT output the full file.

<<<SEARCH>>>
exact {lang} {unit} to replace (copy verbatim from the current {subject})
<<<REPLACE>>>
improved replacement {unit}
<<<END>>>

Rules:
- The SEARCH text must match the current {subject} exactly (including whitespace).
- Keep blocks small and focused; only change what needs to change.
- Multiple blocks are allowed."""


def build_code_gen_prompt(
    *,
    image_data_url: str,
    node_index: int,
    code_prev: str | None,
    rasterized_data_url: str | None,
    goal: str | None,
    lang: str,
    fence: str,
    syntax_rules: str,
    focus_hint: str,
    lang_display: str | None = None,
    source_name: str | None = None,
) -> list[dict]:
    """Build a generation/refinement prompt for a fenced code format.

    Shared by the DOT and Typst backends, which differ only in *lang*,
    *fence*, *syntax_rules*, and the *focus_hint* steering what to improve.
    *lang_display* names the language in the opening sentence when it differs
    from the short *lang* used for code references (e.g. "Graphviz DOT").
    """
    is_edit = code_prev is not None

    system_text = (
        f"Write {lang_display or lang} code that, when rendered, visually"
        f" matches the target image.\n\n{STRUCTURE_FIRST}\n\n{syntax_rules}"
    )
    if source_name:
        # Often the only place the subject is stated, and the model is
        # otherwise working from the picture alone.
        system_text += (
            f"\n- The file is named `{source_name}`. Filenames often name the"
            " subject; weigh it against what you see, and trust the image if"
            " they disagree."
        )
    if is_edit:
        system_text += "\n- Output ONLY search/replace diff blocks, no full file"
    else:
        system_text += f"\n- Output ONLY the {lang} code block, no explanation"

    blocks: list[dict] = [
        {"type": "input_text", "text": system_text},
        {"type": "input_text", "text": "Target image:"},
        {"type": "input_image", "image_url": image_data_url},
    ]

    if not is_edit:
        seed_text = (
            f"Iteration #{node_index}. Write complete {lang} code."
            f" Wrap in ```{fence}\n...\n```"
        )
        if goal:
            seed_text += f"\nUser goal: {goal}"
        blocks.append({"type": "input_text", "text": seed_text})
        return blocks

    if rasterized_data_url:
        blocks.append({"type": "input_text", "text": "Current rendered output:"})
        blocks.append({"type": "input_image", "image_url": rasterized_data_url})

    # The render is already a local optimum in the numeric directions -- it got
    # there by hill-climbing -- so asking for structural change is asking for
    # the only thing left that the next few thousand local steps cannot reach.
    edit_text = (
        f"Iteration #{node_index}. "
        f"The current render below has already been numerically optimized, so "
        f"tuning its values further gains nothing. Find what is structurally "
        f"wrong or missing against the target and fix that: {focus_hint}.\n"
    )
    if goal:
        edit_text += f"\nUser goal (highest priority): {goal}\n"
    edit_text += (
        f"\nCurrent {lang} code:\n```{fence}\n{code_prev}\n```\n\n"
        + diff_format_instructions(lang)
    )
    blocks.append({"type": "input_text", "text": edit_text})

    return blocks
