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
- A part can be tiny. A nostril, a pupil's highlight, a dot inside a shape: \
these are parts, not detail, and they are the ones most often dropped. Four \
runs of one drawing put the beak, eye and wing in every time and the nostril \
in none.
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
Respond with one or more search/replace blocks. Prefer them to a full rewrite:
a block changes only what it names, where re-authoring the whole {subject}
retypes every part you were not trying to change.

<<<SEARCH>>>
exact {lang} {unit} to replace (copy verbatim from the current {subject})
<<<REPLACE>>>
improved replacement {unit}
<<<END>>>

Rules:
- Always return at least one block. If nothing looks clearly wrong, take the \
part that matches the target least well and improve that; a reply with no \
block is a discarded call, not a verdict that the drawing is finished.
- The SEARCH text must match the current {subject} character for character, \
though how the whitespace inside it is written does not matter.
- Keep blocks small and focused; only change what needs to change.
- Multiple blocks are allowed.
- If you cannot copy the text to replace exactly, output the complete \
{subject} instead. That is worth more than a block that matches nothing: a \
reply with neither is a discarded call."""


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
        system_text += (
            "\n- Output search/replace diff blocks, or the whole file if you"
            " cannot copy the text to replace exactly. No explanation."
        )
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
