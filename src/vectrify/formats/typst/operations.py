import random
import re

from PIL import Image

from vectrify.formats.micro_search import with_micro_search

# Matches numeric values with Typst units: 12pt, 1.5em, 50%, 3mm, etc.
_NUM_RE = re.compile(r"(\b|-)(\d+(?:\.\d+)?)(pt|em|%|mm|cm|in)\b")

# Matches a simple named color after fill: or stroke:
_NAMED_COLOR_ATTR_RE = re.compile(r"\b(fill|stroke)\s*:\s*([a-z]+)\b")

# Lines that are renderable shape/layout elements (not page/text setup)
_ELEMENT_LINE_RE = re.compile(
    r"^\s*#(rect|circle|ellipse|line|polygon|place|square|path)\b",
    re.MULTILINE,
)

_TYPST_COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "cyan",
    "magenta",
    "pink",
    "black",
    "white",
    "gray",
    "navy",
    "teal",
    "olive",
    "coral",
    "gold",
    "lime",
    "maroon",
    "silver",
]


def _random_numeric_tweak(typst_code: str) -> str:
    """Find a random numeric value with a unit and scale it by ±30%."""
    matches = list(_NUM_RE.finditer(typst_code))
    if not matches:
        return typst_code

    m = random.choice(matches)
    prefix = m.group(1)
    val = float(m.group(2))
    unit = m.group(3)

    factor = random.uniform(0.7, 1.3)
    new_val = max(0.1, val * factor)
    formatted = f"{prefix}{new_val:.2f}{unit}".replace(".00", "")

    return typst_code[: m.start()] + formatted + typst_code[m.end() :]


def _mutate_color(typst_code: str) -> str:
    """Replace a named fill or stroke color with a random one."""
    matches = list(_NAMED_COLOR_ATTR_RE.finditer(typst_code))
    if not matches:
        return typst_code

    m = random.choice(matches)
    current = m.group(2)
    candidates = [c for c in _TYPST_COLORS if c != current]
    if not candidates:
        return typst_code

    new_color = random.choice(candidates)
    start = m.start(2)
    end = m.end(2)
    return typst_code[:start] + new_color + typst_code[end:]


def _remove_element(typst_code: str) -> str:
    """Remove a random shape element line, keeping at least one."""
    lines = typst_code.splitlines(keepends=True)
    element_indices = [
        i for i, line in enumerate(lines) if _ELEMENT_LINE_RE.match(line)
    ]
    if len(element_indices) <= 1:
        return typst_code

    idx = random.choice(element_indices)
    return "".join(line for i, line in enumerate(lines) if i != idx)


def _reorder_elements(typst_code: str) -> str:
    """Swap two element lines to change rendering order."""
    lines = typst_code.splitlines(keepends=True)
    element_indices = [
        i for i, line in enumerate(lines) if _ELEMENT_LINE_RE.match(line)
    ]
    if len(element_indices) < 2:
        return typst_code

    i, j = random.sample(element_indices, 2)
    lines[i], lines[j] = lines[j], lines[i]
    return "".join(lines)


def _apply_one_mutation(typst_code: str) -> tuple[str, str]:
    _ops = [
        (_mutate_color, "Mutation: color tweak", 0.30),
        (_random_numeric_tweak, "Mutation: numeric tweak", 0.30),
        (_remove_element, "Mutation: removed element", 0.20),
        (_reorder_elements, "Mutation: reordered elements", 0.20),
    ]
    fns, labels, weights = zip(*_ops, strict=True)
    fn, label = random.choices(
        list(zip(fns, labels, strict=True)), weights=list(weights), k=1
    )[0]
    return fn(typst_code), label


def render_typst_png(typst_code: str) -> bytes:
    """Render Typst source to PNG bytes (first page). Raises on failure."""
    import typst

    # Must encode to bytes — typst-py treats a plain str as a file path
    result = typst.compile(typst_code.encode("utf-8"), format="png", ppi=144)
    if isinstance(result, list):
        if not result:
            raise ValueError("Typst generated zero pages.")
        return result[0]
    if isinstance(result, bytes):
        return result
    raise ValueError("Failed to rasterize Typst to PNG bytes.")


def _rasterize_typst(typst_code: str) -> bytes | None:
    try:
        return render_typst_png(typst_code)
    except Exception:
        return None


def mutate_with_micro_search(
    parent_code: str,
    orig_img_fast: Image.Image,
    num_trials: int = 15,
) -> tuple[str, str]:
    return with_micro_search(
        lambda: _apply_one_mutation(parent_code),
        fallback=parent_code,
        rasterize=_rasterize_typst,
        orig_img_fast=orig_img_fast,
        num_trials=num_trials,
        default_summary="Mutation: no improvement",
    )


def crossover_with_micro_search(
    code_a: str,
    code_b: str,
    orig_img_fast: Image.Image,
    num_trials: int = 15,
) -> tuple[str, str]:
    # Extract non-empty element lines from B to inject into A
    lines_b = [
        line
        for line in code_b.splitlines(keepends=True)
        if _ELEMENT_LINE_RE.match(line)
    ]
    lines_a = code_a.splitlines(keepends=True)
    element_indices_a = [
        i for i, line in enumerate(lines_a) if _ELEMENT_LINE_RE.match(line)
    ]

    if not lines_b or not element_indices_a:
        return mutate_with_micro_search(code_a, orig_img_fast, num_trials)

    def _op() -> tuple[str, str]:
        insert_after = random.choice(element_indices_a)
        candidate_lines = [
            *lines_a[: insert_after + 1],
            random.choice(lines_b),
            *lines_a[insert_after + 1 :],
        ]
        return "".join(candidate_lines), "Crossover: element injection"

    return with_micro_search(
        _op,
        fallback=code_a,
        rasterize=_rasterize_typst,
        orig_img_fast=orig_img_fast,
        num_trials=num_trials,
        default_summary="Crossover: no improvement",
    )
