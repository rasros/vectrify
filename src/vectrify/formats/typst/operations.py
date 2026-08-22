"""Structure-aware local search operators for Typst scenes."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from vectrify.formats.mutations import MutationTable, pick_operator

_VISUAL_NAMES = frozenset(
    {"rect", "circle", "ellipse", "line", "polygon", "place", "square", "path"}
)
_NUM_RE = re.compile(r"(?<![\w.])-?(\d+(?:\.\d+)?)(pt|em|%|mm|cm|in)\b")
_ATTR_NUMBER_RE = re.compile(
    r"\b(?P<key>dx|dy|x|y|width|height|radius|rx|ry|size|stroke|thickness)\s*:\s*(?P<value>-?\d+(?:\.\d+)?)(?P<unit>pt|em|%|mm|cm|in)\b"
)
_NAMED_COLOR_ATTR_RE = re.compile(r"\b(fill|stroke)\s*:\s*([a-z]+)\b")
_RGB_COLOR_ATTR_RE = re.compile(
    r"\b(fill|stroke)\s*:\s*rgb\(\s*\"([0-9a-fA-F]{6})\"\s*\)"
)
_PAGE_RE = re.compile(r"#set\s+page\s*\(")
# Kept for callers that used the old line-level helper. New mutations use
# ``scene_units`` instead, so this is only a compatibility view.
_ELEMENT_LINE_RE = re.compile(
    r"^\s*#(rect|circle|ellipse|line|polygon|place|square|path)\b", re.MULTILINE
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


@dataclass(frozen=True)
class SceneUnit:
    start: int
    end: int
    name: str


def _split_lines(typst_code: str) -> list[str]:
    """Compatibility helper; scene operations no longer use physical lines."""
    return [
        line if line.endswith("\n") else line + "\n"
        for line in typst_code.splitlines(keepends=True)
    ]


def _balanced_end(source: str, start: int, opener: str, closer: str) -> int | None:
    """End of a balanced group, without being confused by quoted delimiters."""
    if start >= len(source) or source[start] != opener:
        return None
    depth, quote, escaped = 0, None, False
    for index in range(start, len(source)):
        char = source[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ('"', "'"):
            quote = char
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def scene_units(typst_code: str) -> list[SceneUnit]:
    """Find complete top-level visual expressions, including #place blocks."""
    units: list[SceneUnit] = []
    pattern = re.compile(r"#(" + "|".join(_VISUAL_NAMES) + r")\b")
    cursor = 0
    while match := pattern.search(typst_code, cursor):
        pos = match.end()
        while pos < len(typst_code) and typst_code[pos].isspace():
            pos += 1
        end = (
            _balanced_end(typst_code, pos, "(", ")")
            if pos < len(typst_code) and typst_code[pos] == "("
            else pos
        )
        if end is None:
            cursor = match.end()
            continue
        probe = end
        while probe < len(typst_code) and typst_code[probe].isspace():
            probe += 1
        if probe < len(typst_code) and typst_code[probe] == "[":
            block_end = _balanced_end(typst_code, probe, "[", "]")
            if block_end is None:
                cursor = end
                continue
            end = block_end
        units.append(SceneUnit(match.start(), end, match.group(1)))
        cursor = end  # Nested elements belong to their enclosing #place.
    return units


def canonicalize_page_setup(typst_code: str, canvas: tuple[int, int]) -> str:
    """Install exactly one fixed page setup, removing all model-provided ones."""
    width, height = canvas
    if width <= 0 or height <= 0:
        return typst_code
    spans = []
    for match in _PAGE_RE.finditer(typst_code):
        end = _balanced_end(
            typst_code, typst_code.find("(", match.start(), match.end()), "(", ")"
        )
        if end is not None:
            spans.append((match.start(), end))
    for start, end in reversed(spans):
        if typst_code[end : end + 1] == "\n":
            end += 1
        typst_code = typst_code[:start] + typst_code[end:]
    return (
        f"#set page(width: {width}pt, height: {height}pt, margin: 0pt)\n"
        + typst_code.lstrip("\n")
    )


def _replace(code: str, unit: SceneUnit, text: str) -> str:
    return code[: unit.start] + text + code[unit.end :]


def _choose(code: str) -> SceneUnit | None:
    units = scene_units(code)
    return random.choice(units) if units else None


def _random_numeric_tweak(typst_code: str) -> str:
    """Tweak only scene numbers, so page dimensions and margin never drift."""
    unit = _choose(typst_code)
    if not unit:
        return typst_code
    text, matches = typst_code[unit.start : unit.end], None
    matches = list(_NUM_RE.finditer(text))
    if not matches:
        return typst_code
    match = random.choice(matches)
    value = max(0.1, float(match.group(1)) * random.uniform(0.7, 1.3))
    return _replace(
        typst_code,
        unit,
        text[: match.start()]
        + f"{value:.2f}".replace(".00", "")
        + match.group(2)
        + text[match.end() :],
    )


def _mutate_position(typst_code: str) -> str:
    unit = _choose(typst_code)
    if not unit:
        return typst_code
    text = typst_code[unit.start : unit.end]
    matches = [
        m
        for m in _ATTR_NUMBER_RE.finditer(text)
        if m.group("key") in {"dx", "dy", "x", "y"}
    ]
    if not matches:
        return typst_code
    m = random.choice(matches)
    value = float(m.group("value")) + random.uniform(-12, 12)
    replacement = f"{m.group('key')}: {value:.2f}{m.group('unit')}".replace(".00", "")
    return _replace(typst_code, unit, text[: m.start()] + replacement + text[m.end() :])


def _mutate_size_or_stroke(typst_code: str) -> str:
    unit = _choose(typst_code)
    if not unit:
        return typst_code
    text = typst_code[unit.start : unit.end]
    matches = [
        m
        for m in _ATTR_NUMBER_RE.finditer(text)
        if m.group("key") not in {"dx", "dy", "x", "y"}
    ]
    if not matches:
        return _random_numeric_tweak(typst_code)
    m = random.choice(matches)
    value = max(0.1, float(m.group("value")) * random.uniform(0.7, 1.3))
    replacement = f"{m.group('key')}: {value:.2f}{m.group('unit')}".replace(".00", "")
    return _replace(typst_code, unit, text[: m.start()] + replacement + text[m.end() :])


def _mutate_color(typst_code: str) -> str:
    unit = _choose(typst_code)
    if not unit:
        return typst_code
    text = typst_code[unit.start : unit.end]
    named, rgb = (
        list(_NAMED_COLOR_ATTR_RE.finditer(text)),
        list(_RGB_COLOR_ATTR_RE.finditer(text)),
    )
    if named:
        m = random.choice(named)
        color = random.choice([c for c in _TYPST_COLORS if c != m.group(2)])
        return _replace(typst_code, unit, text[: m.start(2)] + color + text[m.end(2) :])
    if rgb:
        m = random.choice(rgb)
        values = [int(m.group(2)[i : i + 2], 16) for i in range(0, 6, 2)]
        index = random.randrange(3)
        values[index] = max(0, min(255, values[index] + random.choice((-24, 24))))
        return _replace(
            typst_code,
            unit,
            text[: m.start(2)] + "".join(f"{v:02x}" for v in values) + text[m.end(2) :],
        )
    return typst_code


def _remove_element(typst_code: str) -> str:
    units = scene_units(typst_code)
    if len(units) <= 1:
        return typst_code
    unit = random.choice(units)
    end = unit.end + (typst_code[unit.end : unit.end + 1] == "\n")
    return typst_code[: unit.start] + typst_code[end:]


def _reorder_elements(typst_code: str) -> str:
    units = scene_units(typst_code)
    if len(units) < 2:
        return typst_code
    first, second = sorted(random.sample(units, 2), key=lambda unit: unit.start)
    a, b = typst_code[first.start : first.end], typst_code[second.start : second.end]
    return (
        typst_code[: first.start]
        + b
        + typst_code[first.end : second.start]
        + a
        + typst_code[second.end :]
    )


def _add_element(typst_code: str) -> str:
    unit = _choose(typst_code)
    if not unit:
        return typst_code
    duplicate = typst_code[unit.start : unit.end]
    duplicate = re.sub(
        r"\b(dx|dy)\s*:\s*(-?\d+(?:\.\d+)?)(pt|em|%|mm|cm|in)",
        lambda m: f"{m.group(1)}: {float(m.group(2)) + 8:g}{m.group(3)}",
        duplicate,
        count=1,
    )
    return typst_code[: unit.end] + "\n" + duplicate + typst_code[unit.end :]


MUTATIONS: MutationTable = (
    (_mutate_color, "Mutation: color tweak", 0.22),
    (_mutate_position, "Mutation: position tweak", 0.18),
    (_mutate_size_or_stroke, "Mutation: size/stroke tweak", 0.20),
    (_remove_element, "Mutation: removed element", 0.18),
    (_reorder_elements, "Mutation: reordered elements", 0.12),
    (_add_element, "Mutation: added element", 0.10),
)


def apply_mutation(typst_code: str, operator: str | None = None) -> tuple[str, str]:
    fn, name = pick_operator(MUTATIONS, operator)
    return fn(typst_code), name


def render_typst_png(typst_code: str) -> bytes:
    import typst

    result = typst.compile(typst_code.encode("utf-8"), format="png", ppi=144)
    if isinstance(result, list):
        if not result:
            raise ValueError("Typst generated zero pages.")
        return result[0]
    if isinstance(result, bytes):
        return result
    raise ValueError("Failed to rasterize Typst to PNG bytes.")


def apply_crossover(code_a: str, code_b: str) -> tuple[str, str]:
    """Inject one whole visual expression, never a line fragment or setup."""
    units_a, units_b = scene_units(code_a), scene_units(code_b)
    if not units_a or not units_b:
        return apply_mutation(code_a)
    anchor, donor = random.choice(units_a), random.choice(units_b)
    return code_a[: anchor.end] + "\n" + code_b[donor.start : donor.end] + code_a[
        anchor.end :
    ], "Crossover: scene element injection"
