import copy
import functools
import random
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable

import numpy as np

from vectrify.formats.mutations import MutationTable, pick_operator
from vectrify.formats.svg.ownership import (
    drawable_elements,
    overlaps,
    owner_labels,
)

SVG_NS = "http://www.w3.org/2000/svg"

_NUMERIC_ATTRS = frozenset(
    {
        "width",
        "height",
        "x",
        "y",
        "x1",
        "y1",
        "x2",
        "y2",
        "cx",
        "cy",
        "r",
        "rx",
        "ry",
        "font-size",
        "stroke-width",
        "opacity",
        "fill-opacity",
        "stroke-opacity",
    }
)

_COLOR_ATTRS = frozenset({"fill", "stroke", "color", "stop-color"})

_SHAPE_TAGS = frozenset(
    {"rect", "circle", "ellipse", "line", "path", "polygon", "polyline"}
)

_NAMED_SVG_COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "cyan",
    "magenta",
    "pink",
    "brown",
    "black",
    "white",
    "gray",
    "navy",
    "teal",
    "olive",
    "coral",
    "salmon",
    "gold",
    "indigo",
    "lime",
    "aqua",
    "maroon",
    "silver",
    "crimson",
    "turquoise",
]

_NUM_RE = re.compile(r"^(-?\d+(?:\.\d+)?)([a-z%]*)$")
# Path data writes numbers with a leading dot and no separators ("M.5.5"), so
# an anchorless \d+ pattern would match the middle of a coordinate pair.
_PATH_NUM_RE = re.compile(r"(-?(?:\d+\.\d+|\.\d+|\d+))")
_HEX_COLOR_RE = re.compile(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})")


def _local_tag(el: ET.Element) -> str:
    return el.tag.split("}")[-1]


def _is_valid_svg(svg: str) -> bool:
    try:
        root = ET.fromstring(svg)
        return root.tag.lower().endswith("svg")
    except ET.ParseError:
        return False


class _NoChangeError(Exception):
    """Raised by an operator when there is nothing it can mutate."""


# How much error each element answers for, by its position among the drawable
# elements. A module-level channel rather than an argument because every
# operator picks its own target in its own way, and threading a table through
# all of them would touch each one twice.
#
# Keyed by position, not identity: every operator re-parses the SVG, so the
# element objects a caller could point at are not the ones being mutated.
_TARGET_BY_INDEX: dict[int, float] = {}

# Resolved against the parse the operator is actually working on.
_TARGET_WEIGHTS: dict[int, float] = {}


def _pick(candidates: list):
    """Choose an element to mutate, favouring the ones answering for the error.

    Uniform choice spends most of a run on elements that are already right. On
    an emblem seed the background answers for 57% of the error and the star for
    3.5%, and picking uniformly gives them equal attention forever.
    """
    if not candidates:
        raise _NoChangeError
    if not _TARGET_WEIGHTS:
        return random.choice(candidates)

    weights = [_TARGET_WEIGHTS.get(id(_element_of(item)), 0.0) for item in candidates]
    total = sum(weights)
    if total <= 0.0:
        return random.choice(candidates)
    # A floor under every candidate: attributed error says where the error is
    # now, not where a fix is available, so nothing should be unreachable.
    floor = total * _TARGET_FLOOR / len(candidates)
    return random.choices(candidates, weights=[w + floor for w in weights], k=1)[0]


def _element_of(item):
    """Operators offer elements, or tuples with the element first or second."""
    if isinstance(item, tuple):
        for part in item:
            if isinstance(part, ET.Element):
                return part
        return item[0]
    return item


# Share of the selection mass spread evenly over every candidate.
_TARGET_FLOOR = 0.25


def svg_transform(fn: Callable[[ET.Element], None]) -> Callable[[str], str]:
    """Turn an in-place root-element edit into an SVG-string mutation.

    Handles parsing, namespace registration, and serialization, and returns
    the input unchanged when it does not parse or *fn* finds nothing to do
    (signalled by raising _NoChangeError).
    """

    @functools.wraps(fn)
    def wrapper(svg: str) -> str:
        try:
            root = ET.fromstring(svg)
        except ET.ParseError:
            return svg

        _TARGET_WEIGHTS.clear()
        if _TARGET_BY_INDEX:
            for index, (_chain, element) in enumerate(drawable_elements(root)):
                _TARGET_WEIGHTS[id(element)] = _TARGET_BY_INDEX.get(index, 0.0)
        try:
            fn(root)
        except _NoChangeError:
            return svg
        finally:
            _TARGET_WEIGHTS.clear()
        ET.register_namespace("", SVG_NS)
        return ET.tostring(root, encoding="unicode", method="xml")

    return wrapper


def with_retries(
    op: Callable[[], str],
    fallback: str,
    max_retries: int = 3,
) -> str:
    for _ in range(max_retries):
        try:
            result = op()
            if _is_valid_svg(result):
                return result
        except Exception:
            pass
    return fallback


# How much two elements must overlap to count as the same thing. Below this
# they are separate features that happen to sit near each other, or one is a
# piece of the other: half a blade covers 0.5 of the whole blade, and swapping
# the whole for the half is how a picture loses content. It also carries the
# safety the old whole-document correspondence gate used to provide, at a
# fraction of its cost -- that gate demanded 80% of both parents find partners,
# which no seed pair in the corpus managed on four of six cases, so crossover
# never ran there at all.
_MATCH_OVERLAP = 0.7


def _rebuild(root: ET.Element, units: list[tuple[tuple, ET.Element]]) -> str:
    """Reassemble elements into A's structure, recreating each group once."""
    new_root = ET.Element(root.tag, dict(root.attrib))
    wrappers: dict[tuple, ET.Element] = {(): new_root}

    for chain, element in units:
        parent = new_root
        path: tuple = ()
        for step in chain:
            path = (*path, step)
            if path not in wrappers:
                wrappers[path] = ET.SubElement(parent, step[0], dict(step[1]))
            parent = wrappers[path]
        parent.append(copy.deepcopy(element))

    ET.register_namespace("", SVG_NS)
    return ET.tostring(new_root, encoding="unicode", method="xml")


def crossover(svg_a: str, svg_b: str) -> str:
    """Swap elements that draw the same thing between two parents.

    Recombination needs to know which element of A corresponds to which element
    of B, and every cheap answer is wrong here. Document order assumes the
    parents list their elements in the same sequence, which independent
    drawings do not -- splicing by index left children holding a fraction of
    the elements they started with. Element type assumes they encode things
    the same way, but a dot is a <circle> in one lineage and a <path> in
    another. Canvas region assumes a geometry that has to be guessed, and
    guessing it wrongly empties whole drawings.

    What the parents do share is the picture, so elements are matched on the
    pixels they own in the finished render -- occlusion included, so a ring
    matches a ring by its visible annulus. A matched pair contributes exactly
    one element and an unmatched one is carried through untouched, which makes
    losing content impossible: the child has as many elements as A had.
    """
    try:
        root_a = ET.fromstring(svg_a)
        root_b = ET.fromstring(svg_b)
    except ET.ParseError:
        return svg_a

    units_a = drawable_elements(root_a)
    units_b = drawable_elements(root_b)
    if not units_a or not units_b:
        return svg_a

    labels_a = owner_labels(root_a)
    labels_b = owner_labels(root_b)
    scores = overlaps(labels_a, labels_b, len(units_a), len(units_b))

    # Best pairs first, each element used once: an optimal assignment costs
    # more than it buys when most pairs are either obvious or unrelated.
    order = np.argsort(scores, axis=None)[::-1]
    taken_a: set[int] = set()
    taken_b: set[int] = set()
    swapped: dict[int, int] = {}
    for flat in order:
        i, j = divmod(int(flat), len(units_b))
        if scores[i, j] < _MATCH_OVERLAP:
            break
        if i in taken_a or j in taken_b:
            continue
        taken_a.add(i)
        taken_b.add(j)
        if random.random() < 0.5:
            swapped[i] = j

    if not swapped:
        return svg_a

    kept = [
        (chain, units_b[swapped[i]][1] if i in swapped else element)
        for i, (chain, element) in enumerate(units_a)
    ]
    return _rebuild(root_a, kept)


@svg_transform
def mutate_drop_style_property(root: ET.Element) -> None:
    styled = [el for el in root.iter() if el.get("style", "").strip()]
    if not styled:
        raise _NoChangeError

    el = _pick(styled)
    props = [p.strip() for p in el.get("style", "").split(";") if p.strip()]
    if len(props) <= 1:
        raise _NoChangeError

    props.pop(random.randrange(len(props)))
    el.set("style", "; ".join(props))


@svg_transform
def mutate_numeric(root: ET.Element) -> None:
    candidates: list[tuple[ET.Element, str, float, str]] = []
    for elem in root.iter():
        for attr, val in elem.attrib.items():
            bare_attr = attr.split("}")[-1]
            if bare_attr not in _NUMERIC_ATTRS:
                continue
            m = _NUM_RE.match(val.strip())
            if m:
                candidates.append((elem, attr, float(m.group(1)), m.group(2)))

    if not candidates:
        raise _NoChangeError

    elem, attr, num, unit = _pick(candidates)
    factor = random.uniform(0.7, 1.3)
    new_num = num * factor

    if "opacity" in attr:
        new_num = max(0.0, min(1.0, new_num))

    if not unit and num == int(num) and new_num >= 0:
        elem.attrib[attr] = str(round(new_num))
    else:
        formatted = f"{new_num:.3f}".rstrip("0").rstrip(".")
        elem.attrib[attr] = f"{formatted}{unit}"


@svg_transform
def mutate_color(root: ET.Element) -> None:
    """Tweak a fill or stroke color — nudge hex channels or swap named color."""
    # Collect (elem, source, key, current_value) for every color reference
    candidates: list[tuple[ET.Element, str, str, str]] = []
    for elem in root.iter():
        for attr in list(elem.attrib):
            bare = attr.split("}")[-1]
            if bare in _COLOR_ATTRS:
                val = elem.attrib[attr]
                if val and val not in ("none", "inherit", "transparent"):
                    candidates.append((elem, "attr", attr, val))
        style = elem.get("style", "")
        for prop in style.split(";"):
            prop = prop.strip()
            if ":" not in prop:
                continue
            k, v = prop.split(":", 1)
            k, v = k.strip(), v.strip()
            if k in _COLOR_ATTRS and v not in ("none", "inherit", "transparent"):
                candidates.append((elem, "style", k, v))

    if not candidates:
        raise _NoChangeError

    elem, source, key, val = _pick(candidates)

    hex_match = _HEX_COLOR_RE.search(val)
    if hex_match:
        h = hex_match.group(1)
        if len(h) == 3:
            h = h[0] * 2 + h[1] * 2 + h[2] * 2
        r = max(0, min(255, int(h[0:2], 16) + random.randint(-60, 60)))
        g = max(0, min(255, int(h[2:4], 16) + random.randint(-60, 60)))
        b = max(0, min(255, int(h[4:6], 16) + random.randint(-60, 60)))
        new_color = f"#{r:02x}{g:02x}{b:02x}"
    else:
        new_color = random.choice(_NAMED_SVG_COLORS)

    if source == "attr":
        elem.set(key, new_color)
    else:
        props: dict[str, str] = {}
        for prop in elem.get("style", "").split(";"):
            prop = prop.strip()
            if ":" in prop:
                pk, pv = prop.split(":", 1)
                props[pk.strip()] = pv.strip()
        props[key] = new_color
        elem.set("style", "; ".join(f"{pk}:{pv}" for pk, pv in props.items()))


@svg_transform
def mutate_stroke(root: ET.Element) -> None:
    """Add, remove, or change stroke on a random shape element."""
    shapes = [el for el in root.iter() if _local_tag(el) in _SHAPE_TAGS]
    if not shapes:
        raise _NoChangeError

    el = _pick(shapes)
    has_stroke = el.get("stroke") not in (None, "none", "")

    op = random.choice(["add", "remove", "change"])
    if op == "remove" and has_stroke:
        el.set("stroke", "none")
    elif op in ("add", "change"):
        el.set("stroke", random.choice(_NAMED_SVG_COLORS))
        if not el.get("stroke-width"):
            el.set("stroke-width", str(random.choice([1, 2, 3])))


@svg_transform
def mutate_path(root: ET.Element) -> None:
    """Nudge one numeric coordinate in a path 'd' attribute."""
    paths = [el for el in root.iter() if el.get("d")]
    if not paths:
        raise _NoChangeError

    el = _pick(paths)
    d = el.get("d", "")
    nums = list(_PATH_NUM_RE.finditer(d))
    if not nums:
        raise _NoChangeError

    m = random.choice(nums)
    val = float(m.group(1))
    # Use a proportional nudge (±15%) with a minimum of ±2px
    magnitude = max(2.0, abs(val) * 0.15)
    new_val = val + random.uniform(-magnitude, magnitude)
    new_str = f"{new_val:.1f}".rstrip("0").rstrip(".")
    # Neighbouring numbers may abut the replaced one; without a separator the
    # new text would merge with them into a different number, silently shifting
    # every coordinate that follows.
    before, after = d[: m.start()], d[m.end() :]
    if before and (before[-1].isdigit() or before[-1] == "."):
        before += " "
    if after and (after[0].isdigit() or after[0] == "."):
        after = " " + after
    el.set("d", before + new_str + after)


@svg_transform
def mutate_reorder(root: ET.Element) -> None:
    """Swap two adjacent sibling elements to change z-order."""
    candidates = [el for el in root.iter() if len(list(el)) >= 2]
    if not candidates:
        raise _NoChangeError

    parent = random.choice(candidates)
    children = list(parent)
    i = random.randrange(len(children) - 1)
    children[i], children[i + 1] = children[i + 1], children[i]
    for child in list(parent):
        parent.remove(child)
    for child in children:
        parent.append(child)


# Default weights, used when no policy names an operator.
MUTATIONS: MutationTable = (
    (mutate_color, "Mutation: color tweak", 0.25),
    (mutate_numeric, "Mutation: numeric tweak", 0.20),
    (mutate_path, "Mutation: path nudge", 0.15),
    (mutate_stroke, "Mutation: stroke change", 0.10),
    (mutate_reorder, "Mutation: reordered elements", 0.10),
    (mutate_drop_style_property, "Mutation: dropped style property", 0.05),
)


def apply_mutation(
    parent_svg: str,
    operator: str | None = None,
    targets: dict[int, float] | None = None,
) -> tuple[str, str]:
    """Apply *operator* to *parent_svg*, or a weighted-random one if None.

    *targets* weights elements by the error they answer for, keyed by their
    index among the drawable elements in document order.
    """
    fn, name = pick_operator(MUTATIONS, operator)

    def run() -> str:
        _TARGET_BY_INDEX.clear()
        _TARGET_BY_INDEX.update(targets or {})
        try:
            return fn(parent_svg)
        finally:
            _TARGET_BY_INDEX.clear()

    return with_retries(run, fallback=parent_svg), name


def apply_crossover(svg_a: str, svg_b: str) -> tuple[str, str]:
    return (
        with_retries(lambda: crossover(svg_a, svg_b), fallback=svg_a),
        "Local crossover",
    )
