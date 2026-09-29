import functools
import random
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from typing import cast

from vectrify.svg.ownership import (
    drawable_elements,
)
from vectrify.svg.pathdata import PATH_TOKEN_RE
from vectrify.svg.selection import MutationContext, MutationScope, NoChangeError

# Operators are (function, name, weight); the name is what the engine's policy
# selects by, so it has to be stable across tasks, results and learned state.


MutationTable = tuple[tuple[Callable[[str], str], str, float], ...]


def operator_weights(table: MutationTable) -> Mapping[str, float]:
    """The table as a name -> weight mapping, for a policy to start from."""
    return {name: weight for _fn, name, weight in table}


def pick_operator(
    table: MutationTable, operator: str | None = None
) -> tuple[Callable[[str], str], str]:
    """Resolve *operator* against *table*, falling back to a weighted draw.

    An unknown name draws at random rather than raising: the caller may be a
    policy carrying state from a run of a different format, and losing one
    mutation is cheaper than losing the task.
    """
    for fn, name, _weight in table:
        if name == operator:
            return fn, name
    fns, names, weights = zip(*table, strict=True)
    return random.choices(
        list(zip(fns, names, strict=True)), weights=list(weights), k=1
    )[0]


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

# What an attribute means decides how it should move. A coordinate is a
# position on the canvas, so it moves by an offset: multiplying it ties the
# step to distance from the origin, which freezes elements near the top-left
# and flings far ones across the drawing. A size is a magnitude, where a
# proportional step is right -- a 3px radius and a 300px width should not both
# move by 8px. An opacity lives in [0, 1], where multiplying either does
# nothing at 0.05 or clips at 1.
_POSITION_ATTRS = frozenset({"x", "y", "cx", "cy", "x1", "y1", "x2", "y2"})
_OPACITY_ATTRS = frozenset({"opacity", "fill-opacity", "stroke-opacity"})

_COLOR_ATTRS = frozenset({"fill", "stroke", "color", "stop-color"})

_SHAPE_TAGS = frozenset(
    {"rect", "circle", "ellipse", "line", "path", "polygon", "polyline"}
)


_NUM_RE = re.compile(r"^(-?\d+(?:\.\d+)?)([a-z%]*)$")
# Path data writes numbers with a leading dot and no separators ("M.5.5"), so
# an anchorless \d+ pattern would match the middle of a coordinate pair.
_PATH_NUM_RE = re.compile(r"(-?(?:\d+\.\d+|\.\d+|\d+))")
# Path data as commands and numbers, so an argument can be read in the context
# of the command it belongs to.
_PATH_TOKEN_RE = PATH_TOKEN_RE
_HEX_COLOR_RE = re.compile(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})")
# How far one channel may move in a single mutation.
_COLOR_STEP = 8


def _local_tag(el: ET.Element) -> str:
    return el.tag.split("}")[-1]


def _is_valid_svg(svg: str) -> bool:
    try:
        root = ET.fromstring(svg)
        return root.tag.lower().endswith("svg")
    except ET.ParseError:
        return False


def svg_transform(
    fn: Callable[[ET.Element, MutationContext], None],
) -> Callable[[str], str]:
    """Turn an in-place root-element edit into an SVG-string mutation.

    Handles parsing, namespace registration, and serialization, and returns
    the input unchanged when it does not parse or *fn* finds nothing to do
    (signalled by raising ``NoChangeError``).
    """

    @functools.wraps(fn)
    def wrapper(
        svg: str,
        targets: Mapping[int, float] | None = None,
        scope: MutationScope | None = None,
    ) -> str:
        try:
            root = ET.fromstring(svg)
        except ET.ParseError:
            return svg

        try:
            fn(root, MutationContext(root, targets, scope))
        except NoChangeError:
            return svg
        ET.register_namespace("", SVG_NS)
        return ET.tostring(root, encoding="unicode", method="xml")

    # MutationTable exposes format mutations as single-argument callables.
    # `wrapper` also accepts target weights for the internal dispatcher below,
    # but those are not part of the public mutator contract.
    return cast(Callable[[str], str], wrapper)


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


@svg_transform
def mutate_drop_style_property(root: ET.Element, context: MutationContext) -> None:
    styled = [el for el in root.iter() if el.get("style", "").strip()]
    if not styled:
        raise NoChangeError

    el = context.pick(styled)
    props = [p.strip() for p in el.get("style", "").split(";") if p.strip()]
    if len(props) <= 1:
        raise NoChangeError

    props.pop(random.randrange(len(props)))
    el.set("style", "; ".join(props))


def _is_canvas(el: ET.Element, span: float) -> bool:
    """A rect that covers the whole picture: the page, not part of the drawing.

    Seeds open with one to lay down a background. Moving or resizing it can
    only slide the page out from under the drawing and expose the edge, and one
    real run shipped its winner with the canvas 15px off origin. The four
    measures cannot object -- a white rect on a white ground reads the same
    either way until the gap appears -- so nothing but an LLM edit ever put it
    back, which is a poor use of the one structural edit an epoch gets.
    """
    if el.tag.split("}")[-1] != "rect":
        return False
    try:
        width = abs(float(el.get("width", "0").rstrip("px")))
        height = abs(float(el.get("height", "0").rstrip("px")))
    except ValueError:
        return False
    return width >= span * 0.99 and height >= span * 0.99


def movable_elements(root: ET.Element) -> list[tuple[tuple, ET.Element]]:
    """Drawable elements a geometry mutation may touch."""
    span = _canvas_span(root)
    return [
        (chain, el) for chain, el in drawable_elements(root) if not _is_canvas(el, span)
    ]


@svg_transform
def mutate_numeric(root: ET.Element, context: MutationContext) -> None:
    span = _canvas_span(root)
    candidates: list[tuple[ET.Element, str, float, str]] = []
    for elem in root.iter():
        if _is_canvas(elem, span):
            continue
        for attr, val in elem.attrib.items():
            bare_attr = attr.split("}")[-1]
            if bare_attr not in _NUMERIC_ATTRS:
                continue
            if not context.allows_attribute(bare_attr):
                continue
            m = _NUM_RE.match(val.strip())
            if m:
                candidates.append((elem, attr, float(m.group(1)), m.group(2)))

    if not candidates:
        raise NoChangeError

    elem, attr, num, unit = context.pick(candidates)
    bare = attr.split("}")[-1]

    if bare in _OPACITY_ATTRS:
        # Additive and never rounded to an integer: opacity="1" would otherwise
        # only ever be 1 or 0.
        new_num = max(0.0, min(1.0, num + random.uniform(-0.25, 0.25)))
        formatted = f"{new_num:.3f}".rstrip("0").rstrip(".") or "0"
        elem.attrib[attr] = f"{formatted}{unit}"
        return

    if bare in _POSITION_ATTRS:
        step = max(2.0, _canvas_span(root) * 0.05)
        new_num = num + random.uniform(-step, step)
        grew = new_num > num
    else:
        factor = random.uniform(0.7, 1.3)
        new_num = num * factor
        grew = factor >= 1.0

    if not unit and num == int(num) and new_num >= 0:
        rounded = round(new_num)
        if rounded == num:
            # A step that rounds back to where it started cannot move a small
            # integer at all: rx="1" maps to 1 for every factor in the range,
            # so a rounded corner that drifted down to 1 stays square forever,
            # and 0 is worse still. Nudge by one in the direction chosen, so
            # small values are as free to grow as to shrink.
            rounded = max(0, int(num) + (1 if grew else -1))
        elem.attrib[attr] = str(rounded)
        return

    formatted = f"{new_num:.3f}".rstrip("0").rstrip(".")
    elem.attrib[attr] = f"{formatted}{unit}"


@svg_transform
def mutate_color(root: ET.Element, context: MutationContext) -> None:
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
        raise NoChangeError

    # Only a hex value can be fudged. A name has no channels, so "changing" it
    # means jumping to some other colour, which is not a search step -- it is
    # how stray blues and browns arrived in drawings that are otherwise grey.
    candidates = [c for c in candidates if _HEX_COLOR_RE.search(c[3])]
    if not candidates:
        raise NoChangeError

    elem, source, key, val = context.pick(candidates)

    h = _HEX_COLOR_RE.search(val).group(1)  # type: ignore[union-attr]
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    # A small step. At +/-60 a grey wanders into brown in a handful of accepted
    # mutations -- one run shipped #392800 on a drawing whose palette is black,
    # white and two greys -- and the drift is invisible per step, so nothing
    # rejects it until the colour is plainly wrong.
    r = max(0, min(255, int(h[0:2], 16) + random.randint(-_COLOR_STEP, _COLOR_STEP)))
    g = max(0, min(255, int(h[2:4], 16) + random.randint(-_COLOR_STEP, _COLOR_STEP)))
    b = max(0, min(255, int(h[4:6], 16) + random.randint(-_COLOR_STEP, _COLOR_STEP)))
    new_color = f"#{r:02x}{g:02x}{b:02x}"

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
def mutate_stroke(root: ET.Element, context: MutationContext) -> None:
    """Add, remove, or change stroke on a random shape element."""
    shapes = [el for el in root.iter() if _local_tag(el) in _SHAPE_TAGS]
    if not shapes:
        raise NoChangeError

    el = context.pick(shapes)
    has_stroke = el.get("stroke") not in (None, "none", "")

    op = random.choice(["add", "remove"])
    if op == "remove" and has_stroke:
        el.set("stroke", "none")
    elif op == "add" and not has_stroke:
        # The element's own fill, not a colour from anywhere else: an outline
        # belongs to the shape it outlines, and inventing one is what put a
        # blue ring on a black dot. Colour itself is mutate_color's business.
        fill = el.get("fill")
        if not fill or fill in ("none", "inherit", "transparent"):
            raise NoChangeError
        el.set("stroke", fill)
        if not el.get("stroke-width"):
            el.set("stroke-width", str(random.choice([1, 2, 3])))
    else:
        raise NoChangeError


def _nudgeable_numbers(d: str) -> list[re.Match]:
    """Numbers in path data that may be nudged, skipping elliptical-arc flags.

    ``A rx ry rotation large-arc sweep x y`` carries two booleans in the middle.
    Nudging one writes ``sweep="-1.8"``, which is not path data: a renderer
    either coerces it or drops the whole path, and the element vanishes from
    the drawing. Measured before this existed, 26% of mutations on a path
    holding two arcs corrupted a flag.

    The rotation is skipped too, but only where ``rx == ry``. Rotating a circle
    about its own centre is the identity, so nudging it spends a task on a
    candidate that renders identically to its parent, and leaves a number that
    reads as meaningful to everything downstream. Where the arc is genuinely
    elliptical the rotation does turn it, so it stays nudgeable there.
    """
    out: list[re.Match] = []
    command = ""
    argument = 0
    radii: list[float] = [0.0, 0.0]
    for token in _PATH_TOKEN_RE.finditer(d):
        if token.group(1):
            command = token.group(1)
            argument = 0
            continue
        # Arcs take seven arguments and may repeat without restating the
        # command, so the flags are found by position within each group.
        if command in ("A", "a"):
            position = argument % 7
            if position in (0, 1):
                try:
                    radii[position] = float(token.group(2))
                except ValueError:
                    radii[position] = 0.0
            circular = abs(radii[0] - radii[1]) < 1e-9
            if position in (3, 4) or (position == 2 and circular):
                argument += 1
                continue
        out.append(token)
        argument += 1
    return out


@svg_transform
def mutate_path(root: ET.Element, context: MutationContext) -> None:
    """Nudge one numeric coordinate in a path 'd' attribute."""
    paths = [el for el in root.iter() if el.get("d")]
    if not paths:
        raise NoChangeError

    el = context.pick(paths)
    d = el.get("d", "")
    nums = _nudgeable_numbers(d)
    if not nums:
        raise NoChangeError

    m = random.choice(nums)
    val = float(m.group(0))
    # An offset rather than a percentage: path numbers are overwhelmingly
    # coordinates, and scaling one ties the step to distance from the origin,
    # so the same nudge barely stirs a point near the top-left and throws one
    # at the far edge across the drawing.
    magnitude = max(2.0, _canvas_span(root) * 0.03)
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


def _canvas_span(root: ET.Element) -> float:
    """Rough canvas size, for sizing a move in the drawing's own units."""
    box = root.get("viewBox", "").replace(",", " ").split()
    if len(box) == 4:
        try:
            return max(abs(float(box[2])), abs(float(box[3]))) or 100.0
        except ValueError:
            pass
    for attr in ("width", "height"):
        m = _NUM_RE.match(root.get(attr, "").strip())
        if m:
            return abs(float(m.group(1))) or 100.0
    return 100.0


# Which arguments of a path command are x coordinates and which are y, by
# position within the command's argument group. A translate has to move
# coordinates and leave everything else alone: an arc's radii and rotation are
# not positions, and shifting them reshapes the curve instead of moving it.
_PATH_COORDS: dict[str, tuple[int, tuple[int, ...], tuple[int, ...]]] = {
    # command: (arity, x argument indices, y argument indices)
    "M": (2, (0,), (1,)),
    "L": (2, (0,), (1,)),
    "T": (2, (0,), (1,)),
    "H": (1, (0,), ()),
    "V": (1, (), (0,)),
    "C": (6, (0, 2, 4), (1, 3, 5)),
    "S": (4, (0, 2), (1, 3)),
    "Q": (4, (0, 2), (1, 3)),
    "A": (7, (5,), (6,)),
}


def _translate_path(d: str, dx: float, dy: float) -> str:
    """Shift every absolute coordinate in path data by (dx, dy).

    Relative commands describe offsets from wherever the pen already is, so
    moving them would change the shape rather than its position -- with one
    exception: a path opening with a relative moveto starts from the origin, so
    that first pair is a position like any other.
    """
    out: list[str] = []
    command = ""
    argument = 0
    seen_move = False
    last = 0
    for token in _PATH_TOKEN_RE.finditer(d):
        out.append(d[last : token.start()])
        last = token.end()
        text = token.group(0)
        if token.group(1):
            command = text
            argument = 0
            out.append(text)
            continue

        layout = _PATH_COORDS.get(command.upper())
        shift = 0.0
        if layout is not None:
            arity, xs, ys = layout
            slot = argument % arity
            absolute = command.isupper() or (not seen_move and command == "m")
            if absolute:
                shift = dx if slot in xs else dy if slot in ys else 0.0
        if command.upper() == "M":
            seen_move = True

        if shift:
            moved = float(text) + shift
            out.append(f"{moved:.1f}".rstrip("0").rstrip(".") or "0")
        else:
            out.append(text)
        argument += 1
    out.append(d[last:])
    return "".join(out)


def _shift_element(el: ET.Element, dx: float, dy: float) -> bool:
    """Move one element by editing its own numbers. True if anything moved."""
    moved = False
    for attr in ("x", "cx", "x1", "x2"):
        raw = el.get(attr)
        if raw is not None:
            m = _NUM_RE.match(raw.strip())
            if m:
                el.set(attr, f"{float(m.group(1)) + dx:g}{m.group(2)}")
                moved = True
    for attr in ("y", "cy", "y1", "y2"):
        raw = el.get(attr)
        if raw is not None:
            m = _NUM_RE.match(raw.strip())
            if m:
                el.set(attr, f"{float(m.group(1)) + dy:g}{m.group(2)}")
                moved = True

    points = el.get("points")
    if points:
        nums = _PATH_NUM_RE.findall(points)
        if len(nums) >= 2:
            shifted = [
                f"{float(n) + (dx if i % 2 == 0 else dy):g}" for i, n in enumerate(nums)
            ]
            el.set(
                "points",
                " ".join(
                    f"{shifted[i]},{shifted[i + 1]}"
                    for i in range(0, len(shifted) - 1, 2)
                ),
            )
            moved = True

    d = el.get("d")
    if d:
        el.set("d", _translate_path(d, dx, dy))
        moved = True
    return moved


@svg_transform
def mutate_translate(root: ET.Element, context: MutationContext) -> None:
    """Move one element along both axes at once.

    The other numeric operator scales a single attribute, which cannot express
    a move. Scaling a coordinate makes displacement depend on distance from the
    origin -- a dot at cx=300 jumps ten times as far as one at cx=30 for the
    same factor -- and changing one axis at a time means a diagonally displaced
    element has to be accepted twice to arrive, when the half-way state is
    often no better than where it started. That is worst where elements are
    many and small, each too slight for its own move to show in the score.

    Written by editing the element's own numbers rather than by adding a
    transform. A transform is a second description of where a thing is, laid
    over the first: it accumulates -- one real run stacked 23 of them on its
    background rect and walked the canvas off its own viewBox -- and it leaves
    the coordinates saying one thing while the drawing does another, so every
    later mutation reads a position that is not where the element appears.
    """
    units = movable_elements(root)
    if not units:
        raise NoChangeError

    element = context.pick([el for _chain, el in units])
    step = max(2.0, _canvas_span(root) * 0.05)
    dx = random.uniform(-step, step)
    dy = random.uniform(-step, step)

    # A group has no coordinates of its own, so the move goes to everything
    # inside it -- which is what moving a group means.
    moved = False
    for el in element.iter():
        moved = _shift_element(el, dx, dy) or moved
    if not moved:
        raise NoChangeError


@svg_transform
def mutate_reorder(root: ET.Element, context: MutationContext) -> None:
    """Swap two adjacent sibling elements to change z-order."""
    # Both siblings must be editable: swapping moves each of them.
    pairs = [
        (parent, i)
        for parent in root.iter()
        for i, (a, b) in enumerate(zip(parent, list(parent)[1:], strict=False))
        if context.editable(a) and context.editable(b)
    ]
    if not pairs:
        raise NoChangeError

    parent, i = random.choice(pairs)
    children = list(parent)
    children[i], children[i + 1] = children[i + 1], children[i]
    for child in list(parent):
        parent.remove(child)
    for child in children:
        parent.append(child)


# The edit kinds each operator needs, checked against a MutationScope.
OPERATOR_KINDS: dict[str, frozenset[str]] = {
    "Mutation: color tweak": frozenset({"paint"}),
    "Mutation: numeric tweak": frozenset({"geometry"}),
    "Mutation: moved element": frozenset({"geometry"}),
    "Mutation: path nudge": frozenset({"geometry"}),
    "Mutation: stroke change": frozenset({"paint"}),
    "Mutation: reordered elements": frozenset({"structure"}),
    "Mutation: dropped style property": frozenset({"paint"}),
}

# Default weights, used when no policy names an operator.
MUTATIONS: MutationTable = (
    (mutate_color, "Mutation: color tweak", 0.25),
    (mutate_numeric, "Mutation: numeric tweak", 0.20),
    (mutate_translate, "Mutation: moved element", 0.15),
    (mutate_path, "Mutation: path nudge", 0.15),
    (mutate_stroke, "Mutation: stroke change", 0.10),
    (mutate_reorder, "Mutation: reordered elements", 0.10),
    (mutate_drop_style_property, "Mutation: dropped style property", 0.05),
)


def scoped_mutations(scope: MutationScope | None) -> MutationTable:
    """The operators whose edit kinds the scope permits."""
    if scope is None:
        return MUTATIONS
    return tuple(m for m in MUTATIONS if OPERATOR_KINDS[m[1]] <= scope.kinds)


def apply_mutation(
    parent_svg: str,
    operator: str | None = None,
    targets: dict[int, float] | None = None,
    scope: MutationScope | None = None,
) -> tuple[str, str]:
    """Apply *operator* to *parent_svg*, or a weighted-random one if None.

    *targets* weights elements by the error they answer for, keyed by their
    index among the drawable elements in document order. With a *scope*, only
    its elements change and only permitted operators run; an operator the scope
    forbids leaves the parent unchanged, which the worker reports as no change.
    """
    table = scoped_mutations(scope)
    if not table or (
        operator is not None
        and operator in OPERATOR_KINDS
        and all(name != operator for _fn, name, _w in table)
    ):
        return parent_svg, operator or MUTATIONS[0][1]
    fn, name = pick_operator(table, operator)

    def run() -> str:
        targeted_fn = cast(
            Callable[[str, Mapping[int, float] | None, MutationScope | None], str],
            fn,
        )
        return targeted_fn(parent_svg, targets, scope)

    return with_retries(run, fallback=parent_svg), name


def mutation_weights(scope: MutationScope | None = None) -> Mapping[str, float]:
    """Starting operator weights for a policy, limited to what *scope* allows."""
    return operator_weights(scoped_mutations(scope))
