"""Put LLM markup into the form local search can actually edit.

The prompt asks for mutable markup and mostly gets it, but which forms a model
reaches for is a property of the model, not of the task: one model's seeds
carried 147 elements written in relative path commands, and a relative command
describes an offset from wherever the pen already is, so nothing that moves an
element can touch it. Asking more firmly does not fix that; rewriting does.

Every rewrite here preserves the rendering exactly. The point is to remove the
ways a drawing can be correct on screen and unreachable to mutation.
"""

import re
from xml.etree import ElementTree as ET

# Enough of the CSS names to cover what a model actually reaches for. A name
# has no channels to fudge, so mutate_color skips it and the colour is frozen
# for the rest of the run.
_NAMED_TO_HEX = {
    "black": "#000000",
    "white": "#ffffff",
    "red": "#ff0000",
    "green": "#008000",
    "blue": "#0000ff",
    "yellow": "#ffff00",
    "orange": "#ffa500",
    "purple": "#800080",
    "gray": "#808080",
    "grey": "#808080",
    "silver": "#c0c0c0",
    "maroon": "#800000",
    "navy": "#000080",
    "olive": "#808000",
    "teal": "#008080",
    "lime": "#00ff00",
    "aqua": "#00ffff",
    "cyan": "#00ffff",
    "magenta": "#ff00ff",
    "fuchsia": "#ff00ff",
    "pink": "#ffc0cb",
    "brown": "#a52a2a",
    "gold": "#ffd700",
    "indigo": "#4b0082",
    "coral": "#ff7f50",
    "salmon": "#fa8072",
    "crimson": "#dc143c",
    "turquoise": "#40e0d0",
}

_COLOR_ATTRS = ("fill", "stroke", "color", "stop-color")
_PRESENTATION = frozenset(
    {
        *_COLOR_ATTRS,
        "opacity",
        "fill-opacity",
        "stroke-opacity",
        "stroke-width",
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-dasharray",
        "font-size",
        "font-family",
    }
)

_TOKEN_RE = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|(-?(?:\d+\.\d+|\.\d+|\d+))")

# arity, indices that are x, indices that are y
_LAYOUT: dict[str, tuple[int, tuple[int, ...], tuple[int, ...]]] = {
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


def _fmt(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".") or "0"


def absolutize_path(d: str) -> str:
    """Rewrite path data so every coordinate is absolute.

    A relative command says "from wherever you are", which is not a position
    anything can move, scale or reason about. Absolute coordinates are the same
    picture written so that a mutation can reach it.
    """
    tokens = [(m.group(1), m.group(2)) for m in _TOKEN_RE.finditer(d)]
    out: list[str] = []
    x = y = 0.0
    start_x = start_y = 0.0
    command = ""
    args: list[float] = []

    def flush() -> None:
        nonlocal x, y, start_x, start_y, args
        if not command:
            return
        upper = command.upper()
        relative = command.islower()
        if upper == "Z":
            out.append("Z")
            x, y = start_x, start_y
            return
        arity, xs, ys = _LAYOUT[upper]
        for i in range(0, len(args) - arity + 1, arity):
            group = list(args[i : i + arity])
            if relative:
                for j in range(arity):
                    if j in xs:
                        group[j] += x
                    elif j in ys:
                        group[j] += y
            if upper == "H":
                nx, ny = group[0], y
            elif upper == "V":
                nx, ny = x, group[0]
            else:
                nx, ny = group[xs[-1]], group[ys[-1]]
            out.append(upper + " " + " ".join(_fmt(v) for v in group))
            if upper == "M" and i == 0:
                start_x, start_y = nx, ny
            x, y = nx, ny
        args = []

    for cmd, num in tokens:
        if cmd:
            flush()
            command = cmd
            args = []
            if cmd.upper() == "Z":
                flush()
                command = ""
        else:
            args.append(float(num))
    flush()
    return " ".join(out)


def _expand_style(el: ET.Element) -> None:
    """Move presentation properties out of style= and onto the element.

    A mutation reads attributes. A colour or a width parked in a style string is
    invisible to most of them, so the drawing has parts that cannot be tuned.
    """
    style = el.get("style")
    if not style:
        return
    leftover: list[str] = []
    for prop in style.split(";"):
        if ":" not in prop:
            continue
        key, value = (part.strip() for part in prop.split(":", 1))
        if key in _PRESENTATION and value:
            el.set(key, value)
        elif key:
            leftover.append(f"{key}:{value}")
    if leftover:
        el.set("style", ";".join(leftover))
    else:
        del el.attrib["style"]


def _normalize_colors(el: ET.Element) -> None:
    for attr in _COLOR_ATTRS:
        value = el.get(attr)
        if not value:
            continue
        lowered = value.strip().lower()
        if lowered in _NAMED_TO_HEX:
            el.set(attr, _NAMED_TO_HEX[lowered])
        elif re.fullmatch(r"#[0-9a-fA-F]{3}", lowered):
            el.set(attr, "#" + "".join(c * 2 for c in lowered[1:]))


def _points_to_path(el: ET.Element) -> bool:
    """Rewrite polygon and polyline as path. True if the element changed.

    Their geometry lives in `points`, which the path operators cannot see, so a
    shape written this way is fixed in place for the whole run.
    """
    tag = el.tag.split("}")[-1]
    if tag not in ("polygon", "polyline"):
        return False
    nums = re.findall(r"-?(?:\d+\.\d+|\.\d+|\d+)", el.get("points", ""))
    if len(nums) < 4:
        return False
    pairs = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
    d = "M " + " L ".join(f"{px} {py}" for px, py in pairs)
    if tag == "polygon":
        d += " Z"
    el.tag = el.tag.replace(tag, "path")
    del el.attrib["points"]
    el.set("d", d)
    return True


def normalize_svg(svg: str) -> str:
    """Rewrite *svg* into the most mutable form of the same picture.

    Returns the input unchanged if it does not parse -- validation is a separate
    step and reports its own error.
    """
    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        return svg

    for el in root.iter():
        _expand_style(el)
        _normalize_colors(el)
        _points_to_path(el)
        d = el.get("d")
        if d:
            absolute = absolutize_path(d)
            if absolute:
                el.set("d", absolute)

    ET.register_namespace("", "http://www.w3.org/2000/svg")
    return ET.tostring(root, encoding="unicode", method="xml")
