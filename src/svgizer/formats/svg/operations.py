import copy
import io
import random
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable

from PIL import Image

from svgizer.image_utils import rasterize_svg_to_png_bytes
from svgizer.score.utils import lab_l1
from svgizer.search.models import INVALID_SCORE

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
_PATH_NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")
_HEX_COLOR_RE = re.compile(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})")


def _local_tag(el: ET.Element) -> str:
    return el.tag.split("}")[-1]


def _is_valid_svg(svg: str) -> bool:
    try:
        root = ET.fromstring(svg)
        return root.tag.lower().endswith("svg")
    except ET.ParseError:
        return False


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


def with_micro_search(
    op_generator: Callable[[], tuple[str, str]],
    fallback_svg: str,
    orig_img_fast: Image.Image,
    num_trials: int = 15,
    default_summary: str = "No change",
) -> tuple[str, str]:
    best_svg: str | None = None
    best_fast_score = INVALID_SCORE
    best_summary = default_summary
    fast_w, fast_h = orig_img_fast.size

    for _ in range(num_trials):
        cand_svg, summary = op_generator()
        if cand_svg == fallback_svg:
            continue

        try:
            cand_png = rasterize_svg_to_png_bytes(cand_svg, out_w=fast_w, out_h=fast_h)
            cand_img = Image.open(io.BytesIO(cand_png)).convert("RGB")
            score = lab_l1(orig_img_fast, cand_img)

            if score < best_fast_score:
                best_fast_score = score
                best_svg = cand_svg
                best_summary = summary
        except Exception:
            continue

    return best_svg if best_svg is not None else fallback_svg, best_summary


def crossover(svg_a: str, svg_b: str, k: int = 2) -> str:
    """
    K-point crossover: split top-level children into k+1 contiguous segments,
    alternating which parent contributes each segment.
    """
    try:
        root_a = ET.fromstring(svg_a)
        root_b = ET.fromstring(svg_b)

        children_a = list(root_a)
        children_b = list(root_b)
        max_len = max(len(children_a), len(children_b))

        new_root = ET.Element(root_a.tag, root_a.attrib)

        if max_len <= 1:
            src = children_a or children_b
            for child in src:
                new_root.append(copy.deepcopy(child))
            ET.register_namespace("", SVG_NS)
            return ET.tostring(new_root, encoding="unicode", method="xml")

        actual_k = min(k, max_len - 1)
        cuts = sorted(random.sample(range(1, max_len), actual_k))

        segment = 0
        use_a = True
        for i in range(max_len):
            while segment < len(cuts) and i >= cuts[segment]:
                use_a = not use_a
                segment += 1
            children = children_a if use_a else children_b
            if i < len(children):
                new_root.append(copy.deepcopy(children[i]))

        ET.register_namespace("", SVG_NS)
        return ET.tostring(new_root, encoding="unicode", method="xml")
    except ET.ParseError:
        return svg_a


def mutate_remove_node(svg: str) -> str:
    try:
        root = ET.fromstring(svg)
        pairs: list[tuple[ET.Element, ET.Element]] = []
        for parent in root.iter():
            for child in list(parent):
                pairs.append((parent, child))

        if not pairs:
            return svg

        parent_elem, child = random.choice(pairs)
        parent_elem.remove(child)

        ET.register_namespace("", SVG_NS)
        return ET.tostring(root, encoding="unicode", method="xml")
    except ET.ParseError:
        return svg


def mutate_drop_style_property(svg: str) -> str:
    try:
        root = ET.fromstring(svg)
        styled = [el for el in root.iter() if el.get("style", "").strip()]
        if not styled:
            return svg

        el = random.choice(styled)
        props = [p.strip() for p in el.get("style", "").split(";") if p.strip()]
        if len(props) <= 1:
            return svg

        props.pop(random.randrange(len(props)))
        el.set("style", "; ".join(props))

        ET.register_namespace("", SVG_NS)
        return ET.tostring(root, encoding="unicode", method="xml")
    except ET.ParseError:
        return svg


def mutate_numeric(svg: str) -> str:
    try:
        root = ET.fromstring(svg)
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
            return svg

        elem, attr, num, unit = random.choice(candidates)
        factor = random.uniform(0.7, 1.3)
        new_num = num * factor

        if "opacity" in attr:
            new_num = max(0.0, min(1.0, new_num))

        if not unit and num == int(num) and new_num >= 0:
            elem.attrib[attr] = str(round(new_num))
        else:
            formatted = f"{new_num:.3f}".rstrip("0").rstrip(".")
            elem.attrib[attr] = f"{formatted}{unit}"

        ET.register_namespace("", SVG_NS)
        return ET.tostring(root, encoding="unicode", method="xml")
    except ET.ParseError:
        return svg


def mutate_color(svg: str) -> str:
    """Tweak a fill or stroke color — nudge hex channels or swap named color."""
    try:
        root = ET.fromstring(svg)

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
            return svg

        elem, source, key, val = random.choice(candidates)

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

        ET.register_namespace("", SVG_NS)
        return ET.tostring(root, encoding="unicode", method="xml")
    except ET.ParseError:
        return svg


def mutate_stroke(svg: str) -> str:
    """Add, remove, or change stroke on a random shape element."""
    try:
        root = ET.fromstring(svg)
        shapes = [el for el in root.iter() if _local_tag(el) in _SHAPE_TAGS]
        if not shapes:
            return svg

        el = random.choice(shapes)
        has_stroke = el.get("stroke") not in (None, "none", "")

        op = random.choice(["add", "remove", "change"])
        if op == "remove" and has_stroke:
            el.set("stroke", "none")
        elif op in ("add", "change"):
            el.set("stroke", random.choice(_NAMED_SVG_COLORS))
            if not el.get("stroke-width"):
                el.set("stroke-width", str(random.choice([1, 2, 3])))

        ET.register_namespace("", SVG_NS)
        return ET.tostring(root, encoding="unicode", method="xml")
    except ET.ParseError:
        return svg


def mutate_path(svg: str) -> str:
    """Nudge one numeric coordinate in a path 'd' attribute."""
    try:
        root = ET.fromstring(svg)
        paths = [el for el in root.iter() if el.get("d")]
        if not paths:
            return svg

        el = random.choice(paths)
        d = el.get("d", "")
        nums = list(_PATH_NUM_RE.finditer(d))
        if not nums:
            return svg

        m = random.choice(nums)
        val = float(m.group(1))
        # Use a proportional nudge (±15%) with a minimum of ±2px
        magnitude = max(2.0, abs(val) * 0.15)
        new_val = val + random.uniform(-magnitude, magnitude)
        new_str = f"{new_val:.1f}".rstrip("0").rstrip(".")
        el.set("d", d[: m.start()] + new_str + d[m.end() :])

        ET.register_namespace("", SVG_NS)
        return ET.tostring(root, encoding="unicode", method="xml")
    except ET.ParseError:
        return svg


def mutate_reorder(svg: str) -> str:
    """Swap two adjacent sibling elements to change z-order."""
    try:
        root = ET.fromstring(svg)
        candidates = [el for el in root.iter() if len(list(el)) >= 2]
        if not candidates:
            return svg

        parent = random.choice(candidates)
        children = list(parent)
        i = random.randrange(len(children) - 1)
        children[i], children[i + 1] = children[i + 1], children[i]
        for child in list(parent):
            parent.remove(child)
        for child in children:
            parent.append(child)

        ET.register_namespace("", SVG_NS)
        return ET.tostring(root, encoding="unicode", method="xml")
    except ET.ParseError:
        return svg


def crossover_with_micro_search(
    svg_a: str,
    svg_b: str,
    orig_img_fast: Image.Image,
    num_trials: int = 15,
) -> tuple[str, str]:
    def _op():
        cand = with_retries(lambda: crossover(svg_a, svg_b), fallback=svg_a)
        return cand, "Local crossover"

    return with_micro_search(
        _op,
        fallback_svg=svg_a,
        orig_img_fast=orig_img_fast,
        num_trials=num_trials,
        default_summary="Crossover: no improvement",
    )


def mutate_with_micro_search(
    parent_svg: str,
    orig_img_fast: Image.Image,
    num_trials: int = 15,
) -> tuple[str, str]:
    _ops = [
        (mutate_color, "Mutation: color tweak", 0.25),
        (mutate_numeric, "Mutation: numeric tweak", 0.20),
        (mutate_path, "Mutation: path nudge", 0.15),
        (mutate_remove_node, "Mutation: removed node", 0.15),
        (mutate_stroke, "Mutation: stroke change", 0.10),
        (mutate_reorder, "Mutation: reordered elements", 0.10),
        (mutate_drop_style_property, "Mutation: dropped style property", 0.05),
    ]
    fns, labels, weights = zip(*_ops, strict=True)

    def _op():
        fn, label = random.choices(
            list(zip(fns, labels, strict=True)), weights=list(weights), k=1
        )[0]
        cand = with_retries(lambda: fn(parent_svg), fallback=parent_svg)
        return cand, label

    return with_micro_search(
        _op,
        fallback_svg=parent_svg,
        orig_img_fast=orig_img_fast,
        num_trials=num_trials,
        default_summary="Mutation: no improvement",
    )
