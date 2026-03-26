"""
Local (non-LLM) SVG operations: crossover and mutations.

All functions accept SVG string(s) and return a new SVG string.
On parse failure they return the primary input unchanged.

Use `with_retries` to wrap any operation for automatic retry on invalid output:

    svg = with_retries(lambda: mutate_numeric(parent_svg), fallback=parent_svg)
"""

import copy
import random
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable

SVG_NS = "http://www.w3.org/2000/svg"

# Attributes whose numeric values are worth tweaking
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

# Matches a bare number optionally followed by a CSS unit or %
_NUM_RE = re.compile(r"^(-?\d+(?:\.\d+)?)([a-z%]*)$")


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
    """
    Call op() up to max_retries times, returning the first result that is a
    valid SVG document. Returns fallback if every attempt fails or raises.
    """
    for _ in range(max_retries):
        try:
            result = op()
            if _is_valid_svg(result):
                return result
        except Exception:
            pass
    return fallback


def crossover(svg_a: str, svg_b: str, rate: float = 0.5) -> str:
    """
    Produce a child SVG by interleaving top-level children from two parents.
    Each child element is taken from svg_a with probability `rate`, svg_b otherwise.
    Extra tail elements from either parent are included at their respective probability.
    """
    try:
        root_a = ET.fromstring(svg_a)
        root_b = ET.fromstring(svg_b)

        new_root = ET.Element(root_a.tag, root_a.attrib)

        children_a = list(root_a)
        children_b = list(root_b)
        max_len = max(len(children_a), len(children_b))

        for i in range(max_len):
            choice = random.random()
            if i < len(children_a) and i < len(children_b):
                src = children_a[i] if choice < rate else children_b[i]
                new_root.append(copy.deepcopy(src))
            elif i < len(children_a) and choice < rate:
                new_root.append(copy.deepcopy(children_a[i]))
            elif i < len(children_b) and choice >= rate:
                new_root.append(copy.deepcopy(children_b[i]))

        ET.register_namespace("", SVG_NS)
        return ET.tostring(new_root, encoding="unicode", method="xml")
    except ET.ParseError:
        return svg_a


def mutate_remove_node(svg: str) -> str:
    """Remove a randomly chosen element from anywhere in the SVG tree."""
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


def mutate_numeric(svg: str) -> str:
    """Tweak a random numeric attribute value by a random +-10-30% factor."""
    try:
        root = ET.fromstring(svg)

        candidates: list[tuple[ET.Element, str, float, str]] = []
        for elem in root.iter():
            for attr, val in elem.attrib.items():
                bare_attr = attr.split("}")[-1]  # strip namespace prefix
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
