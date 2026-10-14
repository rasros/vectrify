"""Shared helpers for the document tests."""

import io

import cairosvg
import numpy as np
from PIL import Image
from shapely.geometry import box

from vectrify.document import Editor, Selection, import_svg
from vectrify.document.paint import GradientStop, LinearGradient

RINGS = "M0 0H30V30H0Z M5 5V25H25V5Z M60 0H90V30H60Z"


def drawing(data=RINGS, attrs=""):
    return import_svg(
        '<svg width="120" height="80"><g fill="red" transform="translate(5 8)">'
        f'<path id="p" d="{data}" {attrs}/></g></svg>'
    )


SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"
viewBox="0 0 64 64">
<g id="layer"><path id="a" fill="red" d="M4 4 L28 4 L28 28 L4 28 Z"/></g>
<path id="b" fill="blue" d="M36 36 C40 30 54 30 60 36 L60 60 L36 60 Z"/>
</svg>"""


SHARED = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">
<defs id="library"><path id="shape" fill-rule="evenodd"
d="M4 4 H28 V28 H4 Z M10 10 H22 V22 H10 Z"/>
<clipPath id="clip"><use id="clip-use" href="#shape"/></clipPath></defs>
<use id="first" href="#shape" fill="red"/>
<use id="second" href="#shape" fill="blue" transform="translate(30 30)"/>
<rect id="clipped" width="32" height="32" fill="green" clip-path="url(#clip)"/>
</svg>"""


def render(svg):
    png = cairosvg.svg2png(bytestring=svg.encode())
    assert png is not None
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))


def select(*ids, nodes=()):
    return Selection(frozenset(ids), frozenset(nodes))


GRADIENT_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" '
    'xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 10 10">'
    '<defs id="defs"><linearGradient id="base">'
    '<stop id="s0" offset="0" stop-color="#ff0000"/>'
    '<stop id="s1" offset="100%" stop-color="#0000ff" stop-opacity="0.5"/>'
    "</linearGradient>"
    '<linearGradient id="g" xlink:href="#base" gradientUnits="userSpaceOnUse" '
    'x1="0" y1="0" x2="10" y2="0" spreadMethod="reflect"/></defs>'
    '<path id="p" d="M0 0 H10 V10 H0 Z" fill="url(#g)"/>'
    '<rect id="r" width="2" height="2" fill="#00ff00"/></svg>'
)


RAMP = LinearGradient(
    (0, 0), (2, 1), (GradientStop(0, "#000000"), GradientStop(1, "#ffffff"))
)


def private_editor():
    editor = Editor(
        import_svg(
            '<svg width="10" height="10"><path id="p" d="M0 0 H10 V10 H0 Z"/>'
            '<rect id="r" width="1" height="1"/></svg>'
        ),
        selection=Selection.all(),
    )
    with editor.transaction("Private fill") as tx:
        tx.set_fill("p", RAMP)
    return editor


def hits(hit_index, x, y, w=1, h=1, mode="intersect", candidates=None):
    """Objects painting in a box: leaf objects by default, or *candidates*.

    A leaf counts when its paint meets the box (with contain, lies inside it);
    a group only when the box covers all of its paint.
    """
    region = box(x, y, x + w, y + h)
    document = hit_index.document
    if candidates is None:
        candidates = {e.id for e in document.elements() if e.tag not in {"svg", "g"}}
    found = set()
    for oid in candidates:
        area = hit_index.area(oid)
        if area is None:
            continue
        group = document.element(oid).tag in {"svg", "g"}
        meets = (
            region.intersects if mode == "intersect" and not group else region.covers
        )
        if meets(area):
            found.add(oid)
    return found


def pair(attrs_a="", attrs_b="", second="M40 0H50V10H40Z", extra=""):
    return import_svg(
        '<svg width="100" height="100"><g id="g" fill="red">'
        f'<path id="a" d="M0 0H10V10H0Z" {attrs_a}/>{extra}'
        f'<path id="b" d="{second}" {attrs_b}/></g></svg>'
    )


CONTOURS = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">
<path id="fill" fill="red" d="M8 8 C16 4 40 4 48 8 L48 40 L8 40 Z"/>
<path id="outline" fill="none" stroke="black" d="M48 8 C40 4 16 4 8 8"/>
<path id="other" fill="none" stroke="blue" d="M48 40 L48 8"/>
</svg>"""


def contour_editor():
    return Editor(import_svg(CONTOURS), selection=select("fill", "outline"))
