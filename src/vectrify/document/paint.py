"""Linear gradient paint: the spec an edit sets, and reading one back.

A fill or stroke is either a solid colour or ``url(#id)`` naming a
``linearGradient`` in ``defs``. Code that needs one colour for a gradient,
such as area-weighted joins, uses its mean colour along the gradient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

from cairosvg.colors import color

from vectrify.document.model import (
    Document,
    DocumentError,
    Element,
    new_id,
    paint_server,
)
from vectrify.document.svg import SOLID


@dataclass(frozen=True)
class GradientStop:
    offset: float
    colour: str
    opacity: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "offset", float(self.offset))
        object.__setattr__(self, "opacity", float(self.opacity))
        if not 0 <= self.offset <= 1 or not 0 <= self.opacity <= 1:
            raise DocumentError("Stop offsets and opacities must be between 0 and 1")
        if not SOLID.fullmatch(self.colour) or self.colour.lower() == "none":
            raise DocumentError("A gradient stop needs a solid colour")


@dataclass(frozen=True)
class LinearGradient:
    """A gradient from *start* to *end* in the painted object's user space."""

    start: tuple[float, float]
    end: tuple[float, float]
    stops: tuple[GradientStop, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", tuple(float(v) for v in self.start))
        object.__setattr__(self, "end", tuple(float(v) for v in self.end))
        object.__setattr__(self, "stops", tuple(self.stops))
        if len(self.start) != 2 or len(self.end) != 2:
            raise DocumentError("A gradient needs two end points")
        if not all(math.isfinite(v) for v in (*self.start, *self.end)):
            raise DocumentError("Gradient end points must be finite")
        if not self.stops:
            raise DocumentError("A gradient needs at least one stop")
        offsets = [stop.offset for stop in self.stops]
        if offsets != sorted(offsets):
            raise DocumentError("Gradient stops must be in offset order")

    def attributes(self) -> tuple[tuple[str, str], ...]:
        (x1, y1), (x2, y2) = self.start, self.end
        return (
            ("gradientUnits", "userSpaceOnUse"),
            ("x1", repr(x1)),
            ("y1", repr(y1)),
            ("x2", repr(x2)),
            ("y2", repr(y2)),
        )

    def element(self, gradient_id: str, stop_ids: tuple[str, ...] = ()) -> Element:
        """The ``linearGradient`` element, reusing *stop_ids* in order."""
        ids = (*stop_ids, *(new_id("object") for _ in self.stops))
        stops = tuple(
            Element(
                ids[i],
                "stop",
                (
                    ("offset", repr(stop.offset)),
                    ("stop-color", stop.colour),
                    *(
                        (("stop-opacity", repr(stop.opacity)),)
                        if stop.opacity < 1
                        else ()
                    ),
                ),
            )
            for i, stop in enumerate(self.stops)
        )
        return Element(gradient_id, "linearGradient", self.attributes(), stops)


def _offset(value: str | None) -> float:
    if not value:
        return 0.0
    number = float(value[:-1]) / 100 if value.endswith("%") else float(value)
    return min(1.0, max(0.0, number))


def gradient_stops(element: Element) -> list[tuple[float, tuple[float, ...]]]:
    """(offset, rgba) of a gradient's stops, offsets clamped and nondecreasing."""
    stops: list[tuple[float, tuple[float, ...]]] = []
    previous = 0.0
    for stop in element.children:
        offset = max(previous, _offset(stop.get("offset")))
        previous = offset
        r, g, b, a = (float(v) for v in color(stop.get("stop-color") or "black"))
        a *= float(stop.get("stop-opacity") or 1)
        stops.append((offset, (r, g, b, a)))
    return stops


def mean_colour(element: Element) -> tuple[float, float, float, float]:
    """The average rgba along a gradient from 0 to 1, as padding paints it."""
    stops = gradient_stops(element)
    if not stops:
        return (0.0, 0.0, 0.0, 0.0)
    points = [(0.0, stops[0][1]), *stops, (1.0, stops[-1][1])]
    total = [0.0, 0.0, 0.0, 0.0]
    for (t0, c0), (t1, c1) in pairwise(points):
        for i in range(4):
            total[i] += (t1 - t0) * (c0[i] + c1[i]) / 2
    return total[0], total[1], total[2], total[3]


def hex_colour(rgba: tuple[float, ...]) -> str:
    return "#" + "".join(f"{round(255 * min(1.0, max(0.0, v))):02x}" for v in rgba[:3])


def solid_paint(document: Document, value: str) -> str:
    """*value*, or a gradient's mean colour where one colour is needed."""
    server = paint_server(value)
    if server is None:
        return value
    try:
        element = document.element(server)
    except DocumentError:
        return "none"
    return hex_colour(mean_colour(element)) if element.children else "none"
