"""Immutable proposals for node handles and fixed-topology endpoint movement.

These functions know geometry, not transactions, selections, locks or history.
The transaction facade validates command arguments and authorizes each proposal
before replacing its working document. Inputs are never mutated.
"""

from __future__ import annotations

import math
from dataclasses import replace

from vectrify.document.model import (
    Document,
    DocumentError,
    Geometry,
    PathNode,
    Subpath,
    new_id,
)


def on_chord(
    h: tuple[float, ...], a: tuple[float, float], b: tuple[float, float]
) -> bool:
    """Whether handle *h* lies on the straight line from *a* to *b*, so that a
    curve with its other handle retracted is just that line."""
    ax, ay = b[0] - a[0], b[1] - a[1]
    hx, hy = h[0] - a[0], h[1] - a[1]
    length = math.hypot(ax, ay)
    if length == 0:
        return math.hypot(hx, hy) < 1e-9
    along = (hx * ax + hy * ay) / length
    return abs(hx * ay - hy * ax) / length < 1e-6 * max(1.0, length) and (
        -1e-9 <= along <= length + 1e-9
    )


def toward(
    a: tuple[float, float], b: tuple[float, float], share: float
) -> tuple[float, float]:
    """The point *share* of the way from *a* to *b*."""
    return a[0] + (b[0] - a[0]) * share, a[1] + (b[1] - a[1]) * share


def carry_handles(before: Document, after: Document) -> Document:
    """Take the handles of every point an edit moved along with it.

    The curve keeps its shape around a moved point and a retracted handle
    stays a corner; a handle the edit already placed is left where it is.
    A closed contour ending on its moveto shows one point for two nodes, so
    either drags the other.
    """
    while True:
        changes: dict[tuple[str, str], list[float]] = {}
        for geometry in after.geometries:
            old = before.geometry(geometry.id)
            if geometry == old:
                continue
            for subpath, previous in zip(geometry.subpaths, old.subpaths, strict=True):
                nodes, olds = subpath.nodes, previous.nodes
                last = len(nodes) - 1
                twins = (
                    subpath.closed
                    and last > 1
                    and olds[last].endpoint == olds[0].endpoint
                )
                for i, (node, was) in enumerate(zip(nodes, olds, strict=True)):
                    if node.endpoint == was.endpoint:
                        continue
                    dx = node.endpoint[0] - was.endpoint[0]
                    dy = node.endpoint[1] - was.endpoint[1]
                    slots = [(i, 2)] if node.command == "C" else []
                    if i < last and nodes[i + 1].command == "C":
                        slots.append((i + 1, 0))
                    if twins and i in {0, last}:
                        slots.append((last - i, len(nodes[last - i].values) - 2))
                    for j, k in slots:
                        if nodes[j].values[k : k + 2] != olds[j].values[k : k + 2]:
                            continue
                        key = geometry.id, nodes[j].id
                        values = changes.setdefault(key, list(nodes[j].values))
                        values[k : k + 2] = (values[k] + dx, values[k + 1] + dy)
        if not changes:
            return after
        for (gid, nid), values in changes.items():
            geometry = after.geometry(gid)
            node = replace(geometry.node(nid), values=tuple(values))
            after = after.replace_geometry(geometry.replace_node(node))


def node_location(geometry: Geometry, node_id: str) -> tuple[int, Subpath, int]:
    """Locate a stable node ID without depending on document ownership."""
    for index, subpath in enumerate(geometry.subpaths):
        for position, node in enumerate(subpath.nodes):
            if node.id == node_id:
                return index, subpath, position
    raise DocumentError(f"Unknown path node: {node_id}")


def with_node_handles(geometry: Geometry, node_id: str, count: int) -> Geometry | None:
    """Propose zero, one or two handles; None means the point has no segment.

    The caller validates count and authorizes any new closing segment. Existing
    nodes retain their IDs and endpoint constraints. Alignment is a separate
    proposal so the facade can retain its existing authorization sequence.
    """
    index, subpath, position = node_location(geometry, node_id)
    nodes = list(subpath.nodes)
    point = nodes[position].endpoint
    # Materialize an implicit closing line when it needs a handle. Keep
    # existing IDs, including both representations of the closing point.
    last = len(nodes) - 1
    if (
        subpath.closed
        and last > 0
        and position in {0, last}
        and nodes[last].endpoint != nodes[0].endpoint
        and count > 0
    ):
        nodes.append(PathNode(new_id("node"), "L", nodes[0].endpoint))
        last += 1
    seam = (
        subpath.closed
        and last > 0
        and nodes[last].endpoint == nodes[0].endpoint
        and position in {0, last}
    )
    incoming = position if position > 0 else None
    outgoing = position + 1 if position + 1 < len(nodes) else None
    if seam:
        incoming, outgoing = last, 1

    def at(node_index: int) -> tuple[float, float]:
        return nodes[node_index].endpoint

    def handle_in() -> tuple[float, float] | None:
        if incoming is None or nodes[incoming].command != "C":
            return None
        h = nodes[incoming].values[2:4]
        return None if h == point else (h[0], h[1])

    def handle_out() -> tuple[float, float] | None:
        if outgoing is None or nodes[outgoing].command != "C":
            return None
        h = nodes[outgoing].values[0:2]
        return None if h == point else (h[0], h[1])

    def set_in(h: tuple[float, float] | None) -> None:
        if incoming is None:
            return
        node, start = nodes[incoming], at(incoming - 1)
        if h is None:
            if node.command == "C":
                c1 = node.values[0:2]
                nodes[incoming] = (
                    replace(node, command="L", values=point)
                    if on_chord(c1, start, point)
                    else replace(node, values=(*c1, *point, *point))
                )
            return
        c1 = node.values[0:2] if node.command == "C" else toward(start, point, 1 / 3)
        nodes[incoming] = replace(node, command="C", values=(*c1, *h, *point))

    def set_out(h: tuple[float, float] | None) -> None:
        if outgoing is None:
            return
        node, end = nodes[outgoing], at(outgoing)
        if h is None:
            if node.command == "C":
                c2 = node.values[2:4]
                nodes[outgoing] = (
                    replace(node, command="L", values=end)
                    if on_chord(c2, point, end)
                    else replace(node, values=(*point, *c2, *end))
                )
            return
        c2 = node.values[2:4] if node.command == "C" else toward(end, point, 1 / 3)
        nodes[outgoing] = replace(node, command="C", values=(*h, *c2, *end))

    # Smooth handle positions: along the neighbours' chord, a third of each
    # segment long, or toward the only neighbour at an open end.
    before = at(incoming - 1) if incoming is not None else None
    after = at(outgoing) if outgoing is not None else None
    ends = [p for p in (before, after) if p is not None]
    if not ends:
        return None
    tail, head = (before or point), (after or point)
    dx, dy = head[0] - tail[0], head[1] - tail[1]
    if dx == dy == 0:
        dx, dy = head[0] - point[0], head[1] - point[1]
    norm = math.hypot(dx, dy) or 1.0
    ux, uy = dx / norm, dy / norm

    def smooth(neighbour: tuple[float, float] | None, sign: float):
        if neighbour is None:
            return None
        length = math.dist(point, neighbour) / 3
        return (point[0] + sign * ux * length, point[1] + sign * uy * length)

    if count == 0:
        set_in(None)
        set_out(None)
    elif count == 2:
        set_in(handle_in() or smooth(before, -1))
        set_out(handle_out() or smooth(after, 1))
    else:
        only_in = handle_in() is not None and handle_out() is None
        if (only_in and outgoing is not None) or incoming is None:
            set_in(None)
            set_out(handle_out() or smooth(after, 1))
        else:
            set_out(None)
            set_in(handle_in() or smooth(before, -1))
    subpaths = list(geometry.subpaths)
    subpaths[index] = replace(subpath, nodes=tuple(nodes))
    return replace(geometry, subpaths=tuple(subpaths))


def without_handle(geometry: Geometry, node_id: str, offset: int) -> Geometry:
    """Retract one handle of a caller-validated cubic and offset (0 or 2)."""
    index, subpath, position = node_location(geometry, node_id)
    nodes = list(subpath.nodes)
    node = nodes[position]
    start, end = nodes[position - 1].endpoint, node.endpoint
    values = list(node.values)
    values[offset : offset + 2] = start if offset == 0 else end
    nodes[position] = (
        replace(node, command="L", values=end)
        if tuple(values[:2]) == start and tuple(values[2:4]) == end
        else replace(node, values=tuple(values))
    )
    subpaths = list(geometry.subpaths)
    subpaths[index] = replace(subpath, nodes=tuple(nodes))
    return replace(geometry, subpaths=tuple(subpaths))


def with_handle_alignment(geometry: Geometry, node_id: str, aligned: bool) -> Geometry:
    """Set alignment metadata, including both representations of a closed seam."""
    geometry.node(node_id)
    subpaths = []
    for subpath in geometry.subpaths:
        ids = {node_id}
        first, last = subpath.nodes[0], subpath.nodes[-1]
        if (
            subpath.closed
            and first.endpoint == last.endpoint
            and node_id in {first.id, last.id}
        ):
            ids.update((first.id, last.id))
        subpaths.append(
            replace(
                subpath,
                nodes=tuple(
                    replace(node, handles_aligned=aligned) if node.id in ids else node
                    for node in subpath.nodes
                ),
            )
        )
    return replace(geometry, subpaths=tuple(subpaths))


def moved_handle(
    geometry: Geometry, node_id: str, offset: int, point: tuple[float, float]
) -> tuple[tuple[float, ...], PathNode]:
    """Propose one control change and return its anchor's original metadata.

    The caller validates the cubic and offset, then decides whether the anchor
    requires a second proposal to align the opposite handle. Values remain raw
    so PathNode validation still happens inside the facade's update_node change
    guard, preserving transaction poisoning for invalid coordinates.
    """
    _, subpath, position = node_location(geometry, node_id)
    node = subpath.nodes[position]
    anchor = subpath.nodes[position - 1] if offset == 0 else node
    values = list(node.values)
    values[offset : offset + 2] = point
    return tuple(values), anchor


def with_straightened_handles(
    geometry: Geometry, node_id: str, side: int | None = None
) -> Geometry | None:
    """Align existing controls with fixed lengths; None means no edit is needed.

    A caller-validated side (0 outgoing or 2 incoming) fixes that handle. Without
    a side, minimize total squared movement. Retracted controls stay retracted.
    """
    index, subpath, position = node_location(geometry, node_id)
    subpaths = list(geometry.subpaths)
    nodes = list(subpath.nodes)
    point = nodes[position].endpoint
    last = len(nodes) - 1
    seam = (
        subpath.closed
        and last > 0
        and nodes[last].endpoint == nodes[0].endpoint
        and position in {0, last}
    )
    incoming, outgoing = (last, 1) if seam else (position, position + 1)
    if (
        incoming == 0
        or outgoing >= len(nodes)
        or nodes[incoming].command != "C"
        or nodes[outgoing].command != "C"
    ):
        return None
    hin, hout = nodes[incoming].values[2:4], nodes[outgoing].values[:2]
    lin, lout = math.dist(point, hin), math.dist(point, hout)
    if not lin or not lout:
        return None
    vin = ((point[0] - hin[0]) / lin, (point[1] - hin[1]) / lin)
    vout = ((hout[0] - point[0]) / lout, (hout[1] - point[1]) / lout)
    if all(abs(a - b) < 1e-12 for a, b in zip(vin, vout, strict=True)):
        return None
    # With lengths fixed, squared handle displacement is minimized
    # by the sum of their directions weighted by squared length.
    # When the handles belong to different cubics, this also
    # minimizes their integrated squared curve displacement.
    # Scale the lengths first to avoid overflowing their squares.
    scale = max(lin, lout)
    win, wout = (lin / scale) ** 2, (lout / scale) ** 2
    direction = (
        vin
        if side == 2
        else vout
        if side == 0
        else (win * vin[0] + wout * vout[0], win * vin[1] + wout * vout[1])
    )
    norm = math.hypot(*direction)
    if norm < 1e-12:
        direction, norm = (vin if lin > lout else vout), 1.0
    ux, uy = direction[0] / norm, direction[1] / norm
    for segment, offset, length, sign in (
        (incoming, 2, lin, -1),
        (outgoing, 0, lout, 1),
    ):
        if side == offset:
            continue
        node = nodes[segment]
        values = list(node.values)
        values[offset : offset + 2] = (
            point[0] + sign * ux * length,
            point[1] + sign * uy * length,
        )
        nodes[segment] = replace(node, values=tuple(values))
    subpaths[index] = replace(subpath, nodes=tuple(nodes))
    return replace(geometry, subpaths=tuple(subpaths))
