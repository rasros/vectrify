"""Explicit endpoint and corner constraints, independent of positional pins."""

from __future__ import annotations

from vectrify.document.model import Document, DocumentError, Geometry

KINDS = {"position", "tip", "corner", "junction"}


def held_nodes(geometry: Geometry) -> frozenset[str]:
    held = set()
    for contour in geometry.subpaths:
        for i, node in enumerate(contour.nodes):
            if node.feature is None:
                continue
            held.add(node.id)
            if node.feature != "position":
                held.add(contour.nodes[(i + 1) % len(contour.nodes)].id)
                held.add(contour.nodes[(i - 1) % len(contour.nodes)].id)
    return frozenset(held)


def violations(before: Document, after: Document) -> list[dict]:
    found = []
    geometries = {g.id: g for g in after.geometries}
    users = {e.geometry_id for e in after.elements()}
    for geometry in before.geometries:
        new = geometries.get(geometry.id) if geometry.id in users else None
        for contour in geometry.subpaths:
            for i, node in enumerate(contour.nodes):
                if node.feature is None:
                    continue
                reason = None
                next_node = contour.nodes[(i + 1) % len(contour.nodes)]
                try:
                    if new is None:
                        raise DocumentError("removed")
                    kept = new.node(node.id)
                    if kept.endpoint != node.endpoint:
                        reason = "moved"
                    elif kept.feature != node.feature:
                        reason = "protection removed"
                    elif node.feature != "position":
                        successor = next(
                            (
                                sp.nodes[(j + 1) % len(sp.nodes)]
                                for sp in new.subpaths
                                for j, n in enumerate(sp.nodes)
                                if n.id == node.id
                            ),
                            None,
                        )
                        if (
                            kept.command != node.command
                            or kept.values != node.values
                            or (
                                successor is None
                                or successor.command != next_node.command
                                or successor.values != next_node.values
                            )
                        ):
                            reason = "corner or junction geometry changed"
                except DocumentError:
                    reason = "removed"
                if reason:
                    found.append(
                        {
                            "geometry": geometry.id,
                            "contour": contour.id,
                            "node": node.id,
                            "kind": node.feature,
                            "reason": reason,
                        }
                    )
    return found


def require_preserved(before: Document, after: Document) -> None:
    problems = violations(before, after)
    if problems:
        p = problems[0]
        raise DocumentError(
            f"Protected {p['kind']} {p['node']} would be {p['reason']}; "
            "shorten the edited stretch or explicitly release protection"
        )
