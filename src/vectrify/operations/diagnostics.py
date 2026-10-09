"""Structured observations about an edit, separate from client guidance."""

from __future__ import annotations

from vectrify.document import Document


def edit_diagnostics(before: Document, after: Document) -> dict:
    old = {e.id: e for e in before.elements()}
    new = {e.id: e for e in after.elements()}
    effects: dict[str, list] = {
        key: []
        for key in (
            "nodes_removed",
            "corners_rounded",
            "geometry_moved",
            "paint_changed",
            "stacking_changed",
            "regions_merged",
            "protected_feature_violations",
        )
    }
    for oid in old.keys() & new.keys():
        a, b = old[oid], new[oid]
        if a.attributes != b.attributes:
            effects["paint_changed"].append(oid)
        if [c.id for c in a.children] != [c.id for c in b.children]:
            effects["stacking_changed"].append(oid)
        if a.geometry_id is None or b.geometry_id is None:
            continue
        ga, gb = before.geometry_for(oid), after.geometry_for(oid)
        na = {n.id: (sp.id, n) for sp in ga.subpaths for n in sp.nodes}
        nb = {n.id: n for sp in gb.subpaths for n in sp.nodes}
        for nid, (contour, node) in na.items():
            ref = {"object": oid, "contour": contour, "node": nid}
            if nid not in nb:
                effects["nodes_removed"].append(ref)
            elif node.endpoint != nb[nid].endpoint:
                effects["geometry_moved"].append(ref)
            elif node.command != "C" and nb[nid].command == "C":
                effects["corners_rounded"].append(ref)
            if node.pinned and (nid not in nb or node.endpoint != nb[nid].endpoint):
                effects["protected_feature_violations"].append({**ref, "kind": "pin"})
    removed = sorted(old.keys() - new.keys())
    # Report lineage only when a surviving path acquired removed contours.
    for oid in old.keys() & new.keys():
        if old[oid].geometry_id is None or new[oid].geometry_id is None:
            continue
        gained = {sp.id for sp in after.geometry_for(oid).subpaths} - {
            sp.id for sp in before.geometry_for(oid).subpaths
        }
        sources = [
            rid
            for rid in removed
            if old[rid].geometry_id is not None
            and gained & {sp.id for sp in before.geometry_for(rid).subpaths}
        ]
        if sources:
            effects["regions_merged"].append({"object": oid, "sources": sources})
    return {
        "created": sorted(new.keys() - old.keys()),
        "removed": removed,
        "effects": effects,
    }
