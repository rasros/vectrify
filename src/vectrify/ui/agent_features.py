"""Reviewable membership and extraction of multi-object visual features."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import numpy as np
from PIL import ImageDraw
from shapely.geometry import Polygon

from vectrify.document import DocumentError, HitIndex, Selection, export_svg
from vectrify.document.join import path_style
from vectrify.document.model import Geometry
from vectrify.document.regions import object_matrix, samples, split_geometry
from vectrify.document.topology import mapped_point

ROLES = {"fill", "shading", "highlight", "shadow", "outline"}


def inspect(agent: Any, polygon: list, cut: bool) -> tuple[list[dict], bytes]:
    from vectrify.ui.agent import _png, _size

    shape = Polygon(polygon)
    x0, y0, x1, y1 = shape.bounds
    box = (x0, y0, x1 - x0, y1 - y0)
    size = _size(box, 1024)
    image = agent._cached("drawing", box, size).copy()
    draw = ImageDraw.Draw(image)
    document = agent.session.editor.snapshot.document
    candidates = []
    for oid, contours in agent._covering(shape):
        if document.element(oid).tag != "path":
            continue
        geometry = document.geometry_for(oid)
        matrix = object_matrix(document, oid)
        style = path_style(document, document.element(oid))
        for index, paint in contours:
            contour = geometry.subpaths[index]
            split = split_geometry(
                Geometry(geometry.id, (contour,)),
                matrix,
                polygon,
                filled=style["fill"] != "none",
                rule=style["fill-rule"],
                cut=cut,
            )
            candidates.append(
                {
                    "object": oid,
                    "contour": contour.id,
                    "index": index,
                    "paint_component": paint,
                    "paint": style,
                    "role": None,
                    "boundary_cut": split.cut,
                    "extractable": bool(split.inside),
                    "fragment_d": Geometry(geometry.id, split.inside).path_data(),
                    "geometry_users": sorted(document.geometry_users(geometry.id)),
                }
            )
            points = [mapped_point(p, matrix) for p in samples(contour)]
            points = [
                ((x - x0) * size[0] / box[2], (y - y0) * size[1] / box[3])
                for x, y in points
            ]
            if len(points) > 1:
                colour = ["#ff0080", "#00bbff", "#ff9900", "#aa55ff"][
                    len(candidates) % 4
                ]
                draw.line(points, fill=colour, width=2)
                draw.text(points[0], str(len(candidates)), fill=colour)
    return candidates, _png(image)


def isolate(
    agent: Any, polygon: list, members: list[dict], name: str, cut: bool, detach: bool
) -> dict:
    from vectrify.ui.agent import _size, render_document

    editor = agent.session.editor
    document = editor.snapshot.document
    if not isinstance(name, str) or not name.strip():
        raise DocumentError("Give the isolated feature a name")
    if not isinstance(members, list) or not members:
        raise DocumentError("Review candidates and explicitly include/exclude members")
    selected: dict[str, set[str]] = {}
    roles = []
    for member in members:
        if not isinstance(member, dict) or set(member) - {
            "object",
            "contours",
            "role",
            "include",
        }:
            raise DocumentError("Members use object, contours, role and include")
        if member.get("role") not in ROLES or type(member.get("include")) is not bool:
            raise DocumentError(
                "Every member needs an explicit role and include=true/false"
            )
        oid = str(member["object"])
        geometry = document.geometry_for(oid)
        contours = member.get("contours")
        if not isinstance(contours, list) or not contours:
            raise DocumentError("Explicitly name each member contour ID")
        if not set(contours) <= {sp.id for sp in geometry.subpaths}:
            raise DocumentError(f"{oid}: unknown contour ID")
        if member["include"]:
            selected.setdefault(oid, set()).update(contours)
            roles.append(member)
    if not selected:
        raise DocumentError("Include at least one feature component")
    tx = editor.transaction("Isolate feature", selection=Selection(whole_document=True))
    scopes = {}
    cuts = []
    entire = []
    detached = []
    for oid, contour_ids in selected.items():
        geometry = tx.preview.geometry_for(oid)
        if len(tx.preview.geometry_users(geometry.id)) > 1:
            if not detach:
                raise DocumentError(
                    f"{oid}: shared geometry requires explicit detach=true"
                )
            # Detached IDs retain explicit membership by contour index.
            indices = [
                i for i, sp in enumerate(geometry.subpaths) if sp.id in contour_ids
            ]
            tx.detach_geometry(oid)
            geometry = tx.preview.geometry_for(oid)
            contour_ids = {geometry.subpaths[i].id for i in indices}
            detached.append(oid)
        scopes[oid] = frozenset(contour_ids)
        style = path_style(tx.preview, tx.preview.element(oid))
        chosen = replace(
            geometry,
            subpaths=tuple(sp for sp in geometry.subpaths if sp.id in contour_ids),
        )
        split = split_geometry(
            chosen,
            object_matrix(tx.preview, oid),
            polygon,
            filled=style["fill"] != "none",
            rule=style["fill-rule"],
            cut=cut,
        )
        if not split.inside:
            raise DocumentError(
                f"{oid}: chosen contours are not contained; "
                "expand the region or use cut=true"
            )
        if split.cut:
            cuts.append(
                {"object": oid, "contours": sorted(contour_ids), "boundary": polygon}
            )
        if not split.outside and len(chosen.subpaths) == len(geometry.subpaths):
            entire.append(oid)
    extracted = tx.extract_region(polygon, cut=cut, contours=scopes)
    mappings = [
        {"source": oid, "object": made, "source_contours": sorted(selected[oid])}
        for oid, made in extracted
        if made is not None
    ]
    mappings += [
        {"source": oid, "object": oid, "source_contours": sorted(selected[oid])}
        for oid in entire
    ]
    moving = frozenset(m["object"] for m in mappings)
    if not moving:
        raise DocumentError("No complete feature components were extracted")
    split_document = tx.preview
    # Choose the common ancestor and preserve component paint order.
    chains = [split_document.ancestry(oid)[:-1] for oid in moving]
    common = chains[0]
    for chain in chains[1:]:
        common = tuple(a for a, b in zip(common, chain, strict=False) if a.id == b.id)
    parent = common[-1]
    order = list(split_document.elements())
    positions = {e.id: i for i, e in enumerate(order)}
    front = max(moving, key=positions.__getitem__)
    top = next(e for e in split_document.ancestry(front) if e in parent.children)
    top_index = next(i for i, c in enumerate(parent.children) if c.id == top.id)
    index = sum(c.id not in moving for c in parent.children[: top_index + 1])
    # Refuse to pass unrelated painted geometry when gathering a group.
    hits = HitIndex(split_document)
    last = max(
        positions[e.id]
        for e in order
        if top.id in {a.id for a in split_document.ancestry(e.id)}
    )
    for oid in moving:
        area = hits.area(oid)
        if area is None:
            continue
        for element in order[positions[oid] + 1 : last + 1]:
            if element.id in moving or element.tag in {"svg", "g", "defs"}:
                continue
            other = hits.area(element.id)
            if other is not None and area.intersection(other).area > 1e-6:
                raise DocumentError(
                    f"Grouping would change stacking over {element.id}; "
                    "exclude it or choose a smaller complete feature"
                )
    tx.move_objects(moving, parent.id, index)
    group = tx.group_objects(moving)
    tx.set_attributes(group, {"data-vectrify-feature-sources": json.dumps(mappings)})
    # Verify effective paint/transform/stacking after gathering, independently.
    x0, y0, x1, y1 = Polygon(polygon).bounds
    boxes = (
        document.artboard(),
        (float(x0), float(y0), float(x1 - x0), float(y1 - y0)),
    )
    for box in boxes:
        for side in (512, 1024):
            size = _size(box, side)
            before = np.asarray(render_document(export_svg(split_document), box, size))
            after = np.asarray(render_document(export_svg(tx.preview), box, size))
            if not np.array_equal(before, after):
                raise DocumentError(
                    "Grouping would change effective paint or stacking; "
                    "narrow the membership"
                )
    tx.commit()
    editor.select(Selection(frozenset({group})))
    editor.rename_object(group, name)
    return {
        "group": group,
        "name": name,
        "source_mapping": mappings,
        "members": roles,
        "boundary_cuts": cuts,
        "detached": detached,
        "paint_and_stacking_verified": True,
    }
