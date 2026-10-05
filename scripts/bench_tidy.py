"""Tidy saved projects with every path in a group selected simultaneously.

    uv run python scripts/bench_tidy.py --out .bench/sword.json

The fixed sword snapshot includes its reference. Shape properties are measured
in document units, independently of RGB error and render resolution. The outline
is implicitly closed at the blade's base; its stroke is not the interior.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from dataclasses import replace
from pathlib import Path

import cairosvg
import numpy as np
import shapely
from PIL import Image
from scipy.ndimage import binary_erosion
from shapely.affinity import affine_transform

from vectrify.document import Editor, Selection, export_svg, load_project, save_project
from vectrify.document.hit_test import _filled, _flatten
from vectrify.document.join import path_style
from vectrify.document.topology import mapped_point
from vectrify.document.transforms import object_matrix
from vectrify.image_utils import on_white
from vectrify.operations import Budget, Job, OperationRequest, Permissions, method

FIXTURE = Path(__file__).parent / "bench_data/projects/sword.vectrify"
GROUP = "group_343f776e97514ac39bd469b1143cc40a"
OUTLINE = "object_68f9c655457542db9c945eef1e192b0c"


def load(path=FIXTURE):
    project = json.loads(path.read_text())
    document, _selection = load_project(json.dumps(project["document"]))
    data = project["reference"]["data_url"].split(",", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGBA")
    return document, image


def _render(document, white=False):
    png = cairosvg.svg2png(
        bytestring=export_svg(document).encode(),
        background_color="white" if white else None,
    )
    assert png is not None
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))


def _only(document, oids):
    keep = {a.id for oid in oids for a in document.ancestry(oid)}

    def prune(element):
        if element.tag == "defs" or element.id in oids:
            return element
        return replace(
            element,
            children=tuple(
                prune(c) for c in element.children if c.id in keep or c.tag == "defs"
            ),
        )

    return replace(document, root=prune(document.root))


def _axis(document, outline):
    from vectrify.document.transforms import object_matrix

    nodes = document.geometry_for(outline).subpaths[0].nodes
    base = (np.array(nodes[0].values[-2:]) + nodes[-1].values[-2:]) / 2
    tip = np.array(nodes[len(nodes) // 2].values[-2:])
    a, b, c, d, e, f = object_matrix(document, outline)
    return np.array([(a * x + c * y + e, b * x + d * y + f) for x, y in (base, tip)])


# A path's flattening error has a smaller area budget than the strict 0.05
# checks. Bound displacement by the control-hull perimeter in document units.
AREA_ERROR = 0.0001


def _shape(document, oid):
    """Double-precision fill region, avoiding unstable cubic XOR intersections."""
    matrix = object_matrix(document, oid)
    contours = []
    length = 0.0
    for sub in document.geometry_for(oid).subpaths:
        first = mapped_point(sub.nodes[0].endpoint, matrix)
        previous = first
        spans = []
        for node in sub.nodes[1:]:
            points = (
                previous,
                *(
                    mapped_point(p, matrix)
                    for p in zip(node.values[::2], node.values[1::2], strict=True)
                ),
            )
            length += float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            spans.append(points)
            previous = points[-1]
        length += float(np.linalg.norm(np.array(previous) - first))
        contours.append((first, spans))
    tolerance = AREA_ERROR / max(2 * length, 1)
    flattened = []
    for first, spans in contours:
        points = [first]
        for controls in spans:
            points.extend(
                [controls[-1]] if len(controls) == 2 else _flatten(controls, tolerance)
            )
        flattened.append((tuple(points), True))
    return _filled(
        tuple(flattened), path_style(document, document.element(oid))["fill-rule"]
    )


def _symmetry_area(interior, axis):
    axis = np.asarray(axis, dtype=float)
    direction = axis[1] - axis[0]
    direction /= np.linalg.norm(direction)
    linear = 2 * np.outer(direction, direction) - np.eye(2)
    offset = axis[0] - linear @ axis[0]
    reflected = affine_transform(
        interior, (linear[0, 0], linear[0, 1], linear[1, 0], linear[1, 1], *offset)
    )
    return interior.symmetric_difference(reflected).area


def _covered_pixels(document, pixels, union):
    """Count transparent pixel squares fully covered by the fill geometry.

    Coverage need not imply opacity: adjacent antialiased fills and deliberately
    translucent paint can both leave transparency without a geometric gap.
    Map the native SVG viewport, including its aspect-ratio alignment.
    """
    height, width = pixels.shape
    vx, vy, vw, vh = document.artboard()
    scale = np.array((width / vw, height / vh))
    offset = np.zeros(2)
    aspect = (document.root.get("preserveAspectRatio") or "xMidYMid meet").split()
    if aspect[0] == "defer":
        aspect = aspect[1:]
    if aspect[0] != "none":
        scale[:] = scale.max() if aspect[-1] == "slice" else scale.min()
        extra = np.array((width, height)) - np.array((vw, vh)) * scale
        alignment = aspect[0]
        offset = extra * np.array(
            [
                1 if f"{a}Max" in alignment else 0.5 if f"{a}Mid" in alignment else 0
                for a in ("x", "Y")
            ]
        )
    y, x = np.nonzero(pixels)
    left, top = (x - offset[0]) / scale[0] + vx, (y - offset[1]) / scale[1] + vy
    boxes = shapely.box(left, top, left + 1 / scale[0], top + 1 / scale[1])
    shapely.prepare(union)
    return int(shapely.covers(union, boxes).sum())


def properties(document, group=GROUP, outline=OUTLINE, axis=None):
    """Containment per fill, uncovered interior, and reflected silhouette error.

    Reflect about the line from the outline's tip to the midpoint of its base.
    Opaque coverage is also rendered alone,
    so an underlying hilt or background cannot hide a transparent blade gap.
    Report completed-axis symmetry and transparent pixels entirely inside the
    fill union as diagnostics, independently of the original-axis quality gate.
    """
    interior = _shape(document, outline)
    fills = [
        e.id
        for e in document.element(group).children
        if e.tag == "path" and path_style(document, e)["fill"] != "none"
    ]
    shapes = {oid: _shape(document, oid) for oid in fills}
    union = shapely.union_all(tuple(shapes.values()))
    # This case's symmetry axis is a benchmark expectation, not a fitting rule.
    axis = _axis(document, outline) if axis is None else np.asarray(axis, dtype=float)
    spill = {oid: s.difference(interior).area for oid, s in shapes.items()}
    # Measure transparency in the blade alone, independently of any objects
    # underneath. Exclude boundary antialiasing with a two-pixel interior band.
    blade = _render(_only(document, [group]))
    element = document.element(outline)
    mask_document = document.replace_element(
        replace(
            element,
            attributes=tuple(
                {**dict(element.attributes), "fill": "black", "stroke": "none"}.items()
            ),
        )
    )
    mask = _render(_only(mask_document, [outline]))[:, :, 3] == 255
    mask = binary_erosion(mask, iterations=2)
    visible = _render(document)
    visible_transparent = (visible[:, :, 3] < 255) & mask
    return {
        "outside_area": sum(spill.values()),
        "outside_each": spill,
        "gap_area": interior.difference(union).area,
        "symmetry_area": _symmetry_area(interior, axis),
        "symmetry_area_completed_axis": _symmetry_area(
            interior, _axis(document, outline)
        ),
        "interior_area": interior.area,
        "transparent_pixels": int(((blade[:, :, 3] < 255) & mask).sum()),
        "visible_transparent_pixels": int(visible_transparent.sum()),
        "visible_transparent_pixels_inside_fills": _covered_pixels(
            document, visible_transparent, union
        ),
    }


def run(path=FIXTURE, settings=None, group=GROUP, outline=OUTLINE):
    document, image = load(path)
    axis = _axis(document, outline)
    before = properties(document, group, outline, axis)
    editor = Editor(document, selection=Selection(object_ids=frozenset({group})))
    settings = dict(settings or {})
    request = OperationRequest(
        "improve",
        "nodes",
        editor.snapshot,
        editor,
        Permissions(geometry=True, structure=True, paint=True),
        settings={k: v for k, v in settings.items() if k != "rounds"},
        budget=Budget(steps=settings.get("rounds")),
        reference=image,
    )
    job = Job(method("improve", "nodes"), request)
    began = time.perf_counter()
    job.run()
    state = job.state()
    if state["status"] != "ready":
        raise RuntimeError(state)
    if state["result"]["changed"]:
        job.apply()
    final = editor.snapshot.document
    after = properties(final, group, outline, axis)
    target = np.asarray(on_white(image), float)
    mse = [
        float(((_render(d, white=True)[:, :, :3].astype(float) - target) ** 2).mean())
        for d in (document, final)
    ]
    return final, {
        "case": "sword-blade",
        "simultaneous": True,
        "paths": len(document.element(group).children),
        "symmetry_axis": axis.tolist(),
        "settings": settings,
        "seconds": round(time.perf_counter() - began, 3),
        "before": before,
        "after": after,
        "rgb_mse": mse,
        "tidy": state["result"]["metrics"],
        "passed": after["transparent_pixels"] == 0
        and all(after[k] < 0.05 for k in ("outside_area", "gap_area", "symmetry_area")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=FIXTURE)
    parser.add_argument("--group", default=GROUP)
    parser.add_argument("--outline", default=OUTLINE)
    parser.add_argument("--nodes", type=json.loads, default={})
    parser.add_argument("--out", type=Path)
    parser.add_argument("--svg", type=Path)
    parser.add_argument("--save-project", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="Fail if a layout check fails"
    )
    args = parser.parse_args()
    document, row = run(args.project, args.nodes, args.group, args.outline)
    text = json.dumps(row, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    if args.svg:
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        args.svg.write_text(export_svg(document))
    if args.save_project:
        args.save_project.parent.mkdir(parents=True, exist_ok=True)
        project = json.loads(args.project.read_text())
        project["document"] = json.loads(
            save_project(document, Selection(object_ids=frozenset({args.group})))
        )
        args.save_project.write_text(json.dumps(project, allow_nan=False) + "\n")
    if args.check and not row["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
