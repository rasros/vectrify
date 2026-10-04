"""Tidy saved projects with every path in a group selected simultaneously.

    uv run python scripts/bench_tidy.py --out .bench/sword.json
    uv run python scripts/bench_tidy.py --nodes '{"layout": false}'

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
import pathops
from PIL import Image
from scipy.ndimage import binary_erosion

from vectrify.document import Editor, Selection, export_svg, load_project, save_project
from vectrify.document.join import curve_path, path_style
from vectrify.image_utils import on_white
from vectrify.operations import Budget, Job, OperationRequest, Permissions, method

FIXTURE = Path(__file__).parent / "bench_data/projects/sword.vectrify"
GROUP = "group_343f776e97514ac39bd469b1143cc40a"
OUTLINE = "object_68f9c655457542db9c945eef1e192b0c"


def load(path=FIXTURE):
    project = json.loads(path.read_text())
    document, _selection = load_project(json.dumps(project["document"]))
    data = project["reference"]["data_url"].split(",", 1)[1]
    image = on_white(Image.open(io.BytesIO(base64.b64decode(data))))
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


def properties(document, group=GROUP, outline=OUTLINE):
    """Containment per fill, uncovered interior, and reflected silhouette error.

    Reflect about the line from the outline's tip to the midpoint of its base.
    Opaque coverage is also rendered alone,
    so an underlying hilt or background cannot hide a transparent blade gap.
    """
    from vectrify.document.transforms import object_matrix
    from vectrify.refine.layout import axis_for, reflection

    def shape(oid):
        return curve_path(
            document.geometry_for(oid),
            path_style(document, document.element(oid))["fill-rule"],
        ).transform(*object_matrix(document, oid))

    interior = shape(outline)
    fills = [
        e.id
        for e in document.element(group).children
        if e.tag == "path" and path_style(document, e)["fill"] != "none"
    ]
    shapes = {oid: shape(oid) for oid in fills}
    union = pathops.Path()
    for path in shapes.values():
        union = pathops.op(union, path, pathops.PathOp.UNION)
    a, c, b, d, e, f = reflection(axis_for(document, outline))
    reflected = interior.transform(a, b, c, d, e, f)
    spill = {
        oid: abs(pathops.op(s, interior, pathops.PathOp.DIFFERENCE).area)
        for oid, s in shapes.items()
    }
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
    return {
        "outside_area": sum(spill.values()),
        "outside_each": spill,
        "gap_area": abs(pathops.op(interior, union, pathops.PathOp.DIFFERENCE).area),
        "symmetry_area": abs(pathops.op(interior, reflected, pathops.PathOp.XOR).area),
        "interior_area": abs(interior.area),
        "transparent_pixels": int(((blade[:, :, 3] < 255) & mask).sum()),
    }


def run(path=FIXTURE, settings=None, group=GROUP, outline=OUTLINE):
    document, image = load(path)
    before = properties(document, group, outline)
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
    after = properties(final, group, outline)
    target = np.asarray(image, float)
    mse = [
        float(((_render(d, white=True)[:, :, :3].astype(float) - target) ** 2).mean())
        for d in (document, final)
    ]
    return final, {
        "case": "sword-blade",
        "simultaneous": True,
        "paths": len(document.element(group).children),
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
