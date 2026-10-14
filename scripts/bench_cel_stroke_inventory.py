"""Inspect editable outline coverage from the original source, without a human target.

Example: PYTHONPATH=src:. python scripts/bench_cel_stroke_inventory.py --case sword
--candidate .bench/cel-source-junction-band-plans-final/candidate-11.project.json
--out .bench/cel-stroke-inventory --crop 310 1640 420 1870
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

from scripts.bench_cel_planned import MANIFEST
from scripts.cel_bench import finish, prepare
from vectrify.document import export_svg, load_project
from vectrify.document.join import path_style
from vectrify.document.redraw import root_matrix
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.stroke_inventory import StrokeInventory


def run(case, candidate, output, *, seconds=120, crop=None):
    bench = prepare(case, seconds, policy=False)
    reference, work = bench.reference, bench.work
    guard = Operators(
        bench.evidence, bench.graph, bench.options, filled_bands=True
    ).band_guard(work)
    if guard is None:
        raise ValueError("Original source has no bounded independent ink bank")
    data = candidate.read_bytes()
    document, _ = load_project(data.decode())
    report = StrokeInventory(guard).observe(document, work)
    report.update(
        {
            "source_sha256": bench.revision,
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_raster_sha256": hashlib.sha256(bench.raster).hexdigest(),
            "candidate_sha256": hashlib.sha256(data).hexdigest(),
            "case": case["name"],
            "candidate": str(candidate),
            "release_evidence": False,
        }
    )
    overlay = ET.Element(
        "svg", {"width": str(reference.width), "height": str(reference.height)}
    )
    for oid in report["stroke_objects"]:
        style = path_style(document, document.element(oid))
        ET.SubElement(
            overlay,
            "path",
            {
                "d": document.geometry_for(oid).path_data(),
                "transform": "matrix("
                + " ".join(map(str, root_matrix(document, oid)))
                + ")",
                "fill": "none",
                "stroke": "#1495e4",
                "stroke-width": "0.6",
                "stroke-linecap": style["stroke-linecap"],
                "stroke-linejoin": style["stroke-linejoin"],
            },
        )
    profiles = guard.original_profiles(work=work)
    for row in report["profiles"]:
        points = guard.source_breaks(profiles[row["profile"]], work=work).points
        for point in points[row["missing_indices"]]:
            ET.SubElement(
                overlay,
                "circle",
                {
                    "cx": str(point[0]),
                    "cy": str(point[1]),
                    "r": "0.75",
                    "fill": "#e91e63",
                },
            )
    for shared in report["literal_shared_ports"]:
        x, y = shared["point"]
        ET.SubElement(
            overlay,
            "circle",
            {"cx": str(x), "cy": str(y), "r": "1.5", "fill": "#00a65a"},
        )
    actual = Image.fromarray(
        np.rint(render(export_svg(document), reference.size) * 255).astype(np.uint8)
    )
    markup = Image.fromarray(
        np.rint(
            render(ET.tostring(overlay, encoding="unicode"), reference.size) * 255
        ).astype(np.uint8)
    )
    background = Image.new("RGBA", reference.size, "#fafafa")
    background.alpha_composite(actual)
    background.alpha_composite(markup)
    finish(output / "report.json", report, bench.revision)
    background.save(output / "overview.png")
    if crop is not None:
        background.crop(crop).resize(
            ((crop[2] - crop[0]) * 3, (crop[3] - crop[1]) * 3)
        ).save(output / "detail.png")
    print(
        json.dumps(
            {
                "qualified": report["qualified_samples"],
                "missing": report["missing_samples"],
                "stroke_objects": len(report["stroke_objects"]),
                "stroke_contours": report["stroke_contours"],
                "supported_fraction": report["supported_fraction"],
                "shared_ports": len(report["literal_shared_ports"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--crop", type=int, nargs=4)
    args = parser.parse_args()
    cases = json.loads(MANIFEST.read_text())["cases"]
    selected = next(case for case in cases if case["name"] == args.case)
    run(selected, args.candidate, args.out, seconds=args.seconds, crop=args.crop)
