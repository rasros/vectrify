"""Source-only whole-owner stroke inversion on a preserved owned drawing.

Example: PYTHONPATH=src:. python scripts/bench_cel_band_strokes.py --case sword
--parent .bench/cel-filled-band-owned-parent --out .bench/cel-band-strokes
The parent directory contains parent.svg, partition.json and parent-details.json.
This is an offline diagnostic, not an operation scheduling or release result.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.bench_cel_planned import DATA, MANIFEST, source_hash
from vectrify.document import export_svg, import_svg, load_project, save_project
from vectrify.project_file import decode_source
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Options, Work
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import State


def run(case, parent: Path, output: Path, *, seconds=180):
    revision = source_hash()
    project = json.loads(decode_source((DATA / case["file"]).read_bytes()))
    # Decode the original raster only; never parse the human drawing.
    image_bytes = base64.b64decode(project["reference"]["data_url"].split(",", 1)[1])
    with Image.open(io.BytesIO(image_bytes)) as image:
        reference = image.convert("RGBA")
    work, options = Work.start(seconds), Options()
    evidence = collect(reference, None, options, work)
    graph = build(evidence)
    policy = Policy.from_evidence(evidence, graph)
    svg = (parent / "parent.svg").read_text()
    partition = Partition.from_metadata(
        json.loads((parent / "partition.json").read_text())
    )
    if partition is None:
        raise ValueError("Band comparison requires complete preserved ownership")
    document = import_svg(svg)
    partition.validate(document)
    baseline = policy.evaluate(svg)
    if not baseline.valid:
        raise ValueError("Band comparison requires a native-valid preserved parent")
    state = State(
        document,
        svg,
        LocalPolicy(policy).start(svg, baseline),
        "preserved-parent",
        json.loads((parent / "parent-details.json").read_text()),
        partition=partition,
    )
    operators = Operators(evidence, graph, options, filled_bands=True)
    operators.validate_partition(partition, work)
    branch = operators.branch(partition, work)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, edit in enumerate(branch.bands(state, work)):
        candidate = export_svg(edit.document)
        native = policy.evaluate(candidate)
        operators.validate_partition(edit.partition, work)
        edit.component.validate(
            document,
            edit.document,
            partition,
            edit.partition,
            edit.ids,
            edit.bounds,
            work,
        )
        actual = render(candidate, reference.size)
        restored, _ = load_project(save_project(edit.document))
        if not np.array_equal(actual, render(export_svg(restored), reference.size)):
            raise ValueError("Band candidate changed after project round trip")
        local = LocalPolicy(policy).update(
            state.snapshot, candidate, edit.bounds, native.structure
        )
        discrepancy = max(
            abs(local.evaluation.terms[k] - v) for k, v in native.terms.items()
        )
        if not local.canvas.matches(actual) or discrepancy > 2e-7:
            raise ValueError("Band candidate local/native scoring disagrees")
        filename = f"candidate-{index}.svg"
        (output / filename).write_text(candidate)
        Image.fromarray(np.rint(actual * 255).astype(np.uint8)).save(
            output / f"candidate-{index}.png"
        )
        rows.append(
            {
                "file": filename,
                "ids": edit.ids,
                "valid": native.valid,
                "rejections": native.rejections,
                "metrics": native.metrics(),
                "details": edit.details,
                "local_term_max": discrepancy,
                "complete_ownership_validated": True,
                "complete_component_validated": True,
                "project_roundtrip_equal": True,
            }
        )
    if source_hash() != revision:
        raise ValueError("Band benchmark algorithm sources changed during execution")
    report = {
        "source_sha256": revision,
        "parent_sha256": hashlib.sha256(svg.encode()).hexdigest(),
        "source_raster_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "status": "interrupted" if work.interrupted else "complete",
        "case": case["name"],
        "parent": baseline.metrics(),
        "rows": rows,
        "diagnostics": branch.bands.diagnostics,
        "release_evidence": False,
        "scope": "offline whole-owner inversion; no human drawing input",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="sword")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=180)
    args = parser.parse_args()
    cases = json.loads(MANIFEST.read_text())["cases"]
    case = next((case for case in cases if case["name"] == args.case), None)
    if case is None or not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("Choose an existing case and a finite positive time budget")
    report = run(case, args.parent, args.out, seconds=args.seconds)
    print(
        json.dumps({"status": report["status"], "diagnostics": report["diagnostics"]})
    )


if __name__ == "__main__":
    main()
