"""Replay a captured material proposal with source-only stroke co-planning.

The input stores ancestor/planned .project.json documents, corresponding
partition/details JSON and proposal.json (ids, parent, component_parent,
parameters, operator). Project documents preserve exact component identities.
This diagnostic does not establish operation scheduling or release quality.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.bench_cel_planned import DATA, MANIFEST, source_hash
from vectrify.document import export_svg, load_project, save_project
from vectrify.document.join import transformed_geometry
from vectrify.document.redraw import root_matrix
from vectrify.project_file import decode_source
from vectrify.refine.cel_plan.component_edits import ComponentEdit
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.frontier import Frontier
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Options, Work
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.proposals import Operators, bounds
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import Proposal, State


def run(case, captured, output, *, seconds=180):
    revision = source_hash()
    project = json.loads(decode_source((DATA / case["file"]).read_bytes()))
    raster = base64.b64decode(project["reference"]["data_url"].split(",", 1)[1])
    with Image.open(io.BytesIO(raster)) as image:
        reference = image.convert("RGBA")
    work, options = Work.start(seconds), Options()
    evidence = collect(reference, None, options, work)
    graph = build(evidence)
    policy = Policy.from_evidence(evidence, graph)
    local = LocalPolicy(policy)
    ancestor, _ = load_project((captured / "ancestor.project.json").read_text())
    planned, _ = load_project((captured / "planned.project.json").read_text())
    old = Partition.from_metadata(
        json.loads((captured / "ancestor-partition.json").read_text())
    )
    part = Partition.from_metadata(
        json.loads((captured / "planned-partition.json").read_text())
    )
    if old is None or part is None:
        raise ValueError("Captured proposal requires complete ancestor ownership")
    spec = json.loads((captured / "proposal.json").read_text())
    initial = export_svg(ancestor)
    state = State(
        ancestor,
        initial,
        local.start(initial, policy.evaluate(initial)),
        spec["parent"],
        json.loads((captured / "ancestor-details.json").read_text()),
        partition=old,
    )
    ids = tuple(spec["ids"])
    edit = Proposal(
        spec["operator"],
        ids,
        tuple(spec["parameters"]),
        state.key,
        planned,
        bounds(ancestor, planned, ids),
        details=json.loads((captured / "planned-details.json").read_text()),
        partition=part,
        component=ComponentEdit.bind(ancestor, old, spec["component_parent"], work),
    )
    operators = Operators(evidence, graph, options, filled_bands=True)
    operators.validate_partition(old, work)
    operators.validate_partition(part, work)
    branch = operators.branch(old, work)
    parent_svg = export_svg(planned)
    baseline = policy.evaluate(parent_svg)
    if not baseline.valid:
        raise ValueError("Material proposal must be native-valid")
    frontier = Frontier(policy)
    frontier.add(initial, "ancestor")
    frontier.freeze_normalizer()
    frontier.add(parent_svg, "material")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, proposed in enumerate(branch.families.band_planner(state, edit, work)):
        svg = export_svg(proposed.document)
        actual = render(svg, reference.size)
        native = policy.evaluate(svg)
        operators.validate_partition(proposed.partition, work)
        proposed.component.validate(
            ancestor,
            proposed.document,
            old,
            proposed.partition,
            proposed.ids,
            proposed.bounds,
            work,
        )
        updated = local.update(state.snapshot, svg, proposed.bounds, native.structure)
        difference = max(
            abs(updated.evaluation.terms[k] - v) for k, v in native.terms.items()
        )
        restored, _ = load_project(save_project(proposed.document))
        if not updated.canvas.matches(actual) or difference > 2e-7:
            raise ValueError("Co-planned local/native scoring disagrees")
        if not np.array_equal(actual, render(export_svg(restored), reference.size)):
            raise ValueError("Co-planned project round trip changed pixels")
        converted = proposed.details.get(
            "planned_band_strokes", [proposed.details["planned_band_stroke"]]
        )
        changed_ids = {item["id"] for item in converted}
        fitted = proposed.details["planned_band_stroke"].get("source_fit", {})
        caps = fitted.get("caps", ())
        continuation = fitted.get("continuation")
        changed_ids.update(witness["id"] for witness in caps)
        if continuation is not None:
            changed_ids.add(continuation["material"])
        for element in planned.elements():
            if (
                element.tag == "path"
                and element.id not in changed_ids
                and (
                    proposed.document.element(element.id) != element
                    or proposed.document.geometry_for(element.id)
                    != planned.geometry_for(element.id)
                )
            ):
                raise ValueError("Co-planning changed independent material geometry")
        # A later split can convert an earlier split's exact residual path.
        # Reconstruct each residual from its own parent, in declaration order.
        residuals = {
            element.id: planned.geometry_for(element.id)
            for element in planned.elements()
            if element.tag == "path"
        }
        for item in converted:
            original = residuals[item["id"]]
            contour_index = item.get(
                "contour",
                max(
                    range(len(original.subpaths)),
                    key=lambda i: len(original.subpaths[i].nodes),
                ),
            )
            residuals[item["marks"]] = replace(
                original,
                subpaths=tuple(
                    s for i, s in enumerate(original.subpaths) if i != contour_index
                ),
            )
        for item in converted:
            if item["marks"] in changed_ids:
                continue
            if (
                proposed.document.geometry_for(item["marks"]).path_data()
                != residuals[item["marks"]].path_data()
            ):
                raise ValueError("Co-planning changed residual shade or mark geometry")
        for oid in {witness["id"] for witness in caps}:
            original = planned.geometry_for(oid)
            actual_geometry = proposed.document.geometry_for(oid)
            subs = list(actual_geometry.subpaths)
            native_geometry = transformed_geometry(
                actual_geometry, root_matrix(proposed.document, oid)
            )
            for witness in (w for w in caps if w["id"] == oid):
                i = witness["contour"]
                at = 0 if witness["end"] == "start" else -1
                if not np.allclose(
                    native_geometry.subpaths[i].nodes[at].endpoint,
                    witness["to"],
                    rtol=0,
                    atol=1e-8,
                ):
                    raise ValueError("Recovered cap disagrees with its source witness")
                nodes = list(subs[i].nodes)
                nodes[at] = replace(
                    nodes[at],
                    values=(
                        *nodes[at].values[:-2],
                        *original.subpaths[i].nodes[at].endpoint,
                    ),
                )
                subs[i] = replace(subs[i], nodes=tuple(nodes))
            if replace(
                actual_geometry, subpaths=tuple(subs)
            ) != original or proposed.document.element(oid) != planned.element(oid):
                raise ValueError("Cap recovery changed unrelated nodes or stroke style")
        if continuation is not None:
            oid = continuation["material"]
            original = planned.geometry_for(oid)
            actual_geometry = proposed.document.geometry_for(oid)
            if (
                actual_geometry.subpaths[: len(original.subpaths)] != original.subpaths
                or proposed.document.element(oid) != planned.element(oid)
                or not continuation["native_alpha_exact"]
                or not continuation["native_footprint_exact"]
            ):
                raise ValueError(
                    "Continuation changed original material or native alpha"
                )
        name = f"candidate-{index}"
        frontier.add(svg, name, proposed.details)
        (output / f"{name}.svg").write_text(svg)
        (output / f"{name}.project.json").write_text(save_project(proposed.document))
        (output / f"{name}-partition.json").write_text(
            json.dumps(proposed.partition.metadata())
        )
        Image.fromarray(np.rint(actual * 255).astype(np.uint8)).save(
            output / f"{name}.png"
        )
        rows.append(
            {
                "file": f"{name}.svg",
                "valid": native.valid,
                "metrics": native.metrics(),
                "details": proposed.details["planned_band_stroke"],
                "converted_bands": converted,
                "local_term_max": difference,
                "atom_cuts": len(proposed.partition.atoms.cuts),
                "complete_ownership_validated": True,
                "complete_component_validated": True,
                "project_roundtrip_equal": True,
                "all_other_paths_exact": True,
                "source_fit": fitted,
                "objective_delta": {
                    str(c): native.objective(c, frontier.normalizer)
                    - baseline.objective(c, frontier.normalizer)
                    for c in (0, 25, 50, 75, 100)
                },
                "beats_material_at_every_slider": all(
                    native.objective(c, frontier.normalizer)
                    < baseline.objective(c, frontier.normalizer)
                    for c in range(101)
                ),
            }
        )
    if len({row["file"] for row in rows}) != len(rows):
        raise ValueError("Candidate artifact names must be unique")
    if revision != source_hash():
        raise ValueError("Algorithm sources changed during co-planning benchmark")
    report = {
        "source_sha256": revision,
        "case": case["name"],
        "source_raster_sha256": hashlib.sha256(raster).hexdigest(),
        "captured_material_sha256": hashlib.sha256(parent_svg.encode()).hexdigest(),
        "status": "interrupted" if work.interrupted else "complete",
        "parent": baseline.metrics(),
        "rows": rows,
        "diagnostics": branch.families.band_planner.diagnostics,
        "release_evidence": False,
        "scope": "offline captured-proposal replay; no human input",
        "normalizer": frontier.normalizer,
        "selection": {
            str(c): {
                "label": frontier.select(c).label,
                "width": frontier.select(c)
                .metrics.get("planned_band_stroke", {})
                .get("width"),
            }
            for c in (0, 25, 50, 75, 100)
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="sword")
    parser.add_argument("--captured", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=180)
    args = parser.parse_args()
    case = next(
        (
            c
            for c in json.loads(MANIFEST.read_text())["cases"]
            if c["name"] == args.case
        ),
        None,
    )
    if case is None or not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("Choose an existing case and a finite positive time budget")
    report = run(case, args.captured, args.out, seconds=args.seconds)
    print(
        json.dumps({"status": report["status"], "diagnostics": report["diagnostics"]})
    )


if __name__ == "__main__":
    main()
