"""Audit source-cut composition and retained graph storage on frozen artwork.

Generation uses source evidence only. Human scoring follows discovery; this
diagnostic's work allowance does not prove selection in the native operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from scripts.bench_cel_planned import (
    DATA,
    MANIFEST,
    load_case,
    save_render,
    source_hash,
)
from vectrify.document import export_svg, import_svg
from vectrify.refine.cel_plan.core_cells import CoreCells
from vectrify.refine.cel_plan.core_materials import candidate
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.export import export
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.proposals import MAX_BRANCH_BYTES, Operators
from vectrify.refine.cel_plan.score import foreground_mask, measurements, render
from vectrify.refine.cel_plan.search import State
from vectrify.refine.cel_plan.source_ridges import SourceRidges


def run(case, output, *, seconds=180, normalizer=None):
    began, revision = time.monotonic(), source_hash()
    _, reference = load_case(DATA / case["file"])
    mask = foreground_mask(np.asarray(reference, np.float32) / 255)
    if hashlib.sha256(mask.tobytes()).hexdigest() != case["mask_sha256"]:
        raise ValueError("Fixture mask differs from frozen benchmark")
    options, work = Options(refine=False), Work.start(seconds)
    evidence = collect(reference, None, options, work)
    graph = build(evidence, work=work)
    policy = Policy.from_evidence(evidence, graph)
    fallback, _ = export(evidence, evidence.labels, options, work, conservative=True)
    baseline = policy.evaluate(fallback)
    if not baseline.valid:
        raise ValueError(f"Source-only fallback is invalid: {baseline.rejections}")
    policy.establish(baseline)
    result, preparation = candidate(
        evidence,
        graph,
        policy,
        options,
        work,
        normalizer=baseline.cost if normalizer is None else normalizer,
    )
    if result is None:
        raise ValueError(f"No source-only initializer: {preparation}")
    svg, metadata = result
    full = policy.evaluate(svg)
    if not full.valid:
        raise ValueError(f"Source-only initializer is invalid: {full.rejections}")
    partition = Partition.from_metadata(metadata["planning_surfaces"])
    if partition is None:
        raise ValueError("Initializer has no complete source ownership")
    state = State(
        import_svg(svg),
        svg,
        LocalPolicy(policy).start(svg, full),
        hashlib.sha256(svg.encode()).hexdigest(),
        metadata,
        partition=partition,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "initializer.svg").write_text(svg)
    operators = Operators(evidence, graph, options)
    phases = [("initial", state)]
    coarse = next(
        CoreCells(Families(evidence, graph, options), options)(state, work), None
    )
    if coarse is not None:
        if coarse.partition is None:
            raise ValueError("Material parent has no complete source ownership")
        operators.validate_partition(coarse.partition, work)
        svg = export_svg(coarse.document)
        full = policy.evaluate(svg)
        if not full.valid:
            raise ValueError(f"Material parent is invalid: {full.rejections}")
        (output / "coarse-material.svg").write_text(svg)
        phases.append(
            (
                "coarse-material",
                State(
                    coarse.document,
                    svg,
                    LocalPolicy(policy).start(svg, full),
                    hashlib.sha256(svg.encode()).hexdigest(),
                    {
                        **state.details,
                        **(coarse.details or {}),
                        "planning_surfaces": coarse.partition.metadata(),
                    },
                    partition=coarse.partition,
                ),
            )
        )
    rows, diagnostics, resolution_errors = [], {}, {}
    phase = "initial"

    def resolve(partition, current_work):
        try:
            return operators.branch(partition, current_work)
        except ValueError as exc:
            resolution_errors.setdefault(phase, []).append(str(exc))
            raise

    for phase, current in phases:
        if work.interrupted:
            diagnostics[phase] = {"status": "not-started", "interrupted": True}
            break
        if current.partition is None:
            raise ValueError("Source-cut parent has no complete source ownership")
        branch = operators.branch(current.partition, work)
        factory = SourceRidges(branch.evidence, branch.graph, options, resolver=resolve)
        try:
            for index, edit in enumerate(factory.proposals(current, work)):
                if edit.partition is None or edit.details is None:
                    raise ValueError("Source proposal lacks ownership or diagnostics")
                operators.validate_partition(edit.partition, work)
                svg = export_svg(edit.document)
                full = policy.evaluate(svg)
                local = LocalPolicy(policy).update(
                    current.snapshot, svg, edit.bounds, full.structure
                )
                actual = render(svg, evidence.source_size)
                name = f"{phase}-{index}"
                (output / f"{name}.svg").write_text(svg)
                save_render(output / f"{name}.png", actual)
                rows.append(
                    {
                        "name": name,
                        "phase": phase,
                        "native": full.metrics(),
                        "source_ridge": edit.details["source_ridge"],
                        "ink_replacement": edit.details["ink_replacement"],
                        "complete_ownership": edit.partition.follows(current.partition),
                        "local_native_raster_agreement": local.canvas.matches(actual),
                        "local_full_maximum_term_difference": max(
                            abs(local.evaluation.terms[k] - v)
                            for k, v in full.terms.items()
                        ),
                    }
                )
                print(f"{name}: {full.structure['nodes']} nodes", flush=True)
        except StageInterruptedError:
            pass
        diagnostics[phase] = {
            **factory.diagnostics,
            "rim": factory.rim_diagnostics,
            "restoration": factory.restoration_rejections,
            "underpaint": factory.underpaint_rejections,
            "interrupted": work.interrupted,
        }
    generation_seconds = time.monotonic() - began
    discovery_interrupted = work.interrupted
    # Human geometry cannot affect discovery, admission or graph resolution.
    human_svg, _ = load_case(DATA / case["file"])
    human = render(human_svg, evidence.source_size)
    for row in rows:
        actual = render(
            (output / f"{row['name']}.svg").read_text(), evidence.source_size
        )
        row["human"] = measurements(actual, human, mask, features=case["features"])
    if revision != source_hash():
        raise ValueError("Algorithm sources changed during the diagnostic")
    report = {
        "case": case["name"],
        "source_sha256": revision,
        "mask_sha256": case["mask_sha256"],
        "reference_rgba_sha256": hashlib.sha256(
            np.asarray(reference).tobytes()
        ).hexdigest(),
        "human_svg_sha256": hashlib.sha256(human_svg.encode()).hexdigest(),
        "settings": {"complexity": 50, "quality": "balanced", "refine": False},
        "seconds_limit": seconds,
        "normalizer": baseline.cost if normalizer is None else normalizer,
        "generation_and_validation_seconds": generation_seconds,
        "status": "interrupted" if discovery_interrupted else "complete",
        "policy": policy.metadata(),
        "preparation": preparation,
        "graph_cache_limit": MAX_BRANCH_BYTES,
        "original_graph_bytes": operators._graph_bytes(graph),
        "graph_accounting": operators.schedule_diagnostics,
        "diagnostics": diagnostics,
        "resolution_errors": resolution_errors,
        "rows": rows,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="sword")
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--normalizer", type=float)
    parser.add_argument("--out", type=Path, default=Path(".bench/cel-source-graphs"))
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    if args.normalizer is not None and (
        not math.isfinite(args.normalizer) or args.normalizer <= 0
    ):
        parser.error("--normalizer must be finite and positive")
    cases = {c["name"]: c for c in json.loads(MANIFEST.read_text())["cases"]}
    if args.case not in cases:
        parser.error(f"Unknown case: {args.case}")
    run(cases[args.case], args.out, seconds=args.seconds, normalizer=args.normalizer)


if __name__ == "__main__":
    main()
