"""Audit a compact paint-only component hypothesis outside production search.

    PYTHONPATH=src:. python scripts/bench_cel_component.py \
        --normalizer 54564 --out .bench/compact-component

Generation uses source evidence only. Human scoring runs after all proposals
have been generated and checked. This diagnostic has its own time allowance;
it does not establish selection within the operation's 60-second budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

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
from vectrify.refine.cel_plan.score import (
    foreground_mask,
    measurements,
    render,
    svg_metrics,
)
from vectrify.refine.cel_plan.search import State


def digest(value):
    return hashlib.sha256(value).hexdigest()


def run(case, output, *, seconds=180, normalizer=None):
    started, revision = time.monotonic(), source_hash()
    _, reference = load_case(DATA / case["file"])
    truth = np.asarray(reference, dtype=np.float32) / 255
    mask = foreground_mask(truth)
    if digest(mask.tobytes()) != case["mask_sha256"]:
        raise ValueError("Fixture mask differs from frozen benchmark")
    options, work = Options(refine=False), Work.start(seconds)
    evidence = collect(reference, None, options, work)
    graph = build(evidence, work=work)
    policy = Policy.from_evidence(evidence, graph)
    fallback, _ = export(evidence, evidence.labels, options, work, conservative=True)
    baseline = policy.evaluate(fallback)
    if not baseline.valid:
        raise ValueError(f"Source-only baseline is invalid: {baseline.rejections}")
    policy.establish(baseline)
    actual_normalizer = baseline.cost if normalizer is None else normalizer
    output.mkdir(parents=True, exist_ok=True)
    (output / "fallback.svg").write_text(fallback)
    result, preparation = candidate(
        evidence, graph, policy, options, work, normalizer=actual_normalizer
    )
    rows, factory, status = [], None, "unsupported"
    if result is not None:
        svg, metadata = result
        full = policy.evaluate(svg)
        if not full.valid:
            raise ValueError(f"Source-only initializer is invalid: {full.rejections}")
        partition = Partition.from_metadata(metadata["planning_surfaces"])
        if partition is None:
            raise ValueError("Initializer has no source ownership")
        state = State(
            import_svg(svg),
            svg,
            LocalPolicy(policy).start(svg, full),
            "component-initializer",
            metadata,
            partition=partition,
        )
        factory = CoreCells(Families(evidence, graph, options), options, joint=True)
        (output / "initializer.svg").write_text(svg)
        rows.append(
            {
                "name": "initializer",
                "svg_sha256": digest(svg.encode()),
                "native": full.metrics(),
                **svg_metrics(svg),
            }
        )
        status = "complete"
        try:
            for index, edit in enumerate(factory(state, work)):
                if edit.partition is None or edit.details is None:
                    raise ValueError(
                        "Component proposal lacks ownership or diagnostics"
                    )
                name = f"candidate-{index}"
                svg = export_svg(edit.document)
                # Save rejected proposals too; a missing draft is not a
                # faithful alternative rejected only by admission or ranking.
                (output / f"{name}.svg").write_text(svg)
                full = policy.evaluate(svg)
                local = LocalPolicy(policy).update(
                    state.snapshot, svg, edit.bounds, full.structure
                )
                agreement = max(
                    abs(local.evaluation.terms[k] - v) for k, v in full.terms.items()
                )
                sealed = edit.component is not None and edit.component.validate(
                    state.document,
                    edit.document,
                    partition,
                    edit.partition,
                    edit.ids,
                    edit.bounds,
                    work,
                )
                rows.append(
                    {
                        "name": name,
                        "svg_sha256": digest(svg.encode()),
                        "parameters": edit.parameters,
                        "model": edit.details["core_material_cells"],
                        "component_sealed": sealed,
                        "complete_ownership": edit.partition.follows(partition),
                        "local_full_maximum_term_difference": agreement,
                        "local_native_raster_agreement": local.canvas.matches(
                            render(svg, evidence.source_size)
                        ),
                        "native": full.metrics(),
                        **svg_metrics(svg),
                    }
                )
                print(
                    f"{name}: {rows[-1]['nodes']} nodes; {full.rejections}", flush=True
                )
        except StageInterruptedError:
            status = "interrupted"
        if work.interrupted:
            status = "interrupted"
    generation_seconds = time.monotonic() - started
    # Human geometry is now scoring input only, after proposal discovery ends.
    human_svg, _ = load_case(DATA / case["file"])
    human = render(human_svg, evidence.source_size)
    for row in rows:
        svg = (output / f"{row['name']}.svg").read_text()
        actual = render(svg, evidence.source_size)
        row["human"] = measurements(actual, human, mask, features=case["features"])
        row["passes_numerical_gate"] = (
            row["nodes"] <= case["targets"]["nodes"]
            and row["contours"] <= case["targets"]["contours"]
            and row["human"]["mse"] <= case["targets"]["human_mse"]
        )
        save_render(output / f"{row['name']}.png", actual)
        for feature, (x, y, width, height) in case["features"].items():
            sheet = Image.new("RGBA", (3 * width, height), "white")
            for i, pixels in enumerate((truth, human, actual)):
                crop = pixels[y : y + height, x : x + width]
                image = Image.fromarray((crop * 255).round().astype(np.uint8))
                sheet.alpha_composite(image, (i * width, 0))
            sheet.convert("RGB").save(
                output / f"{row['name']}-{feature.replace(' ', '-')}.png"
            )
    if source_hash() != revision:
        raise ValueError("Algorithm source changed during benchmark")
    report = {
        "case": case["name"],
        "purpose": (
            "paint-budget ablation; diagnostic only; not selected operation output"
        ),
        "source_sha256": revision,
        "reference_rgba_sha256": digest(truth.tobytes()),
        "human_svg_sha256": digest(human_svg.encode()),
        "mask_sha256": digest(mask.tobytes()),
        "settings": {"complexity": 50, "quality": "balanced", "refine": False},
        "normalizer": actual_normalizer,
        "normalizer_source": "source-only-baseline"
        if normalizer is None
        else "override",
        "diagnostic_seconds_limit": seconds,
        "generation_and_validation_seconds": generation_seconds,
        "status": status,
        "policy": policy.metadata(),
        "preparation": preparation,
        "diagnostics": factory.diagnostics if factory is not None else {},
        "rows": rows,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="sword")
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--normalizer", type=float)
    parser.add_argument("--out", type=Path, default=Path(".bench/cel-component"))
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("Seconds must be finite and positive")
    if args.normalizer is not None and (
        not math.isfinite(args.normalizer) or args.normalizer <= 0
    ):
        parser.error("Normalizer must be finite and positive")
    cases = {c["name"]: c for c in json.loads(MANIFEST.read_text())["cases"]}
    if args.case not in cases:
        parser.error(f"Unknown case: {args.case}")
    run(cases[args.case], args.out, seconds=args.seconds, normalizer=args.normalizer)


if __name__ == "__main__":
    main()
