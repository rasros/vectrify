"""Audit compact paint/ink component hypotheses outside production search.

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
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.bench_cel_pairs import finite_json
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
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.score import (
    foreground_mask,
    measurements,
    render,
    svg_metrics,
)
from vectrify.refine.cel_plan.search import State
from vectrify.refine.cel_plan.source_strokes import SourceStrokes


def digest(value):
    return hashlib.sha256(value).hexdigest()


def offered(
    state,
    factory,
    operators,
    policy,
    options,
    work,
    compose_materials,
    composition_grouping="ward",
    composition_diagnostics=None,
    composition_layout="regions",
):
    """Only valid source-ink parents seed a bounded material composition pool."""
    parents = 0
    proposals = (
        factory.proposals(state, work)
        if isinstance(factory, SourceStrokes)
        else factory(state, work)
    )
    for edit in proposals:
        yield state, edit
        if not compose_materials or parents >= 2 or work.interrupted:
            continue
        svg = export_svg(edit.document)
        full = policy.evaluate(svg)
        if not full.valid or edit.partition is None:
            continue
        parents += 1
        parent = State(
            edit.document,
            svg,
            LocalPolicy(policy).start(svg, full),
            f"source-stroke-parent-{parents}",
            {**state.details, **(edit.details or {})},
            partition=edit.partition,
        )
        branch = operators.branch(edit.partition, work)
        materials = CoreCells(
            branch.families,
            options,
            joint=True,
            grouping=composition_grouping,
            boundary_fit="curve",
            layout=composition_layout,
        )
        if composition_diagnostics is not None:
            composition_diagnostics.append(
                {
                    "parent_svg_sha256": digest(svg.encode()),
                    "diagnostics": materials.diagnostics,
                }
            )
        for composed in materials(parent, work):
            yield (
                parent,
                replace(
                    composed,
                    details={
                        **(composed.details or {}),
                        "retained_source_strokes": (edit.details or {}).get(
                            "source_strokes"
                        ),
                    },
                ),
            )


def run(
    case,
    output,
    *,
    seconds=180,
    normalizer=None,
    grouping="static",
    boundary_fit="polygon",
    ink_support="paired",
    proposal="core-cells",
    compose_materials=False,
    boundary_contacts=False,
    composition_grouping="ward",
    composition_layout="regions",
):
    started, revision = time.monotonic(), source_hash()
    if case.get("paired"):
        from scripts.cel_pairs import pair

        _, _, reference = pair(
            case, "clean", case["long_side"], case["composition_opacity"]
        )
    else:
        _, reference = load_case(DATA / case["file"])
    truth = np.asarray(reference, dtype=np.float32) / 255
    mask = foreground_mask(truth)
    if digest(mask.tobytes()) != case["mask_sha256"]:
        raise ValueError("Fixture mask differs from frozen benchmark")
    options, work = Options(refine=False), Work.start(seconds)
    evidence = collect(reference, None, options, work)
    graph = build(evidence, work=work)
    policy = Policy.from_evidence(evidence, graph)
    fallback, fallback_metadata = export(
        evidence, evidence.labels, options, work, conservative=True
    )
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
    if result is None and proposal == "source-strokes":
        result = fallback, fallback_metadata
        preparation = {**preparation, "stroke_parent": "conservative-fallback"}
    rows, factory, status = [], None, "unsupported"
    composition_diagnostics = []
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
        operators = Operators(evidence, graph, options)
        factory = (
            SourceStrokes(
                evidence,
                graph,
                options,
                resolver=operators.branch,
                boundary_contacts=boundary_contacts,
            )
            if proposal == "source-strokes"
            else CoreCells(
                Families(evidence, graph, options),
                options,
                joint=True,
                grouping=grouping,
                boundary_fit=boundary_fit,
                ink_support=ink_support,
            )
        )
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
            for index, (parent, edit) in enumerate(
                offered(
                    state,
                    factory,
                    operators,
                    policy,
                    options,
                    work,
                    compose_materials,
                    composition_grouping,
                    composition_diagnostics,
                    composition_layout,
                )
            ):
                if edit.partition is None or edit.details is None:
                    raise ValueError(
                        "Component proposal lacks ownership or diagnostics"
                    )
                if proposal == "source-strokes":
                    operators.validate_partition(edit.partition, work)
                name = f"candidate-{index}"
                svg = export_svg(edit.document)
                # Save rejected proposals too; a missing draft is not a
                # faithful alternative rejected only by admission or ranking.
                (output / f"{name}.svg").write_text(svg)
                full = policy.evaluate(svg)
                local = LocalPolicy(policy).update(
                    parent.snapshot, svg, edit.bounds, full.structure
                )
                agreement = max(
                    abs(local.evaluation.terms[k] - v) for k, v in full.terms.items()
                )
                sealed = edit.component is not None and edit.component.validate(
                    parent.document,
                    edit.document,
                    parent.partition,
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
                        "parent_svg_sha256": digest(parent.svg.encode()),
                        "stage": edit.operator,
                        "model": edit.details.get(
                            "core_material_cells", edit.details.get("source_strokes")
                        ),
                        "retained_source_strokes": edit.details.get(
                            "retained_source_strokes"
                        ),
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
    if case.get("paired"):
        from scripts.cel_pairs import pair

        human_svg, _, _ = pair(
            case, "clean", case["long_side"], case["composition_opacity"]
        )
    else:
        human_svg, _ = load_case(DATA / case["file"])
    human = render(human_svg, evidence.source_size)
    for row in rows:
        svg = (output / f"{row['name']}.svg").read_text()
        actual = render(svg, evidence.source_size)
        metric = "clean" if case.get("paired") else "human"
        row[metric] = measurements(actual, human, mask, features=case["features"])
        if case.get("paired"):
            from scripts.bench_lines import score

            row["lines"] = score(human_svg, svg, *evidence.source_size, "clean")
        row["passes_numerical_gate"] = (
            None
            if "targets" not in case
            else (
                row["nodes"] <= case["targets"]["nodes"]
                and row["contours"] <= case["targets"]["contours"]
                and row[metric]["mse"] <= case["targets"]["human_mse"]
            )
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
            "source component comparison; diagnostic only; "
            "not selected operation output"
        ),
        "source_sha256": revision,
        "reference_rgba_sha256": digest(truth.tobytes()),
        "clean_svg_sha256" if case.get("paired") else "human_svg_sha256": digest(
            human_svg.encode()
        ),
        "evaluation_target": "paired-clean" if case.get("paired") else "human-redraw",
        "target_variant": {
            "long_side": case["long_side"],
            "composition_opacity": case["composition_opacity"],
        }
        if case.get("paired")
        else None,
        "family": case.get("family"),
        "split": case.get("set"),
        "heldout_evaluation": False,
        "mask_sha256": digest(mask.tobytes()),
        "settings": {"complexity": 50, "quality": "balanced", "refine": False},
        "normalizer": actual_normalizer,
        "proposal": proposal,
        "compose_materials": compose_materials,
        "boundary_contacts": boundary_contacts,
        "material_parent_limit": 2 if compose_materials else 0,
        "source_proposal_allowance": "full-diagnostic"
        if proposal == "source-strokes"
        else None,
        "grouping": grouping if proposal == "core-cells" else None,
        "composition_grouping": composition_grouping if compose_materials else None,
        "composition_layout": composition_layout if compose_materials else None,
        "composition_diagnostics": composition_diagnostics,
        "boundary_fit": boundary_fit if proposal == "core-cells" else None,
        "ink_support": ink_support if proposal == "core-cells" else "source-drawn",
        "normalizer_source": "source-only-baseline"
        if normalizer is None
        else "override",
        "diagnostic_seconds_limit": seconds,
        "generation_and_validation_seconds": generation_seconds,
        "status": status,
        "policy": policy.metadata(),
        "preparation": preparation,
        "diagnostics": factory.diagnostics if factory is not None else {},
        "cut_diagnostics": factory.cutter.diagnostics
        if isinstance(factory, SourceStrokes)
        else None,
        "restoration_rejections": factory.restoration_rejections
        if isinstance(factory, SourceStrokes)
        else None,
        "rows": rows,
    }
    report = finite_json(report)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="sword")
    parser.add_argument(
        "--proposal", choices=("core-cells", "source-strokes"), default="core-cells"
    )
    parser.add_argument("--compose-materials", action="store_true")
    parser.add_argument("--boundary-contacts", action="store_true")
    parser.add_argument(
        "--pair", help="Repository tuning artwork; clean 192px, half opacity"
    )
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--normalizer", type=float)
    parser.add_argument(
        "--grouping", choices=("static", "ward", "paint-fit"), default="static"
    )
    parser.add_argument(
        "--composition-grouping", choices=("ward", "paint-fit"), default="ward"
    )
    parser.add_argument(
        "--composition-layout", choices=("regions", "planes"), default="regions"
    )
    parser.add_argument(
        "--boundary-fit", choices=("polygon", "curve"), default="polygon"
    )
    parser.add_argument(
        "--ink-support", choices=("paired", "connected"), default="paired"
    )
    parser.add_argument("--out", type=Path, default=Path(".bench/cel-component"))
    args = parser.parse_args()
    if args.compose_materials and args.proposal != "source-strokes":
        parser.error("Material composition requires --proposal source-strokes")
    if args.boundary_contacts and args.proposal != "source-strokes":
        parser.error("Boundary contacts require --proposal source-strokes")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("Seconds must be finite and positive")
    if args.normalizer is not None and (
        not math.isfinite(args.normalizer) or args.normalizer <= 0
    ):
        parser.error("Normalizer must be finite and positive")
    cases = {c["name"]: c for c in json.loads(MANIFEST.read_text())["cases"]}
    if args.pair:
        from scripts.bench_cel_pairs import feature_boxes
        from scripts.cel_pairs import cases as paired_cases
        from scripts.cel_pairs import pair

        paired = {c["name"]: c for c in paired_cases()}
        if args.pair not in paired:
            parser.error(f"Unknown tuning artwork: {args.pair}")
        case: dict = dict(
            paired[args.pair], paired=True, long_side=192, composition_opacity=0.5
        )
        _, clean, _ = pair(case, "clean", 192, 0.5)
        case["features"] = feature_boxes(case, clean.size)
        case["mask_sha256"] = digest(
            foreground_mask(np.asarray(clean, np.float32) / 255).tobytes()
        )
    elif args.case not in cases:
        parser.error(f"Unknown case: {args.case}")
    else:
        case = cases[args.case]
    run(
        case,
        args.out,
        seconds=args.seconds,
        normalizer=args.normalizer,
        grouping=args.grouping,
        boundary_fit=args.boundary_fit,
        ink_support=args.ink_support,
        proposal=args.proposal,
        compose_materials=args.compose_materials,
        boundary_contacts=args.boundary_contacts,
        composition_grouping=args.composition_grouping,
        composition_layout=args.composition_layout,
    )


if __name__ == "__main__":
    main()
