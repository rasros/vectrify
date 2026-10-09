"""Propose material paint and a fitted silhouette independently of its fringe.

The original source atoms remain the ownership namespace. Whole low-alpha
atoms can contribute to the component silhouette rather than become separate
painted paths. The unchanged native reference and policy decide admissibility.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import binary_erosion, label

from vectrify.refine.cel_plan.export import export
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.materials import coherent_labels
from vectrify.refine.cel_plan.model import (
    Evidence,
    Graph,
    StageInterruptedError,
    Work,
)
from vectrify.refine.cel_plan.ownership import Partition, Surface
from vectrify.refine.cel_plan.policy import Policy

MAX_PIXELS = 1536**2
MAX_COMPONENTS = 8


@dataclass(frozen=True)
class Core:
    opacity: float
    members: tuple[int, ...]
    fringe: tuple[int, ...]
    anchor: tuple[int, int]


def prepare(evidence: Evidence, graph: Graph, policy: Policy, work: Work):
    """Only near-uniform native materials; preserve unsupported opacity models."""
    if evidence.opacity is None or evidence.scale != (1, 1):
        return evidence, (), {"status": "unsupported-scale-or-alpha"}
    if evidence.empty.size > MAX_PIXELS or evidence.filled_line_width > 0:
        return evidence, (), {"status": "bounded-or-explicit-width"}
    opacity = evidence.opacity.copy()
    empty = evidence.empty.copy()
    permitted = np.zeros_like(empty)
    components = np.array([r.component for r in graph.regions], dtype=np.int32)[
        evidence.labels
    ]
    source_holes = np.zeros_like(empty)
    x, y = evidence.offset
    height, width = empty.shape
    # Only protected voids matter here; a single raster speck must not turn
    # material fitting into another opacity-byte partitioning scheme.
    for feature in policy.holes:
        xx, yy, w, h = feature.box
        if xx >= x and yy >= y and xx + w <= x + width and yy + h <= y + height:
            source_holes[yy - y : yy - y + h, xx - x : xx - x + w] |= feature.support
    cores = []
    exclusions: dict[str, int] = {}

    def exclude(reason):
        exclusions[reason] = exclusions.get(reason, 0) + 1

    groups = {}
    for region in graph.regions:
        if region.component and region.id not in graph.hidden and region.area:
            groups.setdefault(region.component, []).append(region)
    for component, native in sorted(groups.items()):
        if work.interrupted:
            raise StageInterruptedError("Material coverage proposal interrupted")
        if len(cores) >= MAX_COMPONENTS:
            exclude("component-limit")
            break
        if sum(region.area for region in native) < 256:
            exclude("small-or-thin")
            continue
        own = (components == component) & ~evidence.empty
        if not binary_erosion(own, iterations=2).any():
            exclude("small-or-thin")
            continue
        samples = evidence.opacity[own]
        levels = np.rint(samples * 255).astype(np.int32)
        modal = int(np.bincount(levels, minlength=256).argmax())
        intrinsic = float(np.median(samples[levels == modal]))
        if samples.max() > intrinsic * 1.02 + 1e-7:
            exclude("variable-opacity")
            continue
        if (source_holes & own & (evidence.opacity > 0.5 / 255)).any():
            exclude("partial-opacity-hole")
            continue
        members = tuple(
            r.id
            for r in native
            if r.opacity_range is not None and r.opacity_range[0] >= intrinsic * 0.5
        )
        selected = np.zeros(len(graph.regions), dtype=bool)
        selected[list(members)] = True
        core = selected[evidence.labels] & own
        _, count = label(core)
        if count != 1 or core.sum() < own.sum() * 0.5:
            exclude("disconnected-or-small-core")
            continue
        deep = binary_erosion(core, iterations=6)
        if (
            not deep.any()
            or np.mean(np.abs(evidence.opacity[deep] - intrinsic) > intrinsic * 0.05)
            > 0.005
        ):
            exclude("nonuniform-material")
            continue
        hard = policy.opacity_inside[y : y + height, x : x + width] & core
        tolerance = policy.opacity_tolerance[y : y + height, x : x + width]
        if ((np.abs(evidence.opacity - intrinsic) > tolerance) & hard).sum() > max(
            4, round(policy.area * 0.0005)
        ):
            exclude("interior-opacity-residual")
            continue
        fringe = tuple(r.id for r in native if not selected[r.id])
        yy, xx = np.argwhere(core)[0]
        cores.append(Core(intrinsic, members, fringe, (int(yy), int(xx))))
        empty[own & ~core] = True
        opacity[core] = intrinsic
        permitted[core] = True
    for array in (empty, opacity, permitted):
        array.flags.writeable = False
    return (
        replace(evidence, empty=empty, opacity=opacity, coverage_fit=permitted),
        tuple(cores),
        {
            "status": "prepared" if cores else "no-material-model",
            "exclusions": exclusions,
        },
    )


def candidate(
    evidence, graph, policy, options, work, *, normalizer, discovery=None
) -> tuple[tuple[str, dict] | None, dict]:
    # Growth can return a complete partition when its optional time slice ends.
    # Export needs its own live work budget to turn that partition into a fully
    # validated proposal; sharing the expired slice would discard every prefix.
    growth_work = discovery or work
    virtual, cores, details = prepare(evidence, graph, policy, growth_work)
    if not cores:
        return None, details
    current = build(virtual, work=growth_work)
    labels, growth = coherent_labels(
        virtual,
        current,
        replace(options, complexity=50),
        growth_work,
        normalizer=normalizer,
    )
    svg, exported = export(
        virtual,
        labels,
        replace(options, complexity=50, tolerance=options.tolerance or 1),
        work,
        layers=True,
        structure=True,
        cost_normalizer=normalizer,
    )
    partition = Partition.from_metadata(exported.get("planning_surfaces"))
    if partition is None:
        return None, {**details, "status": "incomplete-core-ownership"}
    surfaces = list(partition.surfaces)
    primary_owners = partition.owners
    visible_components, _ = label(~virtual.empty)
    for core in cores:
        if not core.fringe:
            continue
        component = int(visible_components[core.anchor])
        base = next((s for s in surfaces if s.id == f"cel-base-{component}"), None)
        if base is not None:
            surfaces[surfaces.index(base)] = Surface(
                base.id, core.fringe, "surface", base.members
            )
            exported["paint_constraints"].append(base.id)
        else:
            owners = {primary_owners[i] for i in core.members}
            if len(owners) != 1:
                return None, {**details, "status": "missing-shared-core-base"}
            surface = next(s for s in surfaces if s.id in owners)
            surfaces[surfaces.index(surface)] = replace(
                surface, members=tuple(sorted((*surface.members, *core.fringe)))
            )
    complete = Partition(tuple(surfaces))
    expected = {r.id for r in graph.regions if r.id not in graph.hidden and r.area}
    if set(complete.owners) != expected:
        return None, {**details, "status": "incomplete-source-ownership"}
    exported["planning_surfaces"] = complete.metadata()
    models = [
        {
            "opacity": core.opacity,
            "core_atoms": len(core.members),
            "fringe_atoms": len(core.fringe),
        }
        for core in cores
    ]
    return (
        svg,
        {**exported, "alpha_model": "material-core-silhouette", "core_models": models},
    ), {**details, "models": models, "growth": growth}
