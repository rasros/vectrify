"""Shared setup and report guard for source-only CEL diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scripts.bench_cel_planned import DATA, load_reference, source_hash
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.model import Options, Work
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.score import foreground_mask


@dataclass
class Prepared:
    revision: str
    reference: Image.Image
    raster: bytes | None
    truth: np.ndarray
    mask: np.ndarray
    mask_sha256: str
    options: Options
    work: Work
    evidence: Any
    graph: Any
    policy: Policy | None


def prepare(
    case,
    seconds,
    *,
    refine=True,
    graph_work=False,
    policy=True,
    reference=None,
):
    """Hash sources, check the frozen mask and build source-only evidence.

    ``reference`` replaces the case raster (paired cases); ``graph_work`` lets
    graph construction observe the time allowance.
    """
    revision, raster = source_hash(), None
    if reference is None:
        reference, raster = load_reference(DATA / case["file"])
    truth = np.asarray(reference, dtype=np.float32) / 255
    mask = foreground_mask(truth)
    mask_sha256 = hashlib.sha256(mask.tobytes()).hexdigest()
    if mask_sha256 != case["mask_sha256"]:
        raise ValueError("Fixture mask differs from frozen benchmark")
    options = Options() if refine else Options(refine=False)
    work = Work.start(seconds)
    evidence = collect(reference, None, options, work)
    graph = build(evidence, work=work) if graph_work else build(evidence)
    return Prepared(
        revision,
        reference,
        raster,
        truth,
        mask,
        mask_sha256,
        options,
        work,
        evidence,
        graph,
        Policy.from_evidence(evidence, graph) if policy else None,
    )


def finish(path: Path, report, revision, *, allow_nan=True):
    """Refuse reports whose algorithm sources changed, then write the JSON."""
    if source_hash() != revision:
        raise ValueError("Algorithm sources changed during the benchmark")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=allow_nan) + "\n")
    return report
