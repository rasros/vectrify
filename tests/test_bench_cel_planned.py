"""The human benchmark keeps comparison truth out of algorithm inputs."""

import hashlib
import json

import numpy as np

from scripts.bench_cel_planned import DATA, MANIFEST, load_case
from vectrify.refine.cel_plan.score import foreground_mask, svg_metrics


def test_compressed_human_fixture_and_frozen_mask_match_the_diagnosis():
    case = json.loads(MANIFEST.read_text())["cases"][0]
    svg, reference = load_case(DATA / case["file"])
    stats = svg_metrics(svg)
    assert {k: stats[k] for k in ("paths", "contours", "nodes", "gradients")} == {
        k: case["baseline"]["human"][k]
        for k in ("paths", "contours", "nodes", "gradients")
    }
    mask = foreground_mask(np.asarray(reference, dtype=np.float32) / 255)
    assert hashlib.sha256(mask.tobytes()).hexdigest() == case["mask_sha256"]
    assert case["training"] is False
