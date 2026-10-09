from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from tests.mcp.helpers import reference_png
from tests.mcp.test_agent import fresh
from vectrify.document import DocumentError
from vectrify.ui.agent_edges import compare_edges, feature_checks


def test_pixel_error_can_improve_while_edge_recall_worsens():
    reference = np.full((100, 100, 3), 200, dtype=np.uint8)
    reference[:, 48:50] = 0
    precise = np.full_like(reference, 255)
    precise[:, 48:50] = 0
    missing = np.full_like(reference, 200)

    def mse(a):
        return np.mean(((a.astype(float) - reference) / 255) ** 2)

    good, _ = compare_edges(
        Image.fromarray(precise), Image.fromarray(reference), (0, 0, 100, 100)
    )
    bad, _ = compare_edges(
        Image.fromarray(missing), Image.fromarray(reference), (0, 0, 100, 100)
    )
    assert mse(missing) < mse(precise)
    assert good["recall"] > 0.99
    assert bad["recall"] == 0
    assert bad["missing_edge_pixels"] > 0
    assert bad["boundary_displacement"]["reference_to_drawing_mean"] is None


def test_boundary_displacement_and_duplicated_edge_annotation():
    reference = np.full((100, 100, 3), 255, dtype=np.uint8)
    reference[:, :50] = 0
    drawing = reference.copy()
    drawing[:, 52:55] = 0
    metrics, picture = compare_edges(
        Image.fromarray(drawing),
        Image.fromarray(reference),
        (0, 0, 100, 100),
        tolerance=6,
    )
    assert metrics["boundary_displacement"]["drawing_to_reference_mean"] > 0
    assert metrics["duplicated_edge_pixels"] > 0
    assert np.any(np.all(np.asarray(picture) == [48, 96, 255], axis=2))


def test_explicit_corner_position_and_turn_checks():
    agent, _ = fresh()
    doc = agent.session.editor.snapshot.document
    node = doc.geometry_for("hill").subpaths[0].nodes[1]
    results = feature_checks(
        doc,
        [{"object": "hill", "node": node.id, "expected": [120, 60], "min_turn": 20}],
    )
    assert results[0]["position_pass"]
    assert results[0]["corner_pass"]
    result = feature_checks(
        doc, [{"object": "hill", "node": node.id, "expected": [100, 60]}]
    )[0]
    assert not result["position_pass"]


def test_compare_exposes_mapped_normal_and_close_crops():
    import base64

    agent, seen = fresh()
    agent.call(
        "set_reference",
        {
            "seen": seen,
            "name": "ref.png",
            "data_url": "data:image/png;base64,"
            + base64.b64encode(reference_png()).decode(),
        },
    )
    reply = agent.call(
        "compare", {"edge_aware": True, "close_region": [100, 40, 100, 100]}
    )
    assert reply.data["edges"]["units_per_pixel"]
    assert reply.data["close"]["mapping"]["pixels"] == [1024, 1024]
    assert len(reply.images) == 8
    with pytest.raises(DocumentError, match="edge_threshold"):
        agent.call("compare", {"edge_aware": True, "edge_threshold": 0})
