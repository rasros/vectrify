"""Separate edge fidelity metrics at a documented inspection scale."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from PIL import Image
from scipy import ndimage


def edges(image: Image.Image, threshold: float) -> np.ndarray:
    pixels = np.asarray(image.convert("RGB"), dtype=float) / 255
    pixels = ndimage.gaussian_filter(pixels, sigma=(0.7, 0.7, 0))
    gx = ndimage.sobel(pixels, axis=1) / 8
    gy = ndimage.sobel(pixels, axis=0) / 8
    strength = np.hypot(gx, gy)
    channel = strength.argmax(axis=2)[..., None]
    magnitude = np.take_along_axis(strength, channel, axis=2)[..., 0]
    dx = np.take_along_axis(gx, channel, axis=2)[..., 0]
    dy = np.take_along_axis(gy, channel, axis=2)[..., 0]
    norm = np.maximum(magnitude, 1e-12)
    ys, xs = np.indices(magnitude.shape)
    a = ndimage.map_coordinates(magnitude, [ys + dy / norm, xs + dx / norm], order=1)
    b = ndimage.map_coordinates(magnitude, [ys - dy / norm, xs - dx / norm], order=1)
    return (magnitude >= threshold) & (magnitude >= a) & (magnitude >= b)


def compare_edges(
    drawing: Image.Image,
    reference: Image.Image,
    box: tuple,
    threshold: float = 0.05,
    tolerance: float = 1.0,
) -> tuple[dict, Image.Image]:
    from vectrify.document import DocumentError

    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise DocumentError("edge_threshold must be in (0, 1]")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise DocumentError("edge_tolerance must be a nonnegative document distance")
    d, r = edges(drawing, threshold), edges(reference, threshold)
    sampling = (box[3] / drawing.height, box[2] / drawing.width)
    dr, nearest = cast(
        tuple[np.ndarray, np.ndarray],
        ndimage.distance_transform_edt(~r, sampling=sampling, return_indices=True),
    )
    dd = cast(np.ndarray, ndimage.distance_transform_edt(~d, sampling=sampling))
    matched_d = d & (dr <= tolerance) if r.any() else np.zeros_like(d)
    matched_r = r & (dd <= tolerance) if d.any() else np.zeros_like(r)
    missing, extra = r & ~matched_r, d & ~matched_d
    # Mark separate drawing components that repeat the same reference support.
    labels, count = ndimage.label(d, structure=np.ones((3, 3)))
    duplicate = np.zeros_like(d)
    claimed: set[tuple[int, int]] = set()
    for label in range(1, count + 1):
        component = labels == label
        ys, xs = np.nonzero(component & matched_d)
        support = set(
            zip(nearest[0, ys, xs].tolist(), nearest[1, ys, xs].tolist(), strict=True)
        )
        if support and len(support & claimed) / len(support) >= 0.5:
            duplicate |= component
        claimed |= support
    distance_d = dr[d] if r.any() else np.array([])
    distance_r = dd[r] if d.any() else np.array([])
    distances = np.concatenate([distance_d, distance_r])
    metrics: dict[str, Any] = {
        "precision": float(matched_d.sum() / d.sum()) if d.any() else 1.0,
        "recall": float(matched_r.sum() / r.sum()) if r.any() else 1.0,
        "boundary_displacement": {
            "drawing_to_reference_mean": float(distance_d.mean())
            if distance_d.size
            else None,
            "reference_to_drawing_mean": float(distance_r.mean())
            if distance_r.size
            else None,
            "symmetric_mean": float(distances.mean()) if distances.size else None,
            "p95": float(np.percentile(distances, 95)) if distances.size else None,
            "units": "document units",
        },
        "drawing_edge_pixels": int(d.sum()),
        "reference_edge_pixels": int(r.sum()),
        "missing_edge_pixels": int(missing.sum()),
        "extra_edge_pixels": int(extra.sum()),
        "duplicated_edge_pixels": int(duplicate.sum()),
        "threshold": threshold,
        "tolerance": tolerance,
        "pixels": list(drawing.size),
        "units_per_pixel": [sampling[1], sampling[0]],
        "legend": {
            "missing": "#ff3030",
            "extra": "#ff9900",
            "duplicate": "#3060ff",
            "matched": "#20b050",
        },
    }
    annotated = (np.asarray(reference.convert("RGB"), dtype=float) * 0.4 + 153).astype(
        np.uint8
    )
    annotated[matched_d] = (32, 176, 80)
    annotated[extra] = (255, 153, 0)
    annotated[missing] = (255, 48, 48)
    annotated[duplicate] = (48, 96, 255)
    return metrics, Image.fromarray(annotated)


def feature_checks(document, checks: list[dict]) -> list[dict]:
    from vectrify.document import DocumentError
    from vectrify.document.regions import object_matrix
    from vectrify.document.topology import mapped_point
    from vectrify.ui.agent import _finite

    results = []
    for check in checks:
        if not isinstance(check, dict) or set(check) - {
            "object",
            "node",
            "expected",
            "max_displacement",
            "min_turn",
        }:
            raise DocumentError(
                "Feature checks use object, node, expected, max_displacement, min_turn"
            )
        oid, nid = str(check["object"]), str(check["node"])
        geometry = document.geometry_for(oid)
        expected = check["expected"]
        if not isinstance(expected, list) or len(expected) != 2:
            raise DocumentError(
                "Feature expected position is [x,y] in document coordinates"
            )
        expected = tuple(_finite(v, "expected position") for v in expected)
        tolerance = _finite(check.get("max_displacement", 0), "max_displacement")
        min_turn = _finite(check.get("min_turn", 30), "min_turn")
        if tolerance < 0 or not 0 <= min_turn <= 180:
            raise DocumentError(
                "Feature displacement must be nonnegative and turn in [0,180]"
            )
        node = geometry.node(nid)
        sp = next(sp for sp in geometry.subpaths if node in sp.nodes)
        index = sp.nodes.index(node)
        matrix = object_matrix(document, oid)
        point = mapped_point(node.endpoint, matrix)
        turn = None
        if sp.closed or 0 < index < len(sp.nodes) - 1:
            before = sp.nodes[(index - 1) % len(sp.nodes)]
            after = sp.nodes[(index + 1) % len(sp.nodes)]
            incoming = (
                [node.values[2:4], node.values[:2], before.endpoint]
                if node.command == "C"
                else [before.endpoint]
            )
            outgoing = (
                [after.values[:2], after.values[2:4], after.endpoint]
                if after.command == "C"
                else [after.endpoint]
            )

            incoming = [mapped_point(p, matrix) for p in incoming]
            outgoing = [mapped_point(p, matrix) for p in outgoing]
            a = next((p for p in incoming if math.dist(p, point) > 1e-12), point)
            b = next((p for p in outgoing if math.dist(p, point) > 1e-12), point)
            u, v = np.array(point) - a, np.array(b) - point
            length = np.linalg.norm(u) * np.linalg.norm(v)
            if length > 1e-12:
                turn = math.degrees(math.acos(float(np.clip(u @ v / length, -1, 1))))
        displacement = math.dist(point, expected)
        results.append(
            {
                "object": oid,
                "node": nid,
                "protected_feature": node.feature,
                "position": list(point),
                "displacement": displacement,
                "turn_degrees": turn,
                "position_pass": displacement <= tolerance,
                "corner_pass": turn is not None and turn >= min_turn,
            }
        )
    return results
