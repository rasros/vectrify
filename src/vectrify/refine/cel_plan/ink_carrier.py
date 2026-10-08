"""Complete source-ended ink alternatives inside an unchanged opacity carrier.

The original source proof remains the diagnostic target. Precise/original
centres and carrier-conditioned source width compete without new connections,
width overrides or relaxed vector footprint proofs. Raster coverage only seeds
an alternative; the existing exact carried-body proof decides feasibility.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pathops

from vectrify.document.join import path_geometry
from vectrify.refine.cel_plan.ink import measure
from vectrify.refine.cel_plan.local import MAX_CROP_PIXELS, MAX_TILES, Box
from vectrify.refine.cel_plan.score import render

MAX_PIXELS = 1536**2
MAX_MASK_BYTES = 16 * 1024**2
MAX_MOVEMENT = 2.0
PAINT_SPREAD = 24


class CarrierFit:
    """One lazily rasterized native-phase carrier per source discovery."""

    def __init__(self, evidence, carrier):
        self.evidence, self.carrier = evidence, carrier
        self.coverage = None
        self.diagnostics = {
            "recovered": 0,
            "precise_source": 0,
            "original_source": 0,
            "carrier_source": 0,
            "coverage_bytes": 0,
            "coverage_tiles": 0,
            "bounds_exclusions": 0,
        }

    def raster(self, work):
        if self.coverage is not None:
            return None if work.interrupted else self.coverage
        evidence = self.evidence
        h, w = evidence.target.shape[:2]
        box = Box(0, 0, w, h)
        if (
            self.carrier is None
            or box.area > MAX_PIXELS
            or box.area * np.dtype(np.float32).itemsize > MAX_MASK_BYTES
        ):
            self.diagnostics["bounds_exclusions"] += 1
            return None
        tiles = tuple(box.chunks(MAX_CROP_PIXELS))
        if len(tiles) > MAX_TILES:
            self.diagnostics["bounds_exclusions"] += 1
            return None
        if work.interrupted:
            return None
        data = path_geometry(self.carrier).path_data()
        fill_rule = (
            "evenodd"
            if self.carrier.fillType == pathops.FillType.EVEN_ODD
            else "nonzero"
        )
        coverage = np.empty((h, w), np.float32)
        sx, sy = evidence.scale
        ox, oy = evidence.offset
        for tile in tiles:
            if work.interrupted:
                return None
            width, height = tile.right - tile.x, tile.bottom - tile.y
            svg = (
                '<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}" '
                f'viewBox="{ox + tile.x / sx} {oy + tile.y / sy} '
                f'{width / sx} {height / sy}" preserveAspectRatio="none">'
                f'<path d="{data}" fill="black" fill-rule="{fill_rule}"/></svg>'
            )
            pixels = render(svg, (width, height))
            if work.interrupted:
                return None
            coverage[tile.slices] = pixels[..., 3]
        if work.interrupted:
            return None
        coverage.flags.writeable = False
        self.coverage = coverage
        self.diagnostics["coverage_bytes"] = coverage.nbytes
        self.diagnostics["coverage_tiles"] = len(tiles)
        return coverage

    def recover(
        self, run, proof, options, work, budget, light, visible, opacity, attempt
    ):
        if options.line_width or self.carrier is None or work.interrupted:
            return []
        evidence = self.evidence
        exact = replace(options, tolerance=min(options.tolerance or 0.75, 0.25))

        def tried(candidate, settings, method):
            movement = np.linalg.norm(
                (candidate.points - proof.points) / evidence.scale, axis=1
            )
            if np.any(movement > MAX_MOVEMENT + 1e-8):
                return []
            for cap in ("round", "butt"):
                if work.interrupted:
                    return []
                found = attempt(
                    run,
                    candidate,
                    evidence,
                    settings,
                    self.carrier,
                    work,
                    False,
                    budget,
                    light,
                    visible,
                    cap=cap,
                    opacity=opacity,
                )
                if found:
                    self.diagnostics["recovered"] += 1
                    self.diagnostics[method] += 1
                    return found
            return []

        anchored = replace(proof, points=run.copy())
        for candidate, settings, method in (
            (proof, exact, "precise_source"),
            (anchored, options, "original_source"),
            (anchored, exact, "original_source"),
        ):
            found = tried(candidate, settings, method)
            if found or work.interrupted:
                return found
        coverage = self.raster(work)
        if coverage is None or work.interrupted:
            return []
        measured = measure(
            run,
            evidence.target,
            proof.width,
            light=light,
            visible=visible,
            opacity=opacity,
            coverage=coverage,
        )
        if measured is None or work.interrupted:
            return []
        movement = np.linalg.norm(
            (measured.points - proof.points) / evidence.scale, axis=1
        )
        allowance = np.clip(
            proof.width / np.sqrt(np.prod(evidence.scale)) / 2, 0.5, MAX_MOVEMENT
        )
        if np.any(movement > allowance + 1e-8) or np.any(
            np.abs(measured.paint - proof.paint) > PAINT_SPREAD
        ):
            return []
        # The source paint is held while fitting the carrier-eligible body.
        # Geometry retains the original source endpoints and exact footprint.
        candidate = replace(measured, paint=proof.paint.copy())
        return tried(candidate, exact, "carrier_source")
