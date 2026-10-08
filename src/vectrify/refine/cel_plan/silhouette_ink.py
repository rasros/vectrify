"""Source-perimeter stroke hypotheses independent of an opaque paint carrier.

An alpha or foreground contour supplies a closed geometric candidate, never
permission to close ink. Painted-side ridge evidence supplies its centre,
width and paint; native raw-source absence tests decide whether that complete
body may enter the pool. Application still needs complete ownership, carrier,
restoration, ordering and native policy proofs.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects, label

from vectrify.document import Geometry
from vectrify.refine import cel
from vectrify.refine.cel_plan.geometry import fitted
from vectrify.refine.cel_plan.ink import measure_closed
from vectrify.refine.cel_plan.ink_models import (
    MAX_MASK_BYTES,
    MAX_MODELS,
    MAX_PIXELS,
    MAX_POINTS,
    MAX_SOURCE_COMPONENTS,
    MAX_WIDTH,
    InkModel,
    covered_selection,
    footprint,
)
from vectrify.refine.cel_plan.ink_models import (
    models as source_models,
)
from vectrify.refine.cel_plan.line_fidelity import SourceProfile
from vectrify.refine.cel_plan.local import MAX_CROP_PIXELS
from vectrify.refine.cel_plan.model import StageInterruptedError
from vectrify.refine.cel_plan.source_absence import SourceAbsence
from vectrify.refine.cel_plan.source_widths import SourceWidths
from vectrify.refine.crossings import crossings
from vectrify.refine.tracing import _loops

MAX_COMPONENTS = 16
MAX_LOOPS = 16
PROFILE_POINTS = 1024
PROFILE_OVERLAP = 256


def profiles(original, proof, evidence, component):
    """Package a cycle with overlapping native observations across its seam.

    Every raw point is retained. Overlap exceeds the inspected gap reach at
    the existing width bound; no chunk endpoint becomes a geometry endpoint.
    Each chunk still meets SourceAbsence's independent native sample bound.
    """
    complete = SourceProfile.from_ink(
        original, proof, evidence, component=component, cyclic=True
    )
    count = len(complete.points) - 1
    overlap = min(PROFILE_OVERLAP, count)
    indices = np.arange(-overlap, count + overlap) % count
    fields = tuple(
        getattr(complete, name)[indices]
        for name in ("points", "sides", "direction", "tolerance")
    )
    result = []
    for start in range(0, len(indices) - 1, PROFILE_POINTS - PROFILE_OVERLAP):
        end = min(len(indices), start + PROFILE_POINTS)
        arrays = tuple(a[start:end] for a in fields)
        for array in arrays:
            array.flags.writeable = False
        result.append(
            SourceProfile(
                arrays[0], arrays[1], arrays[2], arrays[3], component, (False, False)
            )
        )
    return tuple(result)


class SilhouetteInk:
    def __init__(self, evidence, options):
        self.evidence, self.options = evidence, options
        self.diagnostics = dict.fromkeys(
            (
                "components",
                "loops",
                "points",
                "raw_profiles",
                "measured_perimeters",
                "gap_samples_peak",
                "bounds_exclusions",
                "ridge_exclusions",
                "geometry_exclusions",
                "absence_exclusions",
                "models",
                "width_trials",
                "cancelled",
            ),
            0,
        )

    def __call__(self, work):
        try:
            return self._models(work)
        except StageInterruptedError:
            self.diagnostics["cancelled"] += 1
            return ()

    def _models(self, work):
        evidence, options = self.evidence, self.options
        if work.interrupted:
            return ()
        if evidence.empty.size > MAX_PIXELS or evidence.filled_line_width:
            self.diagnostics["bounds_exclusions"] += 1
            return ()
        visible = ~evidence.empty & evidence.foreground
        components, count = label(visible, np.ones((3, 3)))
        if count > MAX_SOURCE_COMPONENTS:
            self.diagnostics["bounds_exclusions"] += 1
            return ()
        areas = np.bincount(components.ravel(), minlength=count + 1)
        order = sorted(range(1, count + 1), key=lambda i: (-areas[i], i))[
            :MAX_COMPONENTS
        ]
        boxes = find_objects(components)
        # All original physical source gaps constrain the new exterior body,
        # including other chains which cannot themselves become a stroke.
        raw_profiles = []
        source_models(
            evidence.drawn & visible,
            evidence,
            options,
            work,
            prune_spurs=True,
            fractional_coverage=evidence.opacity is not None,
            source_profiles=raw_profiles,
        )
        if work.interrupted:
            self.diagnostics["cancelled"] += 1
            return ()
        self.diagnostics["raw_profiles"] += len(raw_profiles)
        namespace = hashlib.sha256(visible.tobytes()).hexdigest()
        maximum_models = min(MAX_MODELS, MAX_MASK_BYTES // visible.nbytes)
        result = []
        used_points = loops = 0
        try:
            for component in order:
                if work.interrupted:
                    raise StageInterruptedError("Source silhouettes interrupted")
                box = boxes[component - 1]
                own = components[box] == component
                alpha = evidence.opacity[box] if evidence.opacity is not None else None
                # Half of this physical component's own measured maximum keeps
                # independent faint components and uniformly translucent art.
                core = own if alpha is None else own & (alpha >= alpha[own].max() * 0.5)
                if not core.any():
                    continue
                self.diagnostics["components"] += 1
                drawn = evidence.drawn[box] & own
                depth = np.asarray(distance_transform_edt(drawn))
                skeleton = cel.thin(drawn)
                typical = (
                    max(2.0, float(np.median(2 * depth[skeleton])))
                    if skeleton.any()
                    else 2.0
                )
                seeds = tuple(dict.fromkeys((1.5 * typical, 2 * typical)))
                seeds = tuple(seed for seed in seeds if seed <= MAX_WIDTH)
                if not seeds:
                    self.diagnostics["bounds_exclusions"] += 1
                    continue
                for loop in _loops(core):
                    if loops >= MAX_LOOPS:
                        self.diagnostics["bounds_exclusions"] += 1
                        break
                    loops += 1
                    if len(loop) < 16 or used_points + len(loop) + 1 > MAX_POINTS:
                        self.diagnostics["bounds_exclusions"] += 1
                        continue
                    used_points += len(loop) + 1
                    self.diagnostics["loops"] += 1
                    self.diagnostics["points"] += len(loop) + 1
                    original = np.array([*loop, loop[0]], float)
                    original += (box[1].start, box[0].start)
                    for seed in seeds:
                        if (
                            len(original)
                            * (2 * int(np.ceil(max(3, 2.5 * seed) / 0.5)) + 1)
                            > MAX_CROP_PIXELS
                        ):
                            self.diagnostics["bounds_exclusions"] += 1
                            continue
                        proof = measure_closed(
                            original,
                            evidence.target,
                            seed,
                            visible=~evidence.empty,
                            opacity=evidence.opacity,
                        )
                        if work.interrupted:
                            raise StageInterruptedError(
                                "Source silhouettes interrupted"
                            )
                        if proof is None:
                            self.diagnostics["ridge_exclusions"] += 1
                            continue
                        if proof.width > MAX_WIDTH:
                            self.diagnostics["bounds_exclusions"] += 1
                            continue
                        self.diagnostics["measured_perimeters"] += 1
                        model = fitted(
                            proof.points / evidence.scale + evidence.offset,
                            options.tolerance or 0.75,
                        )
                        geometry = Geometry(
                            f"source-silhouette-{component}-{loops}", (model.contour,)
                        )
                        if crossings(geometry):
                            self.diagnostics["geometry_exclusions"] += 1
                            continue
                        loop_profiles = profiles(
                            original, proof, evidence, (namespace, component)
                        )
                        try:
                            absence = SourceAbsence(
                                evidence, (*raw_profiles, *loop_profiles), work
                            )
                            self.diagnostics["gap_samples_peak"] = max(
                                self.diagnostics["gap_samples_peak"],
                                len(absence.points),
                            )
                            width = options.line_width or proof.width / np.sqrt(
                                np.prod(evidence.scale)
                            )
                            fitter = SourceWidths(absence)
                            if not options.line_width:
                                width = fitter.fit(
                                    ({"contours": [model.contour], "cap": "round"},),
                                    (width,),
                                    np.sqrt(np.prod(evidence.scale)),
                                    work,
                                )[0]
                            self.diagnostics["width_trials"] += fitter.diagnostics[
                                "fits"
                            ]
                            permitted = absence.permits(geometry, width, "round", work)
                        except ValueError:
                            self.diagnostics["bounds_exclusions"] += 1
                            continue
                        if not permitted:
                            self.diagnostics["absence_exclusions"] += 1
                            continue
                        selected = covered_selection(
                            evidence.drawn & (components == component),
                            geometry,
                            width,
                            evidence,
                            work,
                        )
                        if selected is None or not selected.any():
                            continue
                        selected.flags.writeable = False
                        result.append(
                            InkModel(
                                geometry,
                                footprint(geometry, width),
                                proof.paint,
                                selected,
                                {
                                    "model": "source-silhouette-stroke",
                                    "ownership": "rendered-stroke-coverage",
                                    "width": float(width),
                                    "linecap": "round",
                                    "runs": 1,
                                    "support": proof.support,
                                    "peak_gap": proof.peak_gap,
                                    "source_alpha_level": 0.5,
                                    "source_component": component,
                                    "source_absence_samples": len(absence.points),
                                    "source_profiles": len(raw_profiles)
                                    + len(loop_profiles),
                                    "source_width_fit": dict(fitter.diagnostics),
                                },
                            )
                        )
                        if work.interrupted:
                            raise StageInterruptedError(
                                "Source silhouettes interrupted"
                            )
                        if len(result) >= maximum_models:
                            self.diagnostics["models"] += len(result)
                            return tuple(result)
        except StageInterruptedError:
            self.diagnostics["cancelled"] += 1
            return ()
        if work.interrupted:
            self.diagnostics["cancelled"] += 1
            return ()
        self.diagnostics["models"] += len(result)
        return tuple(result)
