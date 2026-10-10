"""Discover complete attached outline interpretations from the actual parent.

Original source observations delimit opposing primitive rails. Actual genuine
stroke ports bind the newly inferred guide before raw measurement; complete
shared original material curves select restoration paint. A native-exact floor
and corner-preserving centerline are construction evidence only. Positive body,
physical family/component binding and common search acceptance remain separate.
Ordinary search does not schedule this experimental discovery yet.
"""

from dataclasses import dataclass, replace

import numpy as np

from vectrify.document.join import curve_path, path_style, transformed_geometry
from vectrify.document.model import paint_server
from vectrify.document.redraw import root_matrix
from vectrify.refine.cel_plan.band_fit import MAX_MOVEMENT
from vectrify.refine.cel_plan.filled_bands import MAX_NATIVE_PIXELS, MAX_WIDTH, _check
from vectrify.refine.cel_plan.ink import Ink, measure
from vectrify.refine.cel_plan.line_fidelity import (
    MAX_PROFILES,
    SourceLineGuard,
    SourceProfile,
)
from vectrify.refine.cel_plan.paint_continuation import _paint, _supported
from vectrify.refine.cel_plan.source_centerline import SourceCenterline
from vectrify.refine.cel_plan.source_centerline import construct as centerline
from vectrify.refine.cel_plan.source_faces import (
    MAX_MATERIALS,
    MAX_OBSERVATIONS,
    MAX_SOURCE_POINTS,
    _edges,
)
from vectrify.refine.cel_plan.source_faces import discover as faces
from vectrify.refine.cel_plan.source_slices import source_slice
from vectrify.refine.cel_plan.source_spans import SourceSpan
from vectrify.refine.cel_plan.source_spans import discover as primitive_spans
from vectrify.refine.cel_plan.span_restoration import SpanRestoration
from vectrify.refine.cel_plan.span_restoration import construct as restoration
from vectrify.refine.cel_plan.stroke_inventory import _style

MAX_SURFACES = 512
MAX_TABLE_NODES = 32_768
MAX_PORT_PAIRS = 128
MAX_OWNERS = 16
MAX_FIELDS = 64
MAX_FLOORS = 4
MAX_CANDIDATES = 4


@dataclass(frozen=True)
class AttachedSpan:
    owner: str
    span: SourceSpan
    guide: np.ndarray
    ink: Ink
    profile: SourceProfile
    lines: SourceLineGuard
    centerline: SourceCenterline
    floor: SpanRestoration
    bar: str
    proof: dict


@dataclass(frozen=True)
class _Material:
    id: str
    parent: str
    bounds: np.ndarray
    edges: frozenset


@dataclass(frozen=True)
class _Ports:
    id: str
    parent: str
    points: np.ndarray


def _frozen(array):
    result = np.array(array, dtype=float, copy=True)
    result.flags.writeable = False
    return result


def _native(document, oid):
    frame = root_matrix(document, oid)
    determinant = frame[0] * frame[3] - frame[1] * frame[2]
    if (
        not np.isfinite(frame).all()
        or not np.isfinite(determinant)
        or abs(determinant) < 1e-12
    ):
        return None
    return transformed_geometry(document.geometry_for(oid), frame)


def _rank(item):
    return (
        item.proof["field_area"],
        item.span.field.path_data(),
        item.proof["adjoining_area"],
        item.proof["adjoining_material"] or "",
        item.floor.face.path_data() if item.floor.face is not None else "",
    )


class AttachedSpans:
    def __init__(self, evidence, guard):
        self.evidence, self.guard = evidence, guard
        self.diagnostics = dict.fromkeys(
            (
                "owners",
                "source_fields",
                "primitive_fields",
                "port_matches",
                "floor_exclusions",
                "candidates",
                "bounded",
            ),
            0,
        )

    def _tables(self, document, partition, work):
        _check(work)
        size = self.evidence.source_size
        if (
            self.evidence.rgba.shape != (size[1], size[0], 4)
            or self.guard.shape != self.evidence.rgba.shape
        ):
            raise ValueError(
                "Attached discovery requires the original native source frame"
            )
        if np.prod(size) > MAX_NATIVE_PIXELS or len(partition.surfaces) > MAX_SURFACES:
            self.diagnostics["bounded"] += 1
            return None
        partition.validate(document)
        protected = partition.family_dependencies
        materials, ports, count = [], [], 0
        for surface in partition.surfaces:
            _check(work)
            if surface.role == "underlay" or surface.id in protected:
                continue
            oid = surface.id
            if not _supported(document, oid) or not _paint(
                document, path_style(document, document.element(oid))
            ):
                continue
            native = _native(document, oid)
            if native is None or any(not s.closed for s in native.subpaths):
                continue
            count += sum(len(s.nodes) for s in native.subpaths)
            if count > MAX_TABLE_NODES:
                self.diagnostics["bounded"] += 1
                return None
            bounds = np.asarray(curve_path(native).bounds)
            if not np.isfinite(bounds).all() or np.max(np.abs(bounds)) > 1_000_000:
                continue
            materials.append(
                _Material(
                    oid,
                    document.ancestry(oid)[-2].id,
                    _frozen(bounds),
                    frozenset(_edges(native)),
                )
            )
        for element in document.elements():
            _check(work)
            if (
                element.tag != "path"
                or element.id in protected
                or _style(document, element) is None
            ):
                continue
            native = _native(document, element.id)
            if native is None:
                continue
            count += sum(len(s.nodes) for s in native.subpaths)
            if count > MAX_TABLE_NODES:
                self.diagnostics["bounded"] += 1
                return None
            for sub in native.subpaths:
                if sub.closed or len(sub.nodes) < 2:
                    continue
                if len(ports) >= MAX_PORT_PAIRS:
                    self.diagnostics["bounded"] += 1
                    return None
                points = np.array([sub.nodes[i].endpoint for i in (0, -1)])
                if not np.isfinite(points).all() or np.array_equal(
                    points[0], points[-1]
                ):
                    continue
                ports.append(
                    _Ports(
                        element.id,
                        document.ancestry(element.id)[-2].id,
                        _frozen(points),
                    )
                )
        return tuple(materials), tuple(ports)

    def _owners(self, document, materials, ports, work):
        owners = []
        for material in materials:
            _check(work)
            style = path_style(document, document.element(material.id))
            if (
                paint_server(style["fill"]) is not None
                or style["fill-rule"] != "nonzero"
            ):
                continue
            if any(
                p.parent == material.parent
                and (
                    (p.points >= material.bounds[:2] - MAX_MOVEMENT)
                    & (p.points <= material.bounds[2:] + MAX_MOVEMENT)
                ).all()
                for p in ports
            ):
                area = float(np.prod(material.bounds[2:] - material.bounds[:2]))
                owners.append((area, material.id))
        return tuple(oid for _, oid in sorted(owners)[:MAX_OWNERS])

    def __call__(self, document, partition, work):
        """Enumerate eligible original owners without a supplied owner filter."""
        tables = self._tables(document, partition, work)
        if tables is None:
            return
        materials, ports = tables
        for oid in self._owners(document, materials, ports, work):
            self.diagnostics["owners"] += 1
            yield from self._discover(document, oid, materials, ports, work)

    def discover(self, document, partition, oid, work):
        """Inspect one actual primary owner using the same generic discovery."""
        tables = self._tables(document, partition, work)
        if tables is None:
            return ()
        materials, ports = tables
        if (
            oid not in {m.id for m in materials}
            or paint_server(path_style(document, document.element(oid))["fill"])
            is not None
        ):
            return ()
        return self._discover(document, oid, materials, ports, work)

    def _discover(self, document, oid, materials, ports, work):
        _check(work)
        parent = document.ancestry(oid)[-2].id
        profiles = self.guard.original_profiles(work=work)
        if len(profiles) > MAX_PROFILES:
            self.diagnostics["bounded"] += 1
            return ()
        sources, options, points = [], {}, 0
        for number, profile in enumerate(profiles):
            _check(work)
            observed = self.guard.source_breaks(profile, work=work)
            sliced = source_slice(document, oid, observed, work)
            if sliced is None:
                continue
            points += len(observed.points)
            if len(sources) >= MAX_OBSERVATIONS or points > MAX_SOURCE_POINTS:
                self.diagnostics["bounded"] += 1
                return ()
            former, _retained = sliced
            sources.append((observed, former))
            for span in primitive_spans(document, oid, observed, former, work):
                options.setdefault(span.field.path_data(), (number, span))
                if len(options) > MAX_FIELDS:
                    largest = max(
                        options,
                        key=lambda key: (
                            abs(curve_path(options[key][1].field).area),
                            key,
                        ),
                    )
                    del options[largest]
        self.diagnostics["source_fields"] += len(sources)
        self.diagnostics["primitive_fields"] += len(options)
        frame = root_matrix(document, oid)
        raw = self.evidence.rgba
        native_evidence = replace(self.evidence, scale=(1.0, 1.0), offset=(0.0, 0.0))
        results = []
        for number, span in options.values():
            _check(work)
            matched = []
            for port in ports:
                if port.parent != parent:
                    continue
                for ends in (port.points, port.points[::-1]):
                    distance = np.linalg.norm(ends - span.guide[[0, -1]], axis=1)
                    if distance.max() <= MAX_MOVEMENT:
                        matched.append(
                            (float(distance.sum()), port.id, tuple(ends.ravel()), ends)
                        )
            if not matched:
                continue
            _distance, bar, _key, ends = min(matched, key=lambda row: row[:3])
            self.diagnostics["port_matches"] += 1
            guide = np.array(span.guide, copy=True)
            guide[[0, -1]] = ends
            native_field = transformed_geometry(span.field, frame)
            field_path = curve_path(native_field)
            shared = _edges(native_field)
            votes = []
            for material in materials:
                if material.id == oid or material.parent != parent:
                    continue
                common = material.edges & shared
                if common:
                    length = float(
                        sum(
                            np.linalg.norm(np.diff(edge, axis=0), axis=1).sum()
                            for edge in common
                        )
                    )
                    votes.append((-length, material.id, len(common)))
            if not votes:
                continue
            _length, base, shared_count = min(votes)
            length = float(np.linalg.norm(np.diff(guide, axis=0), axis=1).sum())
            if length <= 1e-12:
                continue
            hint = float(np.clip(abs(field_path.area) / length, 0.8, MAX_WIDTH))
            ink = measure(
                guide,
                raw[..., :3] * 255,
                hint,
                visible=raw[..., 3] > 1 / 255,
                opacity=raw[..., 3],
            )
            _check(work)
            if ink is None or ink.support < 0.9 or not 0.8 <= ink.width <= MAX_WIDTH:
                continue
            profile = SourceProfile.from_ink(guide, ink, native_evidence)
            lines = SourceLineGuard(raw, (profile,), work=work)
            observed = lines.source_breaks(profile, work=work)
            constructed = centerline(guide, ink.points, work)
            if observed is None or observed.gaps.any() or constructed is None:
                continue
            bounds = np.asarray(field_path.bounds)
            nearby = tuple(
                sorted(
                    m.id
                    for m in materials
                    if m.parent == parent
                    and m.id not in {oid, base}
                    and (m.bounds[:2] <= bounds[2:]).all()
                    and (m.bounds[2:] >= bounds[:2]).all()
                )
            )
            if len(nearby) > MAX_MATERIALS:
                self.diagnostics["bounded"] += 1
                continue
            adjoining = faces(document, oid, span, sources, nearby, base, work)
            for face in adjoining[:MAX_FLOORS] if adjoining else (None,):
                floor = restoration(
                    document,
                    oid,
                    span,
                    base,
                    self.evidence.source_size,
                    work,
                    face=face,
                )
                if floor is None:
                    self.diagnostics["floor_exclusions"] += 1
                    continue
                results.append(
                    AttachedSpan(
                        oid,
                        span,
                        _frozen(guide),
                        replace(
                            ink, points=_frozen(ink.points), paint=_frozen(ink.paint)
                        ),
                        profile,
                        lines,
                        constructed,
                        floor,
                        bar,
                        {
                            "scope": "complete-attached-span-before-positive-body",
                            "accepted": False,
                            "positive_body_proved": False,
                            "original_profile": number,
                            "field_area": abs(field_path.area),
                            "field_nodes": sum(
                                len(s.nodes) for s in span.field.subpaths
                            ),
                            "genuine_bar": bar,
                            "base_material": base,
                            "shared_base_edges": shared_count,
                            "adjoining_material": face.material if face else None,
                            "adjoining_area": abs(curve_path(face.field).area)
                            if face
                            else 0.0,
                            "adjoining_shared_edges": face.shared_edges if face else 0,
                            "raw_support": ink.support,
                            "raw_width": ink.width,
                            "qualified": int(observed.qualified.sum()),
                            "gaps": int(observed.gaps.sum()),
                            "centerline_nodes": len(
                                constructed.geometry.subpaths[0].nodes
                            ),
                            "floor": floor.proof,
                        },
                    )
                )
                # Retain only the bounded geometric pool, rather than keeping
                # every document/source bank until final ranking.
                if len(results) > MAX_CANDIDATES:
                    results.sort(key=_rank)
                    results.pop()
        _check(work)
        # Provenance indices never rank geometry. Each interpretation still
        # needs complete positive body, family and component acceptance.
        selected = tuple(sorted(results, key=_rank))
        self.diagnostics["candidates"] += len(selected)
        return selected
