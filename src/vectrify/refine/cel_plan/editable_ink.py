"""Experimental quality of visible, source-supported editable ink.

Queries and qualification are frozen before candidate construction. New attached
strokes earn credit only through sealed complete removal families. Filled paint,
occluded strokes, bright overlays and candidate-created queries cannot supply it.
"""

import hashlib
from dataclasses import replace
from xml.etree import ElementTree as ET

import numpy as np
from scipy.ndimage import map_coordinates

from vectrify.document import export_svg
from vectrify.document.join import path_style
from vectrify.document.model import paint_server
from vectrify.document.redraw import root_matrix
from vectrify.document.svg import parse_path
from vectrify.refine.cel_plan.filled_bands import MAX_NATIVE_PIXELS, _check
from vectrify.refine.cel_plan.line_fidelity import MAX_PROFILES
from vectrify.refine.cel_plan.local import _native_raster, _render_tree
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.search import MAX_SEED_NODES, MAX_SEED_OBJECTS
from vectrify.refine.cel_plan.source_absence import ALPHA_TOLERANCE
from vectrify.refine.cel_plan.source_family import _data
from vectrify.refine.cel_plan.span_body_fit import _stroke_root
from vectrify.refine.cel_plan.stroke_inventory import _style

VERSION = 1
MAX_PAINTERS = MAX_SEED_OBJECTS
MAX_STROKES = 256
MAX_NODES = MAX_SEED_NODES
MAX_QUERIES = 32_768
PAINTERS = frozenset(
    {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "use"}
)


def _painters(document, work):
    ids = []
    nodes = 0
    for e in document.elements():
        _check(work)
        if e.tag not in PAINTERS or any(
            a.tag in {"defs", "clipPath", "mask"} for a in document.ancestry(e.id)
        ):
            continue
        ids.append(e.id)
        if e.geometry_id is not None:
            nodes += sum(len(s.nodes) for s in document.geometry_for(e.id).subpaths)
        if len(ids) > MAX_PAINTERS or nodes > MAX_NODES:
            raise ValueError("Editable ink exceeds the native painter bound")
    return tuple(ids)


def _record(document, oid):
    ancestry = tuple(replace(a, children=()) for a in document.ancestry(oid))
    e = document.element(oid)
    geometry = document.geometry_for(oid) if e.geometry_id is not None else None
    style = path_style(document, e)
    resources = tuple(
        document.element(server)
        for attr in ("fill", "stroke")
        if (server := paint_server(style[attr])) is not None
    )
    return hashlib.sha256(repr((ancestry, geometry, resources)).encode()).hexdigest()


class EditableInk:
    def __init__(self, guard, before, partition, work=None):
        work = work if work is not None else Work.start(float("inf"))
        _check(work)
        h, w = guard.shape[:2]
        if h * w > MAX_NATIVE_PIXELS:
            raise ValueError("Editable ink exceeds the native pixel bound")
        self.guard, self.before, self.partition = guard, before, partition
        self.size = (w, h)
        self._frame(before)
        before.validate()
        if partition is not None:
            partition.validate(before)
        originals = _painters(before, work)
        self._records = {oid: _record(before, oid) for oid in originals}
        self._original_strokes = frozenset(
            oid
            for oid in originals
            if before.element(oid).tag == "path"
            and _style(before, before.element(oid)) is not None
        )
        if len(self._original_strokes) > MAX_STROKES:
            raise ValueError("Editable ink exceeds the genuine stroke bound")
        profiles = guard.original_profiles(work=work)
        if len(profiles) > MAX_PROFILES:
            raise ValueError("Editable ink exceeds the original profile bound")
        contracts = []
        qualified = 0
        for p in profiles:
            _check(work)
            observed = guard.source_breaks(p, work=work)
            count = int(observed.qualified.sum())
            if not count:
                continue
            qualified += count
            if qualified > MAX_QUERIES:
                raise ValueError("Editable ink exceeds the qualified query bound")
            contract = guard.fitting_body(p, (0, 0, w, h), work=work)
            if contract is None:
                raise ValueError("Editable ink requires every original body query")
            contracts.append(contract)
        self._contracts = tuple(contracts)
        self.qualified = qualified
        self._gaps = guard.gap_centres(limit=MAX_QUERIES, work=work)
        self._baseline_gaps = self._gap_values(
            before, tuple(sorted(self._original_strokes)), work
        )
        self._baseline_gaps.flags.writeable = False
        native = _native_raster(ET.fromstring(export_svg(before)), self.size).root
        self._painted = guard.observe(native.astype(np.float32) / 255, work=work)

    def _frame(self, document):
        try:
            size = tuple(
                float(document.root.get(k, "").removesuffix("px"))
                for k in ("width", "height")
            )
            view = tuple(
                map(
                    float,
                    document.root.get("viewBox", f"0 0 {size[0]} {size[1]}")
                    .replace(",", " ")
                    .split(),
                )
            )
        except ValueError as exc:
            raise ValueError(
                "Editable ink requires an aligned native viewport"
            ) from exc
        if size != self.size or view != (0, 0, *self.size):
            raise ValueError("Editable ink requires an aligned native viewport")

    def _gap_values(self, document, ids, work):
        if not len(self._gaps) or not ids:
            return np.zeros(len(self._gaps), dtype=np.float32)
        root = _stroke_root(document, ids, self.size, work)
        for e in root.iter():
            if e.tag == "g":
                e.set("opacity", "1")
            elif e.tag == "path":
                e.set("opacity", "1")
                e.set("stroke-opacity", "1")
        alpha = _native_raster(root, self.size).root[..., 3]
        return map_coordinates(
            alpha.astype(np.float32) / 255,
            [self._gaps[:, 1] - 0.5, self._gaps[:, 0] - 0.5],
            order=1,
            mode="constant",
            cval=0,
        )

    def observe(self, document, partition=None, work=None):
        work = work if work is not None else Work.start(float("inf"))
        _check(work)
        self._frame(document)
        document.validate()
        painters = _painters(document, work)
        original_families = (
            self.partition.families if self.partition is not None else ()
        )
        if partition is not None:
            partition.validate(document)
        families = partition.families if partition is not None else ()
        if any(f not in families for f in original_families):
            raise ValueError("Editable ink lost an original physical family")
        eligible = {
            oid
            for oid in self._original_strokes
            if oid in painters and _style(document, document.element(oid)) is not None
        }
        additions = []
        for family in families:
            _check(work)
            if family in original_families:
                continue
            if (
                family.junction not in self._original_strokes
                or family.owner not in self._records
                or family.original != _data(self.before.geometry_for(family.owner))
                or family.frame != tuple(root_matrix(self.before, family.owner))
                or family.attributes
                != tuple(sorted(self.before.element(family.owner).attributes))
                or _record(document, family.junction) != self._records[family.junction]
            ):
                raise ValueError(
                    "Editable ink requires an original attached removal family"
                )
            additions.append(family)
        physical = {oid for f in families for oid in f.paths}
        unaccounted = (
            frozenset(
                oid
                for oid in painters
                if oid not in physical
                and oid not in eligible
                and self._records.get(oid) != _record(document, oid)
            )
            if additions
            else frozenset()
        )
        interference = (
            _native_raster(
                _render_tree(export_svg(document), unaccounted), self.size
            ).root[..., 3]
            if unaccounted and additions
            else None
        )
        excluded = []
        for family in additions:
            if interference is not None:
                field = ET.Element(
                    "svg", {"width": str(self.size[0]), "height": str(self.size[1])}
                )
                ET.SubElement(
                    field,
                    "path",
                    {
                        "d": parse_path(family.removed).path_data(),
                        "transform": "matrix(" + " ".join(map(str, family.frame)) + ")",
                        "fill": "white",
                    },
                )
                mask = _native_raster(field, self.size).root[..., 3]
                if ((mask > 0) & (interference > 0)).any():
                    excluded.append(family.owner)
                    continue
            eligible.add(family.owner)
        if len(eligible) > MAX_STROKES:
            raise ValueError("Editable ink exceeds the genuine stroke bound")
        ids = tuple(sorted(eligible))
        gaps = self._gap_values(document, ids, work)
        new_gaps = int(
            ((gaps > ALPHA_TOLERANCE) & (self._baseline_gaps <= ALPHA_TOLERANCE)).sum()
        )
        root = _stroke_root(document, ids, self.size, work)
        body = _native_raster(root, self.size).root[..., 3]
        actual = _native_raster(ET.fromstring(export_svg(document)), self.size).root
        without = _native_raster(
            _render_tree(export_svg(document), frozenset(painters) - eligible),
            self.size,
        ).root
        visible = np.zeros(body.shape, dtype=np.float32)
        # Chunk native premultiplied luminance against a white backdrop. Only
        # actual darkening can support a frozen dark source-ink observation.
        rows = max(1, 65_536 // self.size[0])
        for y in range(0, self.size[1], rows):
            _check(work)
            a, b = (
                actual[y : y + rows].astype(np.float32),
                without[y : y + rows].astype(np.float32),
            )
            lum_a = (a[..., :3] @ (0.2126, 0.7152, 0.0722)) * a[..., 3] + 255 * (
                255 - a[..., 3]
            )
            lum_b = (b[..., :3] @ (0.2126, 0.7152, 0.0722)) * b[..., 3] + 255 * (
                255 - b[..., 3]
            )
            visible[y : y + rows] = body[y : y + rows] / 255 * (lum_a < lum_b)
        observations = tuple(c.observe(visible, work=work) for c in self._contracts)
        missing = sum(r["missing_samples"] for r in observations)
        rejected = ["editable-ink-gap-completed"] if new_gaps else []
        if additions:
            painted = self.guard.compare_observed(
                self._painted, actual.astype(np.float32) / 255, work=work
            )
            rejected.extend(
                sorted({"editable-ink-" + r["reason"] for r in painted["rejections"]})
            )
        _check(work)
        return {
            "version": VERSION,
            "qualified_samples": self.qualified,
            "missing_samples": missing,
            "missing_fraction": missing / max(1, self.qualified),
            "new_gap_completed": new_gaps,
            "eligible_strokes": list(ids),
            "excluded_interference": excluded,
            "raw_positive_matching_windows": True,
            "rejections": rejected,
        }
