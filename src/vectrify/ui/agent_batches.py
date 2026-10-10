"""Private edit plans and their before/after inspection views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from vectrify.document import Document, DocumentError, Snapshot, export_svg
from vectrify.operations.diagnostics import edit_diagnostics


@dataclass
class Batch:
    epoch: str
    base: Snapshot
    after: Document
    label: str
    reference: dict | None
    uses_reference: bool
    summary: dict[str, Any]
    images: list[tuple[str, bytes]]


def views(
    agent: Any, before: Document, after: Document, region: Any, close_region: Any
) -> tuple[dict, list[tuple[str, bytes]]]:
    from vectrify.ui.agent import (
        _box,
        _png,
        _size,
        crop_reference,
        heat_map,
        render_document,
    )
    from vectrify.ui.agent_look import mapping

    box = _box(region) if region is not None else before.artboard()
    if close_region is None:
        close_region = [
            box[0] + box[2] / 4,
            box[1] + box[3] / 4,
            box[2] / 2,
            box[3] / 2,
        ]
    close = _box(close_region, "close_region")
    images = []
    mappings = {}
    reference = agent._reference_image()
    for name, bounds, side in (("normal", box, 512), ("close", close, 1024)):
        size = _size(bounds, side)
        a = render_document(export_svg(before), bounds, size)
        b = render_document(export_svg(after), bounds, size)
        ref = (
            crop_reference(reference, before.artboard(), bounds, size)
            if reference is not None
            else Image.new("RGB", size, "white")
        )
        difference = np.sqrt(
            ((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) / 255) ** 2
        ).mean(axis=2)
        for kind, picture in (
            ("before", a),
            ("after", b),
            ("reference", ref),
            ("difference", heat_map(difference)),
        ):
            images.append((f"{name}_{kind}", _png(picture)))
        mappings[name] = mapping(bounds, size)
    return {
        "diagnostics": edit_diagnostics(before, after),
        "mapping": mappings,
        "reference_available": reference is not None,
        "difference": "Absolute RGB change between before and after",
    }, images


def resolve(value: Any, aliases: dict[str, list[str]]) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        if value[1:] not in aliases:
            raise DocumentError(f"Unknown batch alias: {value}")
        ids = aliases[value[1:]]
        if len(ids) != 1:
            raise DocumentError("A scalar batch alias must identify one object")
        return ids[0]
    if isinstance(value, list):
        return [resolve(v, aliases) for v in value]
    if isinstance(value, dict):
        return {k: resolve(v, aliases) for k, v in value.items()}
    return value
