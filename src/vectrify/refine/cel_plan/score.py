"""Native-render measurements and representation costs for planned drawings.

Raw reference MSE is reported separately from the planning objective. All
contours count, including disconnected pieces packed into compound paths.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass

import cairosvg
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter

from vectrify.document import Document, import_svg
from vectrify.document.model import paint_server
from vectrify.refine.crossings import crossings

SCORE_VERSION = 3


@dataclass(frozen=True)
class Representation:
    paths: int
    contours: int
    nodes: int
    gradients: int
    stroke_contours: int
    groups: int
    primitive_objects: int = 0

    @property
    def cost(self) -> float:
        return (
            self.nodes
            + 4 * self.contours
            + 2 * (self.paths + self.primitive_objects)
            + 12 * self.gradients
        )

    def metrics(self) -> dict:
        return {**asdict(self), "representation_cost": self.cost}


def representation(document: Document) -> Representation:
    """Charge each visible instance, including geometry referenced by use."""
    objects = [
        e
        for e in document.elements()
        if e.tag in {"path", "use", "rect", "circle", "ellipse", "line"}
        and not any(
            a.tag in {"defs", "clipPath", "mask"} for a in document.ancestry(e.id)
        )
    ]

    def asset(element):
        while element.tag == "use":
            element = document.element((element.get("href") or "")[1:])
        return element

    paths = [element for element in objects if asset(element).tag == "path"]
    primitives = [element for element in objects if asset(element).tag != "path"]
    geometries = [document.geometry_for(e.id) for e in paths]

    def style(element, attribute):
        referenced_style = None
        referenced = element
        while referenced.tag == "use":
            referenced = document.element((referenced.get("href") or "")[1:])
            if referenced.get(attribute) is not None:
                referenced_style = referenced.get(attribute)
        if referenced_style is not None:
            return referenced_style
        for ancestor in reversed(document.ancestry(element.id)):
            value = ancestor.get(attribute)
            if value is not None:
                return value
        return None

    paints = {
        reference
        for element in objects
        for attribute in ("fill", "stroke")
        if (reference := paint_server(style(element, attribute))) is not None
    }
    return Representation(
        paths=len(paths),
        contours=sum(len(g.subpaths) for g in geometries) + len(primitives),
        nodes=sum(len(s.nodes) for g in geometries for s in g.subpaths),
        gradients=len(paints),
        stroke_contours=sum(
            len(document.geometry_for(e.id).subpaths)
            for e in paths
            if style(e, "stroke") not in {None, "none"}
        )
        + sum(style(e, "stroke") not in {None, "none"} for e in primitives),
        groups=sum(e.tag == "g" for e in document.elements()),
        primitive_objects=len(primitives),
    )


def render(svg: str, size: tuple[int, int]) -> np.ndarray:
    png = cairosvg.svg2png(
        bytestring=svg.encode(), output_width=size[0], output_height=size[1]
    )
    if png is None:
        raise ValueError("The SVG renderer returned no pixels")
    with Image.open(io.BytesIO(png)) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.float32) / 255


def premultiplied(rgba: np.ndarray) -> np.ndarray:
    return np.concatenate((rgba[..., :3] * rgba[..., 3:], rgba[..., 3:]), axis=-1)


def composite(rgba: np.ndarray, background: float = 1) -> np.ndarray:
    return rgba[..., :3] * rgba[..., 3:] + background * (1 - rgba[..., 3:])


def foreground_mask(rgba: np.ndarray, dilation: int = 4) -> np.ndarray:
    mask = rgba[..., 3] >= 0.5
    return binary_dilation(mask, iterations=dilation) if dilation else mask


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].mean()) if mask.any() else 0.0


def measurements(
    actual: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    *,
    features: dict[str, tuple[int, int, int, int]] | None = None,
) -> dict:
    """Compare the same pixels at native size, keeping raw RGB and alpha apart."""
    if actual.shape != truth.shape or mask.shape != truth.shape[:2]:
        raise ValueError("Comparison images and frozen mask must have matching sizes")
    # Float64 in 0-255 reproduces the original sword diagnostic exactly.
    rgb = composite(actual).astype(np.float64) * 255
    target = composite(truth).astype(np.float64) * 255
    squared = np.square(rgb - target)
    blurred = np.square(
        gaussian_filter(rgb, (2, 2, 0)) - gaussian_filter(target, (2, 2, 0))
    )
    source_alpha, actual_alpha = truth[..., 3] >= 0.5, actual[..., 3] >= 0.5
    union = source_alpha | actual_alpha
    feature_metrics = {}
    for name, (x, y, width, height) in (features or {}).items():
        if x < 0 or y < 0 or width < 1 or height < 1:
            raise ValueError("Feature rectangles must be positive and inside the image")
        if x + width > mask.shape[1] or y + height > mask.shape[0]:
            raise ValueError("Feature rectangle lies outside the image")
        box = np.s_[y : y + height, x : x + width]
        feature_metrics[name] = masked_mean(squared[box], mask[box])
    return {
        "mse": masked_mean(squared, mask),
        "blur2_mse": masked_mean(blurred, mask),
        "premultiplied_error": masked_mean(
            np.square(premultiplied(actual) - premultiplied(truth)), mask
        ),
        "alpha_iou": float((source_alpha & actual_alpha).sum() / union.sum())
        if union.any()
        else 1.0,
        "alpha_missing_pixels": int((source_alpha & ~actual_alpha).sum()),
        "alpha_spill_pixels": int((actual_alpha & ~source_alpha).sum()),
        "features": feature_metrics,
    }


def opacity_measurements(actual: np.ndarray, truth: np.ndarray) -> dict:
    """Native translucent-content diagnostics, separate from frozen benchmarks."""
    from vectrify.refine.cel_plan.opacity import VISIBLE

    support = truth[..., 3] > VISIBLE
    mask = binary_dilation(support, iterations=4)
    return {
        "support_pixels": int(support.sum()),
        "alpha_mse": masked_mean(np.square(actual[..., 3] - truth[..., 3]), mask),
        "premultiplied_mse": masked_mean(
            np.square(premultiplied(actual) - premultiplied(truth)), mask
        ),
        "black_mse": masked_mean(
            np.square(composite(actual, 0) - composite(truth, 0)), mask
        ),
        "white_mse": masked_mean(np.square(composite(actual) - composite(truth)), mask),
    }


def svg_metrics(svg: str, *, include_crossings: bool = False) -> dict:
    document = import_svg(svg)
    result = representation(document).metrics()
    if include_crossings:
        result["self_crossings"] = sum(
            crossings(document.geometry_for(e.id))
            for e in document.elements()
            if e.tag in {"path", "use"}
            and not any(
                a.tag in {"defs", "clipPath", "mask"} for a in document.ancestry(e.id)
            )
            and _path_asset(document, e.id)
        )
    return result


def _path_asset(document: Document, element_id: str) -> bool:
    element = document.element(element_id)
    while element.tag == "use":
        element = document.element((element.get("href") or "")[1:])
    return element.tag == "path"
