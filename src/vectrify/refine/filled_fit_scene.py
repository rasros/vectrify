"""Document, packed parameters and projection for one filled-path fit.

Tensor-valued fields are deliberately typed as Any: importing this module must
not import the optional Torch dependency. All tensors belong to one fit call.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from PIL import Image

from vectrify.document.paint import hex_colour
from vectrify.refine import paths


class FilledEntry(NamedTuple):
    """One supported SVG layer, in the document's original painter order."""

    element: ET.Element
    contours: list[list[list[tuple[float, float]]]]
    colour: tuple[float, float, float]
    fill_rule: str
    opacity: float


@dataclass(frozen=True)
class FilledFitOptions:
    """Numerical policy shared by scene, raster preparation and optimisation."""

    xing_weight: float
    subpixels: int
    sparse_replay: bool
    max_point_displacement: float | None


@dataclass(frozen=True)
class FilledFitHooks:
    """The editor's optional callbacks, kept for exactly one fit invocation."""

    project_controls: Callable[[list[list[Any]]], Any] | None = None
    control_transform: Callable[[list[list[Any]]], list[list[Any]]] | None = None
    control_loss: Callable[[list[list[Any]]], Any] | None = None
    observe: Callable[[int, list[list[Any]], Any], bool] | None = None
    coverage_transform: Callable[[list[list[Any]], Any], Any] | None = None
    loss_transform: Callable[[Any, Any, int], tuple[Any, Any]] | None = None


@dataclass
class FilledFitScene:
    """Packed Adam parameters and their contour views, plus frozen image state."""

    root: ET.Element
    entries: list[FilledEntry]
    device: str
    soft: bool
    scale: float
    work_width: int
    work_height: int
    monolithic: bool
    coordinate_scale: Any
    control_storage: Any
    seed_controls: Any
    contour_sizes: list[int]
    controls: list[list[Any]]
    path_storage_spans: list[tuple[int, int]]
    color_storage: Any
    channels: int
    goal: Any
    under: Any
    alpha_values: Any
    context_tensors: tuple[Any, ...] | None

    def close_contours(self, options: FilledFitOptions, hooks: FilledFitHooks) -> None:
        """Restore the shared joins of every traced closed Bezier contour.

        Traced paths are closed fixed-segment loops.  The packed parameter storage
        keeps their cubic endpoints as separate Adam values for efficient
        rasterisation, so project them back to a continuous closed contour
        after each update.  Otherwise a subpixel gap becomes an extra implicit
        SVG closing cubic on export, violating the fixed-segment invariant.
        """
        import torch

        with torch.no_grad():
            if self.seed_controls is not None:
                assert options.max_point_displacement is not None
                delta = self.control_storage - self.seed_controls
                factor = (
                    options.max_point_displacement / delta.norm(dim=-1).clamp_min(1e-12)
                ).clamp_max(1)
                self.control_storage.copy_(
                    self.seed_controls + delta * factor[..., None]
                )
            if (
                hooks.project_controls is not None
                and hooks.control_transform is not None
            ):
                hooks.project_controls(self.controls)
            for path in self.controls:
                for contour in path:
                    contour[1:, 0].copy_(contour[:-1, 3])
                    contour[-1, 3].copy_(contour[0, 0])
            if hooks.project_controls is not None and hooks.control_transform is None:
                hooks.project_controls(self.controls)

    def serialize(self, max_point_displacement: float | None) -> str:
        """Write only fitted layers, undoing the exact working-raster axis scales."""
        coordinate_scale_cpu = self.coordinate_scale.cpu()
        for index, (
            (element, _contours, _colour, _fill_rule, _opacity),
            path,
        ) in enumerate(zip(self.entries, self.controls, strict=True)):
            colour = self.color_storage[index]
            data = " ".join(
                paths.to_path_d(
                    (control.detach().cpu() / coordinate_scale_cpu).tolist(),
                    precision=3 if max_point_displacement is not None else 1,
                )
                + " Z"
                for control in path
            )
            element.set("d", data)
            element.set("fill", hex_colour(colour.detach().cpu().tolist()))
            if self.alpha_values is not None:
                element.set(
                    "fill-opacity",
                    f"{float(self.alpha_values[index].detach().clamp(0, 1).cpu()):.8g}",
                )
        return ET.tostring(self.root, encoding="unicode")


def prepare_scene(
    svg: str,
    target: Image.Image,
    *,
    options: FilledFitOptions,
    hooks: FilledFitHooks,
    optimisation_long_side: int | None,
    monolithic: bool | None,
    backdrop: Image.Image | None,
    learn_alpha: bool,
    fit_context: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    device: str | None,
) -> FilledFitScene:
    """Parse supported layers and build all fit-owned tensor storage lazily."""
    import torch

    if options.max_point_displacement is not None and (
        not math.isfinite(options.max_point_displacement)
        or options.max_point_displacement < 0
    ):
        raise ValueError("max_point_displacement must be finite and nonnegative")

    root = ET.fromstring(svg)

    def opacity(element) -> float:
        """Read the directly applied SVG fill opacity, clamped for Adam."""
        try:
            fill_opacity = float(element.get("fill-opacity", "1"))
            element_opacity = float(element.get("opacity", "1"))
        except ValueError:
            return 1.0
        return min(1.0, max(0.0, fill_opacity * element_opacity))

    entries: list[FilledEntry] = []
    for element in root.iter():
        if element.tag.split("}")[-1] != "path" or not element.get("d"):
            continue
        colour = paths.fill_rgb(element.get("fill"))
        if colour is None:
            continue
        try:
            contours = paths.parse_filled_cubics(element.get("d", ""))
        except paths.UnsupportedPathError:
            continue
        fill_rule = element.get("fill-rule", "nonzero").strip().lower()
        if fill_rule not in {"evenodd", "nonzero"}:
            continue
        entries.append(
            FilledEntry(element, contours, colour, fill_rule, opacity(element))
        )
    if not entries:
        raise paths.UnsupportedPathError("no opaque filled cubic paths to optimise")

    from vectrify.refine.cuda_renderer import available as native_available

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # The native kernels take only these subpixel grids; any other layout
    # renders through the portable polyline coverage instead.
    soft = (
        device != "cuda" or not native_available() or options.subpixels not in {1, 2, 4}
    )
    width, height = target.size
    scale = (
        1.0
        if optimisation_long_side is None
        else min(1.0, optimisation_long_side / max(width, height))
    )
    work_width, work_height = round(width * scale), round(height * scale)
    if monolithic is None:
        # At a 64px fitting resolution the complete opaque-layer
        # graph is small, avoids renderer replay for each bounded layer batch,
        # and has exactly the same painter-order MSE derivative.  Preserve the
        # bounded path at larger resolutions, where its saved alpha/canvas
        # representation is intentionally memory conservative.
        monolithic = device == "cuda" and work_width * work_height <= 64 * 64
    # The target is resized to the integer working raster.  Map coordinates
    # with those exact axis scales too: applying the single nominal scale to
    # both axes subtly shifts every horizontal edge when rounding makes the
    # working raster's aspect ratio differ from the source image.
    coordinate_scale = torch.tensor(
        [work_width / width, work_height / height],
        dtype=torch.float32,
        device=device,
    )
    initial_controls = [
        [
            (
                torch.tensor(contour, dtype=torch.float32, device=device)
                * coordinate_scale
            )
            for contour in contours
        ]
        for _element, contours, _colour, _fill_rule, _opacity in entries
    ]
    # A detailed drawing has hundreds of contours.  Keeping each one as a
    # separate Adam parameter turns one optimiser update into hundreds of tiny
    # CUDA kernels.  Store equal-width contour slots in one parameter and use
    # narrow views below, retaining every original contour length in the SVG
    # and Xing terms.  The native coverage primitive itself uses 16-cubic
    # chunks, but a drawing can have longer contours; storage must
    # therefore use the document maximum rather than that renderer chunk size.
    flat_controls = [control for path in initial_controls for control in path]
    contour_sizes = [len(control) for control in flat_controls]
    storage_width = max(contour_sizes)

    def pad_storage_control(control: Any) -> Any:
        if len(control) == storage_width:
            return control
        return torch.cat(
            (
                control,
                control[:1].expand(storage_width - len(control), -1, -1),
            )
        )

    control_storage = torch.nn.Parameter(
        torch.stack([pad_storage_control(control) for control in flat_controls])
    )
    seed_controls = (
        None
        if options.max_point_displacement is None
        else control_storage.detach().clone()
    )
    controls = []
    path_storage_spans = []
    storage_offset = 0
    for path in initial_controls:
        path_start = storage_offset
        views = []
        for _control in path:
            size = contour_sizes[storage_offset]
            views.append(control_storage[storage_offset, :size])
            storage_offset += 1
        controls.append(views)
        path_storage_spans.append((path_start, storage_offset))
    # Match the fixed-width geometry storage above: one colour parameter
    # avoids launching Adam's tiny update kernels once per SVG layer.
    color_storage = torch.nn.Parameter(
        torch.tensor(
            [colour for _element, _contours, colour, _fill_rule, _opacity in entries],
            dtype=torch.float32,
            device=device,
        )
    )
    channels = 4 if fit_context is not None and fit_context[0].shape[-1] == 4 else 3
    goal_image = target.resize((work_width, work_height))
    if channels == 4:
        from vectrify.image_utils import on_white

        goal_array = np.dstack(
            (
                np.asarray(on_white(goal_image)),
                np.asarray(goal_image.convert("RGBA"))[:, :, 3],
            )
        )
    else:
        goal_array = np.asarray(goal_image.convert("RGB"))
    goal = torch.tensor(goal_array.astype(np.float32) / 255.0, device=device)
    under = (
        None
        if backdrop is None
        else torch.tensor(
            np.asarray(
                backdrop.convert("RGB").resize((work_width, work_height)),
                dtype=np.float32,
            )
            / 255.0,
            device=device,
        )
    )
    alpha_values = (
        torch.nn.Parameter(
            torch.tensor(
                [entry[4] for entry in entries],
                dtype=torch.float32,
                device=device,
            )
        )
        if learn_alpha
        else None
    )
    # An editor-selected path needs the original painter-order response,
    # including clipping and isolated group opacity, rather than a backdrop
    # with the selected path implicitly painted on top of everything.
    if hooks.loss_transform is not None and not monolithic:
        raise ValueError("Image loss transformation requires monolithic fitting")
    if hooks.control_transform is not None and not monolithic:
        raise ValueError("Control transformation requires monolithic fitting")
    if hooks.control_loss is not None and not monolithic:
        raise ValueError("Control loss requires monolithic fitting")
    if hooks.control_transform is not None and hooks.project_controls is None:
        raise ValueError("Control transformation requires a control projection")
    context_tensors = None
    if fit_context is not None:
        if not monolithic or scale != 1:
            raise ValueError("Selected-path context requires unscaled monolithic paths")
        if any(array.shape != (height, width, channels) for array in fit_context):
            raise ValueError("Selected-path context must match the target raster")
        context_tensors = tuple(
            torch.tensor(array, dtype=torch.float32, device=device)
            for array in fit_context
        )

    return FilledFitScene(
        root=root,
        entries=entries,
        device=device,
        soft=soft,
        scale=scale,
        work_width=work_width,
        work_height=work_height,
        monolithic=monolithic,
        coordinate_scale=coordinate_scale,
        control_storage=control_storage,
        seed_controls=seed_controls,
        contour_sizes=contour_sizes,
        controls=controls,
        path_storage_spans=path_storage_spans,
        color_storage=color_storage,
        channels=channels,
        goal=goal,
        under=under,
        alpha_values=alpha_values,
        context_tensors=context_tensors,
    )
