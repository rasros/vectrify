"""Optional CUDA winding operator for SAMVG's fixed 16-cubic contours."""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _extension() -> Any | None:
    """Load the ahead-of-time extension when the installed wheel contains it."""
    try:
        return importlib.import_module("vectrify._samvg_cuda")
    except ImportError:
        return None


def available() -> bool:
    """Whether this installation can execute the fixed-contour CUDA path."""
    # Importing Torch first loads libtorch's shared libraries before the
    # optional extension is resolved.  Without this, a clean wheel process can
    # incorrectly report the bundled CUDA operator as unavailable.
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return _extension() is not None


def winding(
    controls: Any,
    box: tuple[int, int, int, int],
    *,
    samples: int,
    x_offset: float,
    y_offset: float,
) -> Any | None:
    """Return a differentiable native winding field, otherwise ``None``."""
    import torch

    extension = _extension()
    if (
        extension is None
        or not controls.is_cuda
        or controls.dtype != torch.float32
        or controls.ndim != 4
        or controls.shape[1:] != (16, 4, 2)
        or samples not in {8, 16, 32}
    ):
        return None
    left, top, right, bottom = box
    height, width = bottom - top, right - left

    class Winding(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            value = value.contiguous()
            ctx.save_for_backward(value)
            return extension.forward(
                value, height, width, samples, left + x_offset, top + y_offset
            )

        @staticmethod
        def backward(ctx: Any, *upstreams: Any) -> Any:
            (value,) = ctx.saved_tensors
            upstream = upstreams[0]
            return extension.backward(
                value,
                upstream.contiguous(),
                height,
                width,
                samples,
                left + x_offset,
                top + y_offset,
            )

    return Winding.apply(controls)


def windings(
    controls: Any,
    box: tuple[int, int, int, int],
    *,
    samples: int,
    subpixels: int,
) -> Any | None:
    """Return all subpixel winding fields with one native backward reduction."""
    import torch

    extension = _extension()
    if (
        extension is None
        or not controls.is_cuda
        or controls.dtype != torch.float32
        or controls.ndim != 4
        or controls.shape[1:] != (16, 4, 2)
        or samples not in {8, 16, 32}
        or subpixels not in {1, 2, 4}
    ):
        return None
    left, top, right, bottom = box
    height, width = bottom - top, right - left

    class Windings(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            value = value.contiguous()
            ctx.save_for_backward(value)
            return extension.forwards(
                value, height, width, samples, subpixels, left, top
            )

        @staticmethod
        def backward(ctx: Any, *upstreams: Any) -> Any:
            (value,) = ctx.saved_tensors
            upstream = upstreams[0]
            return extension.backwards(
                value,
                upstream.contiguous(),
                height,
                width,
                samples,
                subpixels,
                left,
                top,
            )

    return Windings.apply(controls)


def coverage(
    controls: Any,
    box: tuple[int, int, int, int],
    *,
    subpixels: int,
    fill_rule: str,
) -> Any | None:
    """Analytic cubic coverage with a boundary-local differentiable pass.

    This intentionally accepts one closed contour per batch item.  Callers
    combine holes through the winding oracle until the native multi-contour
    interface can preserve SVG fill-rule composition in one operation.
    """
    import torch

    extension = _extension()
    if (
        extension is None
        or not controls.is_cuda
        or controls.dtype != torch.float32
        or controls.ndim != 4
        or controls.shape[1:] != (16, 4, 2)
        or subpixels not in {1, 2, 4}
        or fill_rule not in {"nonzero", "evenodd"}
    ):
        return None
    left, top, right, bottom = box
    height, width = bottom - top, right - left

    class Coverage(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            value = value.contiguous()
            ctx.save_for_backward(value)
            return extension.coverage_forward(
                value, height, width, subpixels, left, top, fill_rule == "evenodd"
            )

        @staticmethod
        def backward(ctx: Any, *upstreams: Any) -> Any:
            (value,) = ctx.saved_tensors
            upstream = upstreams[0]
            return extension.coverage_backward(
                value,
                upstream.contiguous(),
                height,
                width,
                subpixels,
                left,
                top,
            )

    return Coverage.apply(controls)


def stroke_coverage(
    controls: Any,
    widths: Any,
    box: tuple[int, int, int, int],
    *,
    subpixels: int = 2,
) -> Any | None:
    """Differentiable cubic-tube coverage with round caps and joins on CUDA."""
    import torch

    extension = _extension()
    if (
        extension is None
        or not controls.is_cuda
        or controls.dtype != torch.float32
        or controls.ndim != 4
        or controls.shape[1:] != (16, 4, 2)
        or not widths.is_cuda
        or widths.dtype != torch.float32
        or widths.ndim != 1
        or widths.shape[0] != controls.shape[0]
        or subpixels not in {1, 2, 4}
    ):
        return None
    left, top, right, bottom = box
    height, width = bottom - top, right - left

    class StrokeCoverage(torch.autograd.Function):
        @staticmethod
        def forward(ctx, values, stroke_widths):
            values = values.contiguous()
            stroke_widths = stroke_widths.contiguous()
            ctx.save_for_backward(values, stroke_widths)
            return extension.stroke_forward(
                values, stroke_widths, height, width, subpixels, left, top
            )

        @staticmethod
        def backward(ctx: Any, *upstreams: Any) -> Any:
            values, stroke_widths = ctx.saved_tensors
            upstream = upstreams[0]
            control_gradients, width_gradients = extension.stroke_backward(
                values,
                stroke_widths,
                upstream.contiguous(),
                height,
                width,
                subpixels,
                left,
                top,
            )
            return control_gradients, width_gradients

    return StrokeCoverage.apply(controls, widths)


def multi_coverage_forward(
    controls: Any,
    offsets: list[int],
    box: tuple[int, int, int, int],
    *,
    subpixels: int,
    fill_rule: str,
) -> Any | None:
    """Return exact filtered multi-contour coverage without an autograd graph.

    This is used for the bounded compositing pass, whose geometry gradients
    are replayed separately.  A path's contour windings are combined before
    the fill-rule decision, retaining holes and self-overlap semantics.
    """
    import torch

    extension = _extension()
    if (
        extension is None
        or not controls.is_cuda
        or controls.dtype != torch.float32
        or controls.ndim != 4
        or controls.shape[1:] != (16, 4, 2)
        or len(offsets) < 2
        or subpixels not in {1, 2, 4}
        or fill_rule not in {"nonzero", "evenodd"}
    ):
        return None
    left, top, right, bottom = box
    return extension.multi_coverage_forward(
        controls.contiguous(),
        torch.tensor(offsets, dtype=torch.int64, device=controls.device),
        bottom - top,
        right - left,
        subpixels,
        left,
        top,
        fill_rule == "evenodd",
    )


def multi_coverage(
    controls: Any,
    offsets: list[int],
    box: tuple[int, int, int, int],
    *,
    subpixels: int,
    fill_rule: str,
    boundary_indices: Any | None = None,
    boundary_offsets: list[int] | None = None,
    topology_workspace: Any | None = None,
) -> Any | None:
    """Differentiable analytic coverage for paths made of fixed contours."""
    import torch

    extension = _extension()
    if (
        extension is None
        or not controls.is_cuda
        or controls.dtype != torch.float32
        or controls.ndim != 4
        or controls.shape[1:] != (16, 4, 2)
        or len(offsets) < 2
        or subpixels not in {1, 2, 4}
        or fill_rule not in {"nonzero", "evenodd"}
    ):
        return None
    left, top, right, bottom = box
    height, width = bottom - top, right - left
    offset_tensor = torch.tensor(offsets, dtype=torch.int64, device=controls.device)
    if boundary_indices is None:
        boundary_indices = torch.cat(
            [
                torch.arange(
                    offsets[index + 1] - offsets[index], device=controls.device
                )
                for index in range(len(offsets) - 1)
            ]
        )
    if boundary_offsets is None:
        boundary_offsets = offsets
    if (
        not isinstance(boundary_indices, torch.Tensor)
        or boundary_indices.dtype != torch.int64
        or not boundary_indices.is_cuda
        or len(boundary_offsets) != len(offsets)
        or boundary_offsets[-1] != boundary_indices.numel()
    ):
        return None
    boundary_offset_tensor = torch.tensor(
        boundary_offsets, dtype=torch.int64, device=controls.device
    )
    topology_shape = (len(offsets) - 1, height, width)
    if topology_workspace is None:
        topology_workspace = torch.empty(
            topology_shape, dtype=torch.uint16, device=controls.device
        )
    if (
        not isinstance(topology_workspace, torch.Tensor)
        or topology_workspace.dtype != torch.uint16
        or not topology_workspace.is_cuda
        or tuple(topology_workspace.shape) != topology_shape
    ):
        return None

    class MultiCoverage(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            value = value.contiguous()
            coverage, topology = extension.multi_coverage_forward_topology(
                value,
                offset_tensor,
                height,
                width,
                subpixels,
                left,
                top,
                fill_rule == "evenodd",
                topology_workspace,
            )
            ctx.save_for_backward(
                value, offset_tensor, boundary_offset_tensor, boundary_indices, topology
            )
            return coverage

        @staticmethod
        def backward(ctx: Any, *upstreams: Any) -> Any:
            (
                value,
                saved_offsets,
                saved_boundary_offsets,
                saved_boundary_indices,
                topology,
            ) = ctx.saved_tensors
            upstream = upstreams[0]
            return extension.multi_coverage_backward_topology(
                value,
                saved_offsets,
                saved_boundary_offsets,
                saved_boundary_indices,
                topology,
                upstream.contiguous(),
                height,
                width,
                subpixels,
                left,
                top,
            )

    return MultiCoverage.apply(controls)
