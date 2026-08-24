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
        def backward(ctx, upstream):
            (value,) = ctx.saved_tensors
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
