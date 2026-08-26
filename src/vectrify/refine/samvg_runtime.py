"""Small, dependency-light runtime helpers for SAMVG model loading.

Keeping device policy here lets the segmentation pipeline be tested without
loading a checkpoint and prevents CUDA assumptions leaking into CPU installs.
"""

from __future__ import annotations

from typing import Any


def pipeline_options(torch: Any, model: str) -> dict[str, Any]:
    """Return Transformers pipeline options for the available Torch device."""
    options: dict[str, Any] = {"model": model}
    if torch.cuda.is_available():
        options.update(device=0, dtype=torch.float16)
    else:
        # Transformers uses -1 for CPU; device=0 explicitly selects cuda:0.
        options["device"] = -1
    return options


def device_name(torch: Any) -> str:
    """Return the device name used by direct SAM prompt decoding."""
    return "cuda" if torch.cuda.is_available() else "cpu"
