"""Small helpers the editor's backend and its agent modules share."""

from __future__ import annotations

import io
import math
from collections.abc import Callable
from typing import Any

from PIL import Image

from vectrify.document import DocumentError, StaleRevisionError

# What a session refuses a request with: anything else is a bug.
REFUSALS = (DocumentError, ValueError, OSError)


def finite(value: Any, what: str) -> float:
    """*value* as a finite float, refused with *what* named otherwise."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DocumentError(f"{what} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise DocumentError(f"{what} must be finite")
    return result


def png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def refusal(
    exc: BaseException, describe: Callable[[BaseException], str] = str
) -> tuple[int, dict[str, Any]]:
    """The status and error body a refused request answers with.

    A stale revision is a conflict; anything else is a bad request, worded by
    *describe*, and says where the drawing is when the refusal made a revision.
    """
    from vectrify.ui.agent import RefusedError

    if isinstance(exc, StaleRevisionError):
        return 409, {"error": str(exc)}
    if isinstance(exc, RefusedError):
        return 400, {"error": str(exc), "where": exc.where}
    return 400, {"error": describe(exc)}
