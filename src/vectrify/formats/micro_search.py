"""Shared micro-search: try N candidate edits, keep the best-rendering one.

Every format backend improves a candidate the same way — generate a batch of
random edits, rasterize each, and keep whichever lands closest to the target.
Only the renderer and the edit generator differ, so both are parameters.
"""

import io
from collections.abc import Callable

from PIL import Image

from vectrify.score.utils import lab_l1
from vectrify.search.models import INVALID_SCORE

MAX_DISTANCE = 1.0


def fast_lab_l1(reference: Image.Image, candidate_png: bytes) -> float:
    """Perceptual distance between *reference* and a rendered candidate.

    The candidate is resized to the reference's exact dimensions: lab_l1
    compares pixel-wise and renderers pick their own output size, so without
    this every comparison of a differently-shaped render would fail.
    Returns MAX_DISTANCE if the candidate cannot be read.
    """
    try:
        candidate = Image.open(io.BytesIO(candidate_png)).convert("RGB")
        if candidate.size != reference.size:
            candidate = candidate.resize(reference.size, Image.Resampling.LANCZOS)
        return lab_l1(reference, candidate)
    except Exception:
        return MAX_DISTANCE


def with_micro_search(
    op_generator: Callable[[], tuple[str, str]],
    fallback: str,
    rasterize: Callable[[str], bytes | None],
    orig_img_fast: Image.Image,
    num_trials: int = 15,
    default_summary: str = "No change",
) -> tuple[str, str]:
    """Return the (content, summary) whose render is closest to the target.

    *op_generator* yields a (candidate, summary) pair per trial. Candidates
    identical to *fallback* and ones *rasterize* cannot render are skipped;
    if nothing survives, (fallback, default_summary) is returned.
    """
    best_content: str | None = None
    best_score = INVALID_SCORE
    best_summary = default_summary

    for _ in range(num_trials):
        candidate, summary = op_generator()
        if candidate == fallback:
            continue

        png = rasterize(candidate)
        if png is None:
            continue

        score = fast_lab_l1(orig_img_fast, png)
        if score < best_score:
            best_score = score
            best_content = candidate
            best_summary = summary

    return (best_content if best_content is not None else fallback), best_summary
