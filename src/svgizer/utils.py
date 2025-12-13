import difflib
import logging
import random
from typing import List, Optional

from svgizer.models import SearchNode

STALENESS_THRESHOLD = 0.995


def setup_logger(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(processName)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def is_stale(prev_svg: Optional[str], new_svg: str) -> bool:
    if prev_svg is None:
        return False
    if prev_svg == new_svg:
        return True
    ratio = difflib.SequenceMatcher(None, prev_svg, new_svg).ratio()
    return ratio >= STALENESS_THRESHOLD


def calculate_elite_prob(progress01: float, p_start: float, p_end: float) -> float:
    progress01 = max(0.0, min(1.0, progress01))
    return float(p_start + (p_end - p_start) * progress01)


def choose_from_top_k_weighted(best_k: List[SearchNode]) -> int:
    if not best_k:
        return 0
    weights = [1.0 / (i + 1.0) for i in range(len(best_k))]
    node = random.choices(best_k, weights=weights, k=1)[0]
    return node.id