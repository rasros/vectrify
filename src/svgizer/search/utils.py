import random

from svgizer.search.models import SearchNode


def calculate_elite_prob(progress01: float, p_start: float, p_end: float) -> float:
    progress01 = max(0.0, min(1.0, progress01))
    return float(p_start + (p_end - p_start) * progress01)


def choose_from_top_k_weighted(best_k: list[SearchNode]) -> int:
    if not best_k:
        return 0
    weights = [1.0 / (i + 1.0) for i in range(len(best_k))]
    node = random.choices(best_k, weights=weights, k=1)[0]
    return node.id
