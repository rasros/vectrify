"""SVG complexity scoring. Lower is simpler (better in multi-objective search)."""


def svg_complexity(svg: str | None) -> float:
    """Returns the UTF-8 byte length of the SVG string as a raw complexity value.

    Used as the second objective alongside visual quality in NSGA-II selection.
    Returns 0.0 for None/empty SVGs (e.g. the initial sentinel node).
    """
    if not svg:
        return 0.0
    return float(len(svg.encode()))
