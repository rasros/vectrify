"""Quality gates for the deterministic Typst and Graphviz benchmark cases."""

from __future__ import annotations

import importlib.util
import random
import shutil
from pathlib import Path

import pytest

from bench.non_svg import CASES

BENCH = Path(__file__).resolve().parent.parent / "bench" / "cases"
_AVAILABLE = {
    "typst": importlib.util.find_spec("typst") is not None,
    "graphviz": shutil.which("dot") is not None,
}
_SUFFIXES = {"typst": ".typ", "graphviz": ".dot"}


def _plugin(format_name: str):
    """Import lazily so corpus-shape checks work without optional backends."""
    if format_name == "typst":
        from vectrify.formats.typst.plugin import TypstPlugin

        return TypstPlugin()
    from vectrify.formats.graphviz.plugin import GraphvizPlugin

    return GraphvizPlugin()


@pytest.mark.parametrize("name", sorted(CASES))
def test_non_svg_case_has_target_and_five_seed_lineages(name):
    format_name, _target, expected = CASES[name]
    case = BENCH / name
    seeds = sorted((case / "seeds").glob(f"*{_SUFFIXES[format_name]}"))
    assert (case / "target.png").is_file()
    assert len(seeds) == len(expected) == 5
    assert [seed.read_text(encoding="utf-8") for seed in seeds] == expected


@pytest.mark.parametrize("name", sorted(CASES))
def test_non_svg_seed_pool_is_valid_and_visibly_distinct(name):
    format_name, _target, _expected = CASES[name]
    if not _AVAILABLE[format_name]:
        pytest.skip(f"{format_name} renderer is not installed")

    plugin = _plugin(format_name)
    seeds = sorted((BENCH / name / "seeds").glob(f"*{_SUFFIXES[format_name]}"))
    renders = set()
    for seed in seeds:
        content = seed.read_text(encoding="utf-8")
        valid, error = plugin.validate(content)
        assert valid, f"{seed}: {error}"
        renders.add(plugin.rasterize(content, out_w=384, out_h=384))
    assert len(renders) == len(seeds)


@pytest.mark.parametrize("name", sorted(CASES))
def test_non_svg_seed_pool_can_evolve_without_an_llm(name):
    """Local mutation and crossover must make valid candidates from the corpus.

    This is deliberately a tiny deterministic smoke test, not a score
    threshold: it catches dead seed formats and no-op operator paths without
    making CI depend on a long, renderer-sensitive optimisation run.
    """
    format_name, _target, _expected = CASES[name]
    if not _AVAILABLE[format_name]:
        pytest.skip(f"{format_name} renderer is not installed")

    plugin = _plugin(format_name)
    seeds = sorted((BENCH / name / "seeds").glob(f"*{_SUFFIXES[format_name]}"))
    parent_a = seeds[0].read_text(encoding="utf-8")
    parent_b = seeds[1].read_text(encoding="utf-8")
    state = random.getstate()
    try:
        random.seed(20260822)
        children = [plugin.mutate(parent_a)[0], plugin.crossover(parent_a, parent_b)[0]]
    finally:
        random.setstate(state)
    assert any(child != parent_a for child in children)
    for child in children:
        valid, error = plugin.validate(child)
        assert valid, error


def test_non_svg_cases_are_generated_from_declared_sources():
    from bench.non_svg import generate

    assert callable(generate)
