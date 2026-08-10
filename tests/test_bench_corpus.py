import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from vectrify.formats.svg.operations import _COLOR_ATTRS, _NUMERIC_ATTRS
from vectrify.formats.svg.plugin import SvgPlugin

BENCH = Path(__file__).resolve().parent.parent / "bench"
CASES = sorted(d for d in (BENCH / "cases").iterdir() if d.is_dir())

sys.path.insert(0, str(BENCH.parent))


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_case_has_a_target_and_a_seed(case):
    assert (case / "target.png").is_file()
    assert (case / "seed.svg").is_file()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_seed_renders(case):
    svg = (case / "seed.svg").read_text(encoding="utf-8")
    plugin = SvgPlugin()
    ok, err = plugin.validate(svg)
    assert ok, err
    assert plugin.rasterize(svg, out_w=384, out_h=384)[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_seed_geometry_is_reachable_by_an_operator(case):
    """A seed drawn with attributes no operator mutates caps the case silently.

    <polygon points> is the trap: it renders identically to a path and is
    mutable by nothing, so a case using it can never converge however long it
    runs, and the bench reports that as the search being bad.
    """
    root = ET.fromstring((case / "seed.svg").read_text(encoding="utf-8"))
    mutable = _NUMERIC_ATTRS | _COLOR_ATTRS | {"d"}
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag in {"svg", "g", "defs", "linearGradient", "radialGradient"}:
            continue
        reachable = {a.split("}")[-1] for a in el.attrib} & mutable
        assert reachable, f"<{tag}> in {case.name} has no mutable attribute"
        assert "points" not in el.attrib, f"<{tag}> in {case.name} uses points="


def test_generator_is_deterministic():
    from bench.generate import CASES as BUILDERS

    for name, (_target, seed) in BUILDERS.items():
        assert seed() == seed(), f"{name} seed.svg is not reproducible"


def test_every_case_directory_is_generated():
    from bench.generate import CASES as BUILDERS

    assert {c.name for c in CASES} == set(BUILDERS)
