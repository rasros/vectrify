"""Fitting a linear gradient fill to the reference, in closed form."""

import math

import numpy as np
import pytest

from tests.operations.test_colours_cleanup import render, request
from vectrify.document import Editor, Selection, import_svg
from vectrify.document.paint import gradient_stops
from vectrify.document.redraw import root_matrix
from vectrify.operations import Job, Permissions, method

# The shape sits in a scaled, skewed group so the fit has to map the ramp
# from root user space into the shape's own.
DOC = (
    '<svg width="100" height="100" viewBox="0 0 100 100">'
    '<rect id="bg" width="100" height="100" fill="#ffffff"/>'
    '<g transform="translate(10 15) scale(1.25 0.8) skewX(10)">'
    '<path id="a" d="M0 0 L56 0 L56 65 L0 65 Z" fill="#808080"{stroke}/></g></svg>'
)
# The reference's ramp, in root user space: dark red to pale blue.
START, END = (0.0, 5.0), (100.0, 80.0)
COLOURS = ((0.7, 0.1, 0.1), (0.3, 0.6, 0.9))


def ramp_reference(size=100):
    """The drawing painted with the known ramp, rendered by cairosvg."""
    (x1, y1), (x2, y2) = START, END
    hexes = ["#" + "".join(f"{round(v * 255):02x}" for v in c) for c in COLOURS]
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" '
        'viewBox="0 0 100 100"><defs><linearGradient id="r" '
        f'gradientUnits="userSpaceOnUse" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}">'
        f'<stop offset="0" stop-color="{hexes[0]}"/>'
        f'<stop offset="1" stop-color="{hexes[1]}"/></linearGradient>'
        '<clipPath id="c"><path transform="translate(10 15) scale(1.25 0.8) '
        'skewX(10)" d="M0 0 L56 0 L56 65 L0 65 Z"/></clipPath></defs>'
        '<rect width="100" height="100" fill="#ffffff"/>'
        '<rect width="100" height="100" fill="url(#r)" clip-path="url(#c)"/></svg>'
    )
    return render(svg, size)


def expected(point):
    (x1, y1), (x2, y2) = START, END
    dx, dy = x2 - x1, y2 - y1
    t = ((point[0] - x1) * dx + (point[1] - y1) * dy) / (dx * dx + dy * dy)
    t = min(1.0, max(0.0, t))
    return tuple(a + (b - a) * t for a, b in zip(*COLOURS, strict=True))


def fit(editor, fill="linear"):
    job = Job(
        method("improve", "colours"),
        request(
            editor,
            "colours",
            Selection(object_ids=frozenset({"a"})),
            Permissions(paint=True),
            ramp_reference(),
            resolution=100,
            fill=fill,
        ),
    )
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    return job, state["result"]["metrics"]


def to_root(document, point):
    a, b, c, d, e, f = root_matrix(document, "a")
    return a * point[0] + c * point[1] + e, b * point[0] + d * point[1] + f


def test_linear_fit_recovers_a_known_gradient():
    editor = Editor(import_svg(DOC.format(stroke="")))
    job, metrics = fit(editor)
    assert metrics["gradients"] == 1
    assert metrics["after"]["error"] < metrics["before"]["error"] / 20
    job.apply()
    document = editor.snapshot.document
    fill = document.element("a").get("fill") or ""
    assert fill.startswith("url(#")
    gradient = document.element(fill[5:-1])
    assert gradient.get("gradientUnits") == "userSpaceOnUse"
    # Endpoints are in the shape's own space; its ramp runs parallel to the
    # known one in root space, and its stops match the known colours there.
    numbers = [float(gradient.get(k) or 0) for k in ("x1", "y1", "x2", "y2")]
    start = to_root(document, numbers[:2])
    end = to_root(document, numbers[2:])
    # The gradient's parameter changes along (lx, ly) in local space; in root
    # space it changes along M^-T of that, which must be the known axis.
    a, b, c, d, _, _ = root_matrix(document, "a")
    lx, ly = numbers[2] - numbers[0], numbers[3] - numbers[1]
    det = a * d - b * c
    gx = (d * lx - b * ly) / det
    gy = (-c * lx + a * ly) / det
    known = (END[0] - START[0], END[1] - START[1])
    cosine = abs(gx * known[0] + gy * known[1]) / (
        math.hypot(gx, gy) * math.hypot(*known)
    )
    assert cosine > 0.999
    stops = gradient_stops(gradient)
    assert len(stops) == 2
    for (_, rgba), point in zip(stops, (start, end), strict=True):
        assert np.allclose(rgba[:3], expected(point), atol=4 / 255)
    assert editor.undo_labels == ("Fit gradients",)
    editor.undo()
    assert editor.snapshot.document.element("a").get("fill") == "#808080"
    assert not any(
        e.tag == "linearGradient" for e in editor.snapshot.document.elements()
    )


def test_refitting_reuses_the_gradient_and_a_flat_fit_removes_it():
    editor = Editor(import_svg(DOC.format(stroke=' stroke="#808080"')))
    fit(editor)[0].apply()
    document = editor.snapshot.document
    fill = document.element("a").get("fill")
    assert document.element("a").get("stroke") == fill
    gradients = [e.id for e in document.elements() if e.tag == "linearGradient"]
    assert len(gradients) == 1
    again, _ = fit(editor)
    if again.state()["result"]["changed"]:
        again.apply()
    document = editor.snapshot.document
    assert document.element("a").get("fill") == fill
    assert [e.id for e in document.elements() if e.tag == "linearGradient"] == (
        gradients
    )
    flat_job, _ = fit(editor, "flat")
    flat_job.apply()
    document = editor.snapshot.document
    assert not (document.element("a").get("fill") or "").startswith("url(")
    assert document.element("a").get("stroke") == document.element("a").get("fill")
    assert not any(e.tag == "linearGradient" for e in document.elements())


def test_an_even_fill_stays_flat():
    flat = import_svg(DOC.format(stroke="").replace("#808080", "#3366aa"))
    editor = Editor(import_svg(DOC.format(stroke="")))
    job = Job(
        method("improve", "colours"),
        request(
            editor,
            "colours",
            Selection(object_ids=frozenset({"a"})),
            Permissions(paint=True),
            render(
                DOC.format(stroke="")
                .replace("#808080", "#3366aa")
                .replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ')
            ),
            resolution=100,
            fill="linear",
        ),
    )
    job.run()
    state = job.state()
    assert state["status"] == "ready", state
    assert state["result"]["metrics"]["gradients"] == 0
    job.apply()
    fill = editor.snapshot.document.element("a").get("fill")
    assert fill == flat.element("a").get("fill")


def test_linear_fit_finds_a_ramp_that_starts_and_ends_inside_the_shape():
    # The reference ramps only across the middle of the square, x 40 to 60,
    # flat either side: the fitted ends land there, not at the square's edges.
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" '
        'viewBox="0 0 100 100"><defs><linearGradient id="r" '
        'gradientUnits="userSpaceOnUse" x1="40" y1="0" x2="60" y2="0">'
        '<stop offset="0" stop-color="#b22626"/>'
        '<stop offset="1" stop-color="#408ce6"/></linearGradient></defs>'
        '<rect width="100" height="100" fill="#ffffff"/>'
        '<rect x="10" y="10" width="80" height="80" fill="url(#r)"/></svg>'
    )
    editor = Editor(
        import_svg(
            '<svg width="100" height="100" viewBox="0 0 100 100">'
            '<rect id="bg" width="100" height="100" fill="#ffffff"/>'
            '<path id="a" d="M10 10 L90 10 L90 90 L10 90 Z" fill="#808080"/></svg>'
        )
    )
    job = Job(
        method("improve", "colours"),
        request(
            editor,
            "colours",
            Selection(object_ids=frozenset({"a"})),
            Permissions(paint=True),
            render(svg),
            resolution=100,
            fill="linear",
        ),
    )
    job.run()
    job.apply()
    document = editor.snapshot.document
    fill = document.element("a").get("fill") or ""
    gradient = document.element(fill[5:-1])
    xs = sorted(float(gradient.get(k) or 0) for k in ("x1", "x2"))
    ys = [float(gradient.get(k) or 0) for k in ("y1", "y2")]
    assert abs(xs[0] - 40) < 1.5
    assert abs(xs[1] - 60) < 1.5
    assert abs(ys[0] - ys[1]) < 1


def test_the_batched_ramp_errors_match_one_at_a_time():
    from vectrify.operations.methods.colours import _ramp_error, _ramp_errors

    rng = np.random.default_rng(3)
    s = rng.uniform(-5, 5, 300)
    cov = rng.uniform(0, 1, (300, 3))
    values = rng.uniform(0, 1, (300, 3))
    t0 = np.array([-5.0, -2.0, 0.0, 1.0])
    t1 = np.array([5.0, 3.0, 0.5, 4.0])
    batched = _ramp_errors(s, cov, values, t0, t1)
    single = [_ramp_error(s, cov, values, a, b)[0] for a, b in zip(t0, t1, strict=True)]
    assert batched == pytest.approx(single, rel=1e-9)
