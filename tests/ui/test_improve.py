"""Selected path fitting keeps editor scope, constraints, and revision safety."""

import base64
import io
import time
from dataclasses import replace
from threading import Event

import cairosvg
import numpy as np
import pytest
from PIL import Image

from vectrify.document import (
    DocumentError,
    Editor,
    Selection,
    StaleRevisionError,
    import_svg,
)
from vectrify.refine.selected import (
    FitContext,
    FitOptions,
    FitResult,
    fit_selected_path,
    validate_selection,
)
from vectrify.ui.session import Session

SVG = """<svg width="64" height="64">
<defs><clipPath id="clip"><path d="M8 8 H55 V55 H8 Z"/></clipPath></defs>
<rect width="64" height="64" fill="#eeeeff"/>
<g opacity="0.7" clip-path="url(#clip)" transform="translate(1 2)">
<path id="a" fill="#800000" fill-rule="evenodd"
 d="M12 12 L40 12 L40 40 L12 40 Z M20 20 L25 20 L25 25 L20 25 Z"/></g>
<rect id="front" x="30" y="25" width="30" height="10" fill="blue"/></svg>"""
SELECTION = Selection(object_ids=frozenset({"a"}))


@pytest.fixture(autouse=True)
def fit_ready(monkeypatch):
    """These tests stub the fit itself, so they need no PyTorch to pass the check."""
    monkeypatch.setattr(
        "vectrify.operations.methods.path_fit.fit_problem", lambda: None
    )


def target(svg=SVG):
    data = cairosvg.svg2png(bytestring=svg.encode(), background_color="white")
    assert data is not None
    return Image.open(io.BytesIO(data))


def reference(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return {
        "name": "target.png",
        "opacity": 0.5,
        "data_url": "data:image/png;base64,"
        + base64.b64encode(stream.getvalue()).decode(),
    }


def test_context_preserves_clipping_opacity_transform_and_front_objects():
    context = FitContext(import_svg(SVG), SELECTION, target(), FitOptions())
    # Recover exact coverage from a white fill on transparent isolated path,
    # and compare the compositing response with another actual SVG render.
    context.path.set("fill", "#ffffff")
    white = context.array(context.render())
    context.path.set("fill", "#000000")
    black = context.array(context.render())
    alpha = np.divide(
        white - black,
        context.transmission,
        out=np.zeros_like(white),
        where=context.transmission > 0.01,
    )
    context.path.set("fill", "#5588bb")
    expected = context.array(context.render())
    reconstructed = context.base + alpha * (
        context.delta + context.transmission * np.array([85, 136, 187]) / 255
    )
    # Cairo's intermediate 8-bit rounding contributes up to a few levels.
    assert np.max(abs(expected - reconstructed)) < 0.025


@pytest.mark.parametrize(
    "options",
    [
        {"steps": 0},
        {"steps": 1001},
        {"nodes": False, "handles": False, "color": False},
        {"displacement": float("nan")},
        {"resolution": 4096},
    ],
)
def test_invalid_fit_options(options):
    with pytest.raises(DocumentError):
        FitOptions(**options)


def test_fit_scope_locks_and_shared_dependencies():
    doc = import_svg(SVG)
    with pytest.raises(DocumentError, match="one path"):
        validate_selection(doc, Selection(), FitOptions())
    editor = Editor(doc, selection=SELECTION)
    editor.set_locks("a", frozenset({"geometry"}))
    with pytest.raises(DocumentError, match="locked"):
        validate_selection(editor.snapshot.document, SELECTION, FitOptions())
    validate_selection(
        editor.snapshot.document,
        SELECTION,
        FitOptions(nodes=False, handles=False, color=True),
    )
    linked = import_svg(SVG.replace("</svg>", '<use href="#a"/></svg>'))
    with pytest.raises(DocumentError, match="referenced"):
        validate_selection(linked, SELECTION, FitOptions())


def test_bulk_node_proposal_preserves_pins_and_undo():
    doc = import_svg(SVG)
    editor = Editor(doc, selection=SELECTION)
    node = doc.geometry_for("a").subpaths[0].nodes[1]
    editor.pin_node("a", node.id)
    with pytest.raises(DocumentError, match="pinned"), editor.transaction("Fit") as tx:
        tx.update_nodes("a", {node.id: (41, 12)})
    assert (
        editor.snapshot.document.geometry_for("a").node(node.id).endpoint
        == node.endpoint
    )


def start_fit(session, **settings):
    return session.operation(
        {
            "command": "start",
            "action": "improve",
            "method": "path-fit",
            "epoch": session.epoch,
            "revision": session.editor.snapshot.revision,
            "permissions": {"paint": True},
            "settings": {"nodes": False, "handles": False, "color": True, **settings},
        }
    )


def wait(session, job):
    for _ in range(500):
        state = session.operation({"command": "status", "job": job["id"]})
        if state["status"] != "running":
            return state
        time.sleep(0.01)
    pytest.fail("Operation did not finish")


def test_job_apply_is_single_undoable_edit_and_rejects_stale_reference(monkeypatch):
    result = FitResult("a", {}, "#bb0000", 0.2, 0.1, {}, 8, (64, 64))
    monkeypatch.setattr(
        "vectrify.operations.methods.path_fit.fit_selected_path",
        lambda *_a, **_kw: result,
    )
    session = Session(import_svg(SVG), reference=reference(target()))
    session.editor.select(SELECTION)
    job = start_fit(session)
    state = wait(session, job)
    assert state["status"] == "ready", state
    assert state["result"]["metrics"]["after"] == {"error": 0.1}
    before = session.editor.snapshot.document
    session.operation({"command": "apply", "job": job["id"]})
    assert session.editor.undo_labels == ("Optimize path",)
    assert session.editor.snapshot.document.element("a").get("fill") == "#bb0000"
    session.editor.undo()
    assert session.editor.snapshot.document == before
    with pytest.raises(DocumentError, match="expired"):
        session.operation({"command": "apply", "job": job["id"]})
    fresh = start_fit(session)
    assert wait(session, fresh)["status"] == "ready"
    session.reference = reference(Image.new("RGB", (64, 64), "white"))
    with pytest.raises(StaleRevisionError, match="reference"):
        session.operation({"command": "apply", "job": fresh["id"]})


def test_job_result_is_stale_after_an_edit(monkeypatch):
    result = FitResult("a", {}, "#bb0000", 0.2, 0.1, {}, 8, (64, 64))
    monkeypatch.setattr(
        "vectrify.operations.methods.path_fit.fit_selected_path",
        lambda *_a, **_kw: result,
    )
    session = Session(import_svg(SVG), reference=reference(target()))
    session.editor.select(SELECTION)
    job = start_fit(session)
    wait(session, job)
    with session.editor.transaction("Paint") as tx:
        tx.set_attributes("a", {"fill": "#00ff00"})
    with pytest.raises(StaleRevisionError):
        session.operation({"command": "apply", "job": job["id"]})
    assert session.editor.snapshot.document.element("a").get("fill") == "#00ff00"


def test_fit_permissions_are_checked_before_the_job_starts():
    session = Session(import_svg(SVG), reference=reference(target()))
    session.editor.select(SELECTION)
    with pytest.raises(DocumentError, match="paint"):
        session.operation(
            {
                "command": "start",
                "action": "improve",
                "method": "path-fit",
                "epoch": session.epoch,
                "revision": 0,
                "settings": {"nodes": False, "handles": False, "color": True},
            }
        )
    assert session.jobs == {}


def test_job_cancel_does_not_change_document(monkeypatch):
    from vectrify.operations import Job, OperationRequest, Permissions, method

    session = Session(import_svg(SVG), reference=reference(target()))
    session.editor.select(SELECTION)
    job = Job(
        method("improve", "path-fit"),
        OperationRequest(
            action="improve",
            method="path-fit",
            snapshot=session.editor.snapshot,
            editor=session.editor,
            permissions=Permissions(geometry=True, paint=True),
            reference=target(),
        ),
    )
    monkeypatch.setattr(
        "vectrify.operations.methods.path_fit.fit_selected_path",
        lambda *_a, **_kw: pytest.fail("Cancelled job should not fit"),
    )
    job.stop.set()
    job.run()
    assert job.state()["status"] == "cancelled"
    assert session.editor.snapshot.revision == 0


def require_gpu():
    torch = pytest.importorskip("torch")
    from vectrify.refine.cuda_renderer import available

    if not torch.cuda.is_available() or not available():
        pytest.skip("CUDA extension unavailable")


def test_existing_fitter_improves_selected_fill_with_geometry_locked():
    require_gpu()
    doc = import_svg(SVG)
    result = fit_selected_path(
        doc,
        SELECTION,
        target(SVG.replace("#800000", "#dd0000")),
        FitOptions(nodes=False, handles=False, color=True, steps=20, resolution=64),
    )
    assert result.after < result.before * 0.7
    assert not result.values
    editor = Editor(doc, selection=SELECTION)
    with editor.transaction("Optimize path", selection=SELECTION) as tx:
        result.write(tx)
    assert editor.snapshot.document.geometries == doc.geometries
    assert editor.snapshot.document.element("front") == doc.element("front")
    editor.undo()
    assert editor.snapshot.document == doc


def test_gpu_fit_preserves_pins_holes_node_filter_and_displacement():
    require_gpu()
    doc = import_svg(SVG)
    geometry = doc.geometry_for("a")
    node = geometry.subpaths[0].nodes[0]
    doc = doc.replace_geometry(geometry.replace_node(replace(node, pinned=True)))
    other = geometry.subpaths[0].nodes[1]
    selection = replace(SELECTION, node_ids=frozenset({node.id, other.id}))
    result = fit_selected_path(
        doc,
        selection,
        target(SVG.replace("L40 12", "L42 12")),
        FitOptions(color=False, steps=12, resolution=64, displacement=1),
    )
    assert node.id not in result.values
    assert set(result.values) <= {other.id}
    for node_id, values in result.values.items():
        old = geometry.node(node_id).values
        assert np.linalg.norm(np.array(values) - old) <= 1.00001
    editor = Editor(doc, selection=selection)
    with editor.transaction("Optimize path", selection=selection) as tx:
        result.write(tx)
    assert len(editor.snapshot.document.geometry_for("a").subpaths) == 2
    assert result.after <= result.before


def test_stopping_returns_unchanged_or_better_candidate():
    require_gpu()
    stop = Event()

    def progress(step, _message):
        if step == 2:
            stop.set()

    doc = import_svg(SVG)
    result = fit_selected_path(
        doc,
        SELECTION,
        target(),
        FitOptions(steps=50, resolution=64),
        stop=stop,
        progress=progress,
    )
    assert result.steps < 50
    assert result.after <= result.before


@pytest.mark.parametrize("join", ["round", "miter"])
def test_outlined_region_fits_fill_and_stroke_together(join):
    require_gpu()
    outlined = SVG.replace(
        'fill="#800000"',
        f'fill="#800000" stroke="#800000" stroke-width="3" stroke-linejoin="{join}"',
    )
    doc = import_svg(outlined)
    result = fit_selected_path(
        doc,
        SELECTION,
        target(outlined.replace("#800000", "#cc0000")),
        FitOptions(nodes=False, handles=False, color=True, steps=12, resolution=64),
    )
    assert result.changed
    assert result.after < result.before
    assert result.fill == result.stroke
    assert not result.values
    editor = Editor(doc, selection=SELECTION)
    with editor.transaction("Optimize path", selection=SELECTION) as tx:
        result.write(tx)
    fitted = editor.snapshot.document.element("a")
    assert fitted.get("fill") == fitted.get("stroke") == result.fill


def test_transparent_reference_is_composited_on_white():
    context = FitContext(
        import_svg(SVG),
        SELECTION,
        Image.new("RGBA", (64, 64), (0, 0, 0, 0)),
        FitOptions(),
    )
    assert np.asarray(context.target).min() == 255


def test_miter_geometry_fit_preserves_sharp_join_and_improves_reference_match():
    require_gpu()
    svg = (
        '<svg width="64" height="64"><path id="a" fill="#800000" '
        'stroke="#800000" stroke-width="6" stroke-miterlimit="8" '
        'd="M32 20 L40 52 L24 52 Z"/></svg>'
    )
    doc = import_svg(svg)
    result = fit_selected_path(
        doc,
        SELECTION,
        target(svg.replace("M32 20", "M34 20")),
        FitOptions(color=False, steps=12, resolution=64),
    )
    assert result.values
    assert result.after < result.before
    editor = Editor(doc, selection=SELECTION)
    with editor.transaction("Optimize path", selection=SELECTION) as tx:
        result.write(tx)
    assert (
        editor.snapshot.document.element("a").attributes == doc.element("a").attributes
    )
    assert editor.undo().document == doc


def test_check_reports_whether_an_operation_would_run_without_running_it():
    session = Session(import_svg(SVG))

    def check(method, **settings):
        return session.operation(
            {
                "command": "check",
                "action": "improve",
                "method": method,
                "epoch": session.epoch,
                "revision": session.editor.snapshot.revision,
                "permissions": {"geometry": True, "structure": True},
                "settings": settings,
            }
        )

    assert check("nodes", simplify=True) == {
        "ok": False,
        "error": "Select the paths to optimize",
    }
    session.editor.select(SELECTION)
    assert check("nodes", simplify=True) == {"ok": True}
    assert not session.jobs


def _ring(curves: int) -> str:
    import itertools
    import math

    points = [
        (
            32 + 20 * math.cos(2 * math.pi * i / curves),
            32 + 20 * math.sin(2 * math.pi * i / curves),
        )
        for i in range(curves + 1)
    ]
    d = f"M{points[0][0]:.3f} {points[0][1]:.3f}" + "".join(
        f" C{a[0]:.3f} {a[1]:.3f} {b[0]:.3f} {b[1]:.3f} {b[0]:.3f} {b[1]:.3f}"
        for a, b in itertools.pairwise(points)
    )
    return (
        '<svg width="64" height="64"><rect width="64" height="64" fill="#eeeeff"/>'
        f'<path id="a" d="{d} Z" fill="#800000"/></svg>'
    )


@pytest.mark.parametrize("curves", [16, 17, 40])
def test_closed_contours_of_any_length_move(curves):
    require_gpu()
    svg = _ring(curves)
    result = fit_selected_path(
        import_svg(svg),
        SELECTION,
        target(svg.replace('id="a"', 'id="a" transform="translate(3 2)"')),
        FitOptions(color=False, steps=20, resolution=64, displacement=4),
    )
    assert result.values
    assert result.after < result.before


@pytest.fixture
def cpu_only(monkeypatch):
    """Hide CUDA from the path fitter, as on a machine without a GPU."""
    pytest.importorskip("torch")
    monkeypatch.setattr(
        "vectrify.refine.selected.gpu_problem",
        lambda: "GPU fitting needs an NVIDIA GPU with CUDA",
    )


@pytest.mark.parametrize("curves", [4, 40])
@pytest.mark.usefixtures("cpu_only")
def test_cpu_fit_moves_an_offset_fill_toward_the_reference(curves):
    svg = _ring(curves)
    result = fit_selected_path(
        import_svg(svg),
        SELECTION,
        target(svg.replace('id="a"', 'id="a" transform="translate(3 2)"')),
        FitOptions(color=False, steps=20, resolution=64, displacement=4),
    )
    assert result.values
    assert result.after < result.before * 0.7


@pytest.mark.usefixtures("cpu_only")
def test_cpu_fit_refuses_outlined_shapes_clearly():
    outlined = SVG.replace(
        'fill="#800000"', 'fill="#800000" stroke="#800000" stroke-width="3"'
    )
    with pytest.raises(DocumentError, match="outlined shape needs an NVIDIA GPU"):
        validate_selection(import_svg(outlined), SELECTION, FitOptions())


@pytest.mark.usefixtures("cpu_only")
def test_path_fit_check_accepts_unstroked_fills_without_a_gpu():
    session = Session(import_svg(SVG), reference=reference(target()))
    session.editor.select(SELECTION)

    def check(checked):
        return checked.operation(
            {
                "command": "check",
                "action": "improve",
                "method": "path-fit",
                "epoch": checked.epoch,
                "revision": checked.editor.snapshot.revision,
                "permissions": {"geometry": True},
                "settings": {"nodes": True, "handles": True, "color": False},
            }
        )

    assert check(session) == {"ok": True}
    outlined = Session(
        import_svg(
            SVG.replace(
                'fill="#800000"', 'fill="#800000" stroke="#800000" stroke-width="3"'
            )
        ),
        reference=reference(target()),
    )
    outlined.editor.select(SELECTION)
    assert check(outlined) == {
        "ok": False,
        "error": "Fitting an outlined shape needs an NVIDIA GPU",
    }


@pytest.mark.parametrize("curves", [4, 40])
def test_cpu_coverage_matches_the_native_kernel(curves):
    require_gpu()
    import torch

    from vectrify.refine.cuda_renderer import multi_coverage
    from vectrify.refine.paths import _fused_chunks, parse_filled_cubics
    from vectrify.refine.soft_coverage import soft_coverage

    d = import_svg(_ring(curves)).geometry_for("a").path_data()
    contours = [torch.tensor(c, dtype=torch.float32) for c in parse_filled_cubics(d)]
    box = (0, 0, 64, 64)
    soft = soft_coverage(contours, box)
    packed = torch.cat([_fused_chunks(c.cuda()) for c in contours])
    native = multi_coverage(
        packed, [0, len(packed)], box, subpixels=2, fill_rule="nonzero"
    )
    assert native is not None
    difference = (soft - native[0].cpu()).abs()
    assert float(difference.mean()) < 0.01
    assert float(difference.max()) < 0.4
    assert abs(float(soft.sum() - native[0].sum())) < 5
