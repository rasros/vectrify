"""Retrace shape replaces a path's outline with its object's, as one edit."""

import base64
import io
import time

import numpy as np
import pytest
from PIL import Image

import vectrify.refine.retrace as retrace
from vectrify.document import DocumentError, Selection, import_svg
from vectrify.ui.session import Session

# 100x50 artboard, 200x100 reference: two pixels a unit. The group maps the
# path's own coordinates to the artboard's, so the trace must undo it.
DOC = """<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"
viewBox="0 0 100 50">
<rect id="bg" width="100" height="50" fill="#ffffff"/>
<g id="layer" transform="translate(10 5) scale(2)">
<path id="below" d="M0 0H4V4H0Z" fill="#00ff00"/>
<path id="shape" d="M12 5H26V15H12Z" fill="#c80000" opacity="0.9"/>
<path id="above" d="M40 0H44V4H40Z" fill="#0000ff"/>
</g></svg>"""


def object_mask() -> np.ndarray:
    """The red object in the reference's pixels, with a square hole."""
    mask = np.zeros((100, 200), dtype=bool)
    mask[20:80, 60:140] = True
    mask[60:72, 125:135] = False
    return mask


def reference_image() -> Image.Image:
    pixels = np.full((100, 200, 3), 255, dtype=np.uint8)
    pixels[object_mask()] = (200, 0, 0)
    return Image.fromarray(pixels)


def data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class FakeSam:
    """A SAM backend that always answers with one known mask."""

    def __init__(self, mask):
        self.mask = mask
        self.loads = self.encodes = self.decodes = self.releases = 0
        self.prompts = []

    def load(self, model):
        self.loads += 1
        return {"model": model}

    def encode(self, _runtime, image):
        self.encodes += 1
        return retrace.Encoding(object(), image.size[::-1], image.size[::-1])

    def decode(self, _runtime, _encoding, prompt):
        self.decodes += 1
        self.prompts.append(prompt)
        # A wrong candidate too: the choice must not take it.
        wrong = np.zeros_like(self.mask)
        wrong[:, :30] = True
        mask = self.mask
        left, top, right, bottom = (round(v) for v in prompt.box or (0, 0, 0, 0))
        if not mask[top:bottom, left:right].any():
            # Elsewhere, the object is a square around the box.
            mask = np.zeros_like(self.mask)
            mask[max(0, top - 2) : bottom + 2, max(0, left - 2) : right + 2] = True
        return [(mask, 0.9), (wrong, 0.99), (mask, 0.8)]

    def release(self, _runtime):
        self.releases += 1


@pytest.fixture
def fake_sam(monkeypatch):
    backend = FakeSam(object_mask())
    monkeypatch.setattr(retrace, "SAM_CACHE", retrace.SamCache(backend, idle=None))
    monkeypatch.setattr(retrace, "sam_problem", lambda: None)
    return backend


def session() -> Session:
    return Session(
        import_svg(DOC),
        reference=Session.validate_reference(
            {"name": "ref.png", "data_url": data_url(reference_image())}
        ),
    )


def send(s, command, **data):
    return s.action(
        {
            "command": command,
            "epoch": s.epoch,
            "revision": s.editor.snapshot.revision,
            **data,
        }
    )


def retrace_job(s, **settings):
    job = s.operation(
        {
            "command": "start",
            "action": "improve",
            "method": "retrace",
            "epoch": s.epoch,
            "revision": s.editor.snapshot.revision,
            "permissions": {"geometry": True, "structure": True},
            "settings": settings,
        }
    )
    deadline = time.monotonic() + 60
    while job["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.02)
        job = s.operation({"command": "status", "job": job["id"]})
    return job


def coverage(document, oid="shape") -> np.ndarray:
    item = retrace.target(document, oid, reference_image(), "nonzero")
    assert item is not None
    return item.coverage


def iou(a, b) -> float:
    return np.count_nonzero(a & b) / np.count_nonzero(a | b)


def test_sam_retrace_gives_the_mask_outline_in_the_paths_own_frame(fake_sam):
    s = session()
    send(s, "select", objects=["shape"])
    before = s.editor.snapshot.document
    job = retrace_job(s)
    assert job["status"] == "ready", job
    assert job["result"]["changed"]
    metrics = job["result"]["metrics"]
    assert metrics["mode"] == "sam"
    assert metrics["paths"] == 1
    assert metrics["after"]["error"] < metrics["before"]["error"]
    s.operation({"command": "apply", "job": job["id"]})
    after = s.editor.snapshot.document
    assert iou(coverage(after), object_mask()) > 0.97
    # The outline is in the path's own coordinates: the group still maps it.
    xs = [
        n.endpoint[0] for sp in after.geometry_for("shape").subpaths for n in sp.nodes
    ]
    assert min(xs) == pytest.approx(10, abs=0.3)
    assert max(xs) == pytest.approx(30, abs=0.3)
    # The hole stays a hole.
    assert len(after.geometry_for("shape").subpaths) == 2
    assert not coverage(after)[66, 130]
    # Identity, paint and stacking are kept; only the geometry changed.
    element = after.element("shape")
    assert element == before.element("shape")
    assert [c.id for c in after.element("layer").children] == [
        "below",
        "shape",
        "above",
    ]
    assert s.state()["undo"] == ["Retrace shape"]
    send(s, "undo")
    assert s.editor.snapshot.document == before
    # The prompts were the path's box and points inside it.
    prompt = fake_sam.prompts[0]
    assert prompt.box == (68.0, 30.0, 124.0, 70.0)
    assert 1 in prompt.labels


def test_repeat_retraces_reuse_the_embedding(fake_sam):
    s = session()
    send(s, "select", objects=["shape"])
    for _ in range(2):
        job = retrace_job(s)
        assert job["status"] == "ready", job
    assert (fake_sam.loads, fake_sam.encodes) == (1, 1)
    assert fake_sam.decodes > 1


def test_colour_retrace_recovers_a_flat_colour_region():
    s = session()
    send(s, "select", objects=["shape"])
    job = retrace_job(s, mode="colour")
    assert job["status"] == "ready", job
    assert job["result"]["metrics"]["mode"] == "colour"
    s.operation({"command": "apply", "job": job["id"]})
    assert iou(coverage(s.editor.snapshot.document), object_mask()) > 0.97


def test_colour_mode_is_used_when_sam_cannot_run(monkeypatch):
    monkeypatch.setattr(retrace, "sam_problem", lambda: "no GPU")
    s = session()
    send(s, "select", objects=["shape"])
    job = retrace_job(s)
    assert job["status"] == "ready", job
    assert job["result"]["metrics"]["mode"] == "colour"
    assert job["result"]["metrics"]["paths"] == 1


@pytest.mark.usefixtures("fake_sam")
def test_multiselection_retraces_each_path():
    s = session()
    send(s, "select", objects=["shape", "above"])
    job = retrace_job(s)
    assert job["status"] == "ready", job
    assert job["result"]["metrics"]["paths"] == 2
    s.operation({"command": "apply", "job": job["id"]})
    assert s.state()["undo"] == ["Retrace shape"]


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        (lambda s: send(s, "select", objects=["bg"]), "only visible paths"),
        (lambda s: send(s, "select", objects=["layer"]), "only visible paths"),
        (
            lambda s: (
                send(s, "select", objects=["shape"]),
                send(s, "locks", object="shape", locks=["geometry"]),
            ),
            "locked",
        ),
        (
            lambda s: (
                send(s, "select", objects=["shape"]),
                send(
                    s,
                    "pin",
                    object="shape",
                    node=s.editor.snapshot.document.geometry_for("shape")
                    .subpaths[0]
                    .nodes[0]
                    .id,
                    pinned=True,
                ),
            ),
            "unpin",
        ),
        (lambda _s: None, "Select the paths"),
    ],
)
def test_retrace_refuses_what_it_cannot_replace(setup, message):
    s = session()
    setup(s)
    with pytest.raises(DocumentError, match=message):
        retrace_job(s)


def test_retrace_refuses_shared_geometry():
    s = session()
    with s.editor.transaction("Share", selection=Selection.all()) as tx:
        tx.share_geometry("below", "shape")
    send(s, "select", objects=["shape"])
    with pytest.raises(DocumentError, match="shared"):
        retrace_job(s)


def test_retrace_needs_a_reference():
    s = Session(import_svg(DOC))
    send(s, "select", objects=["shape"])
    with pytest.raises(DocumentError, match="reference"):
        retrace_job(s)
