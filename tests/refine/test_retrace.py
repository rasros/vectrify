"""The retrace SAM cache: one model and embedding, reused, then released."""

from typing import ClassVar

import numpy as np
from PIL import Image

import vectrify.ui.session as session_module
from vectrify.document import import_svg
from vectrify.operations import OperationResult, Proposal, contract
from vectrify.refine.retrace import Encoding, Prompt, SamCache, _mask_logits
from vectrify.ui.session import Session


class Backend:
    def __init__(self):
        self.calls = []

    def load(self, model):
        self.calls.append(("load", model))
        return {"model": model}

    def encode(self, _runtime, image):
        self.calls.append(("encode", image.size))
        return Encoding(object(), image.size[::-1], image.size[::-1])

    def decode(self, _runtime, _encoding, _prompt):
        self.calls.append(("decode",))
        return [(np.ones((2, 2), dtype=bool), 1.0)] * 3

    def release(self, runtime):
        self.calls.append(("release", runtime["model"]))


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def cache(**options):
    backend, clock = Backend(), Clock()
    return SamCache(backend, idle=60.0, clock=clock, **options), backend, clock


def names(backend):
    return [call[0] for call in backend.calls]


def test_the_embedding_is_reused_until_the_reference_changes():
    sam, backend, _clock = cache()
    first = Image.new("RGB", (8, 4), "red")
    sam.masks(first, [Prompt(points=((1, 1),), labels=(1,))], model="sam")
    sam.masks(first.copy(), [Prompt(), Prompt()], model="sam")
    assert names(backend) == ["load", "encode", "decode", "decode", "decode"]
    sam.masks(Image.new("RGB", (8, 4), "blue"), [Prompt()], model="sam")
    assert names(backend)[-2:] == ["encode", "decode"]
    # Another model is loaded in place of the first.
    sam.masks(first, [Prompt()], model="other")
    assert backend.calls[-4:] == [
        ("release", "sam"),
        ("load", "other"),
        ("encode", (8, 4)),
        ("decode",),
    ]
    sam.release()


def test_an_idle_cache_releases_the_model_and_embedding():
    sam, backend, clock = cache()
    sam.masks(Image.new("RGB", (4, 4)), [Prompt()], model="sam")
    clock.now = 59.0
    assert not sam.expire()
    assert sam.loaded
    assert sam.encoded
    clock.now = 61.0
    assert sam.expire()
    assert not sam.loaded
    assert not sam.encoded
    assert backend.calls[-1] == ("release", "sam")
    # The next retrace loads and encodes again.
    sam.masks(Image.new("RGB", (4, 4)), [Prompt()], model="sam")
    assert names(backend)[-3:] == ["load", "encode", "decode"]
    sam.release()
    assert not sam.loaded


def test_the_idle_timer_releases_without_being_asked():
    backend = Backend()
    sam = SamCache(backend, idle=0.01)
    sam.masks(Image.new("RGB", (4, 4)), [Prompt()], model="sam")
    timer = sam._timer
    assert timer is not None
    timer.join(5)
    assert not sam.loaded
    assert backend.calls[-1] == ("release", "sam")


def test_the_path_becomes_a_mask_prompt_in_sams_padded_frame():
    mask = np.zeros((50, 100), dtype=bool)
    mask[:, :50] = True
    logits = _mask_logits(mask, (512, 1024))
    assert logits.shape == (256, 256)
    # The left half of the image's rows is inside; the padding is outside.
    assert logits[10, 10] > 0 > logits[10, 200]
    assert logits[200, 10] < 0


class Spy:
    def __init__(self):
        self.releases = 0

    def release(self):
        self.releases += 1


class GpuJob:
    action: ClassVar[str] = "generate"
    name: ClassVar[str] = "test-gpu"
    background: ClassVar[bool] = False
    needs_reference: ClassVar[bool] = False
    resources: ClassVar[frozenset[str]] = frozenset({"gpu"})

    def validate(self, request):
        pass

    def run(self, request, _context):
        return OperationResult(Proposal(request.transaction("Nothing"), False))


def test_the_session_releases_sam_for_a_new_reference_and_other_gpu_jobs(
    monkeypatch,
):
    spy = Spy()
    monkeypatch.setattr(session_module, "SAM_CACHE", spy)
    monkeypatch.setitem(contract._METHODS, ("generate", "test-gpu"), GpuJob())
    s = Session(import_svg('<svg width="10" height="10"/>'))

    def send(command, **data):
        return s.action(
            {
                "command": command,
                "epoch": s.epoch,
                "revision": s.editor.snapshot.revision,
                **data,
            }
        )

    url = (
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlE"
        "QVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    send("reference", reference={"name": "a.png", "data_url": url})
    assert spy.releases == 1
    # Only the opacity changed: the embedding still holds.
    send("reference", reference={"name": "a.png", "data_url": url, "opacity": 1})
    assert spy.releases == 1
    s.operation(
        {
            "command": "start",
            "action": "generate",
            "method": "test-gpu",
            "epoch": s.epoch,
            "revision": s.editor.snapshot.revision,
        }
    )
    assert spy.releases == 2
    send("reference", reference=None)
    assert spy.releases == 3
