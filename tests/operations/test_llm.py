"""LLM Generate and Improve go through the same scoped, replayed edits."""

import pytest
from PIL import Image

from vectrify.document import (
    DocumentError,
    Editor,
    Selection,
    export_svg,
    import_svg,
)
from vectrify.operations import Job, OperationRequest, Permissions, method
from vectrify.operations.methods.llm import to_canvas

DOC = (
    '<svg width="200" height="100" viewBox="0 0 200 100">'
    '<rect id="bg" width="200" height="100" fill="#ffffff"/>'
    '<rect id="a" x="20" y="20" width="40" height="40" fill="#330000"/>'
    '<rect id="b" x="120" y="20" width="40" height="40" fill="#000033"/></svg>'
)


def reference():
    image = Image.new("RGB", (400, 200), "white")
    image.paste((200, 0, 0), (40, 40, 120, 120))
    image.paste((0, 0, 200), (240, 40, 320, 120))
    return image


class Fake:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt, config):
        self.prompts.append((prompt, config))
        return self.replies.pop(0)


@pytest.fixture
def fake(monkeypatch):
    from vectrify.llm import keys

    keys.save({"openai": "test"})
    holder = {}

    def install(*replies):
        holder["client"] = Fake(replies)
        monkeypatch.setattr(
            "vectrify.llm.get_provider", lambda *_a, **_k: holder["client"]
        )
        return holder["client"]

    return install


def request(editor, action, selection, permissions, **settings):
    snapshot = editor.snapshot
    return OperationRequest(
        action=action,
        method="llm",
        snapshot=type(snapshot)(snapshot.revision, snapshot.document, selection),
        editor=editor,
        permissions=permissions,
        settings={"provider": "openai", "resolution": 128, **settings},
        reference=reference(),
    )


def run(req):
    job = Job(method(req.action, "llm"), req)
    job.run()
    return job


def test_generate_places_the_drawing_over_the_region(fake):
    client = fake(
        "Here it is:\n```svg\n<svg xmlns='http://www.w3.org/2000/svg' "
        "viewBox='0 0 400 200'><rect x='40' y='40' width='80' height='80' "
        "fill='#c80000'/></svg>\n```"
    )
    ed = Editor(import_svg(DOC))
    job = run(request(ed, "generate", Selection.all(), Permissions(structure=True)))
    state = job.state()
    assert state["status"] == "ready", state
    assert state["result"]["metrics"]["shapes"] == 1
    prompt, config = client.prompts[0]
    assert config.model
    assert "viewBox='0 0 400 200'" in prompt[0]["text"]
    job.apply()
    group = ed.snapshot.document.root.children[-1]
    assert group.get("transform") == "matrix(0.5 0 0 0.5 0.0 0.0)"


def test_generate_rescales_a_reply_with_its_own_viewbox():
    svg = to_canvas(
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 50'>"
        "<rect width='10' height='10'/></svg>",
        (400, 200),
    )
    assert "matrix(4.0 0 0 4.0" in svg
    assert 'viewBox="0 0 400 200"' in svg


def test_unusable_replies_fail_with_the_reason(fake):
    fake("<svg xmlns='http://www.w3.org/2000/svg'><text>hi</text></svg>")
    ed = Editor(import_svg(DOC))
    job = run(request(ed, "generate", Selection.all(), Permissions(structure=True)))
    assert job.state()["status"] == "failed"
    assert "could not be used" in job.state()["error"]


def edited(**fills):
    svg = export_svg(import_svg(DOC))
    for oid, fill in fills.items():
        old = {"a": "#330000", "b": "#000033"}[oid]
        svg = svg.replace(old, fill)
    return svg


def test_edit_changes_only_the_selection_and_counts_what_it_skipped(fake):
    extra = '<circle cx="100" cy="80" r="5" fill="#00ff00"/></svg>'
    client = fake(edited(a="#c80000", b="#0000c8").replace("</svg>", extra))
    ed = Editor(import_svg(DOC))
    job = run(
        request(
            ed,
            "improve",
            Selection(object_ids=frozenset({"a"})),
            Permissions(paint=True, structure=True),
            instruction="Match the reference colours",
        )
    )
    state = job.state()
    assert state["status"] == "ready", state
    assert state["result"]["metrics"]["skipped"] == 2
    prompt = client.prompts[0][0][0]["text"]
    assert "Match the reference colours" in prompt
    assert "Only change these elements (by id)" in prompt
    job.apply()
    document = ed.snapshot.document
    assert document.element("a").get("fill") == "#c80000"
    assert document.element("b").get("fill") == "#000033"
    assert len(document.root.children) == 3


def test_whole_drawing_edit_may_add_shapes_with_structure_permission(fake):
    extra = '<circle cx="100" cy="80" r="5" fill="#00ff00"/></svg>'
    fake(edited(b="#0000c8").replace("</svg>", extra))
    ed = Editor(import_svg(DOC))
    job = run(
        request(
            ed,
            "improve",
            Selection.all(),
            Permissions(paint=True, structure=True),
            instruction="Add a dot",
        )
    )
    assert job.state()["status"] == "ready", job.state()
    job.apply()
    document = ed.snapshot.document
    assert [c.tag for c in document.root.children][-1] == "circle"
    assert document.element("b").get("fill") == "#0000c8"


def test_edit_needs_an_instruction():
    ed = Editor(import_svg(DOC))
    with pytest.raises(DocumentError, match="Describe"):
        method("improve", "llm").validate(
            request(ed, "improve", Selection.all(), Permissions(paint=True))
        )


def test_edit_keeps_full_precision_where_the_model_changed_nothing(fake):
    doc = (
        '<svg width="200" height="100"><path id="p" fill="#330000" '
        'd="M10.123456 10.654321 L50.987654 10.111111 L30.5 40.25 Z"/></svg>'
    )
    exported = export_svg(import_svg(doc))
    fake(exported.replace("#330000", "#c80000"))
    ed = Editor(import_svg(doc))
    before = ed.snapshot.document.geometry_for("p")
    job = run(
        request(
            ed,
            "improve",
            Selection(object_ids=frozenset({"p"})),
            Permissions(paint=True, geometry=True),
            instruction="Fix the colour",
        )
    )
    assert job.state()["result"]["metrics"]["edits"] == 1, job.state()
    job.apply()
    assert ed.snapshot.document.geometry_for("p") == before
    assert ed.snapshot.document.element("p").get("fill") == "#c80000"


def test_the_file_name_is_offered_as_a_subject_hint(fake):
    client = fake(
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 200'>"
        "<rect width='10' height='10'/></svg>"
    )
    ed = Editor(import_svg(DOC))
    snapshot = ed.snapshot
    req = OperationRequest(
        action="generate",
        method="llm",
        snapshot=type(snapshot)(snapshot.revision, snapshot.document, Selection.all()),
        editor=ed,
        permissions=Permissions(structure=True),
        settings={"provider": "openai", "resolution": 128},
        reference=reference(),
        source_name="little-duck.png",
    )
    run(req)
    assert "`little-duck.png`" in client.prompts[0][0][0]["text"]


def test_the_session_prefers_the_reference_name_and_skips_placeholders():
    from vectrify.ui.session import Session

    session = Session(import_svg(DOC), name="Untitled.svg")
    assert session.source_name() is None
    session.name = "logo.svg"
    assert session.source_name() == "logo.svg"
    session.reference = {"name": "duck.png", "data_url": "", "opacity": 0.5}
    assert session.source_name() == "duck.png"
