"""Admission and publication stay independent across the search stages."""

import pytest

from vectrify.document import Editor, Selection, export_svg, import_svg
from vectrify.refine.cel_plan import search as beam
from vectrify.refine.cel_plan.frontier import Frontier
from vectrify.refine.cel_plan.model import Options, Work
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.proposals import bounds
from vectrify.refine.cel_plan.score import render


@pytest.fixture
def native_seed():
    document = import_svg(
        '<svg width="48" height="48"><defs><linearGradient id="paint">'
        '<stop offset="0" stop-color="#ac5030"/>'
        '<stop offset="1" stop-color="#b45030"/></linearGradient></defs>'
        '<path id="body" fill="url(#paint)" d="M4 4H44V44H4Z"/></svg>'
    )
    svg = export_svg(document)
    target = svg.replace('fill="url(#paint)"', 'fill="#b05030"')
    frontier = Frontier(Policy(render(target, (48, 48))))
    assert frontier.add(svg, "Native seed")
    frontier.freeze_normalizer()
    return frontier, document


def test_native_seed_admission_keeps_the_supplied_document(native_seed, monkeypatch):
    frontier, document = native_seed
    seen = []

    def unexpected_import(_svg):
        pytest.fail("A supplied native document must keep its editor identities")

    def proposals(state, _work):
        seen.append(state.document)
        return iter(())

    monkeypatch.setattr(beam, "import_svg", unexpected_import)
    report = beam.search(
        frontier,
        Options(refine=False),
        Work.start(30),
        proposals,
        seed_document=document,
    )
    assert seen == [document]
    assert seen[0] is document
    assert report["attempted"] == report["checkpointed"] == 0


def test_mismatched_native_seed_fails_before_canvas_or_discovery(
    native_seed, monkeypatch
):
    frontier, _document = native_seed
    other = import_svg('<svg width="48" height="48"/>')

    def unexpected_start(*_args):
        pytest.fail("Mismatched native seeds must fail before allocating a canvas")

    def unexpected_proposals(*_args):
        pytest.fail("Mismatched native seeds must fail before discovery")

    monkeypatch.setattr(beam.LocalPolicy, "start", unexpected_start)
    with pytest.raises(ValueError, match="Native search seed does not match"):
        beam.search(
            frontier,
            Options(refine=False),
            Work.start(30),
            unexpected_proposals,
            seed_document=other,
        )


@pytest.mark.parametrize("cancel_before_publish", [False, True])
def test_expansion_requires_an_independent_live_publication_stage(
    native_seed, cancel_before_publish
):
    frontier, document = native_seed
    baseline = frontier.baseline
    options, work = Options(quality="fast", refine=False), Work.start(30)

    def proposals(state, _work):
        if state.edits:
            return
        editor = Editor(state.document, selection=Selection(whole_document=True))
        with editor.transaction("Flat paint proposal") as transaction:
            transaction.set_fill("body", "#b05030")
        changed = editor.snapshot.document
        yield beam.Proposal(
            "paint",
            ("body",),
            ("#b05030",),
            state.key,
            changed,
            bounds(state.document, changed, ("body",)),
        )

    seed = beam._search_seed(frontier, options, work, proposals, document)
    assert seed is not None
    width, limit = beam.LIMITS[options.quality]
    run = beam._SearchRun(
        frontier=frontier,
        options=options,
        work=work,
        proposals=proposals,
        started=0,
        initial=seed.state,
        evaluator=seed.evaluator,
        beam=beam._Beam([seed.state], seed.retained_bytes),
        budget=beam._CheckpointBudget.allocate(work, None, 0),
        width=width,
        limit=limit,
    )
    run.expand()
    assert run.counts.accepted == 1
    assert frontier.baseline is baseline
    assert len(frontier.entries) == 1
    assert frontier.select(50).metrics["gradients"] == 1

    if cancel_before_publish:
        work.stop.set()
    checkpoints = run.publish_checkpoints()
    assert checkpoints.count == (0 if cancel_before_publish else 1)
    assert checkpoints.disagreements == 0
    assert frontier.select(50).metrics["gradients"] == (
        1 if cancel_before_publish else 0
    )
    assert frontier.baseline is baseline
