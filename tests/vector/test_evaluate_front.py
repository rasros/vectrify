"""The evaluator's cache: what it recalls, what it re-measures, what it costs.

A panel call is the expensive thing in a run, and the pool it is asked about
changes slowly, so most of any call is a repeat of the last one. These pin that
the repeat is free -- and that the cheap paths stay cheap when the evaluator is
absent, broken, or has nothing new to look at.
"""

from vectrify.formats.models import VectorStatePayload
from vectrify.score.metrics import FRONT_SCORE
from vectrify.search import ChainState, SearchNode
from vectrify.vector.runner import evaluate_front


class FakePlugin:
    def rasterize(self, content, out_w, out_h):
        _ = (out_w, out_h)
        return content.encode()


class CountingScorer:
    """Records every field it is asked to score."""

    def __init__(self, values: list[float] | None = None):
        self.calls: list[int] = []
        self._values = values

    def rank(self, _ref, pngs):
        self.calls.append(len(pngs))
        if self._values is not None:
            return self._values[: len(pngs)]
        return [0.1 * (i + 1) for i in range(len(pngs))]


def _node(node_id: int, content: str = "<svg/>") -> SearchNode:
    return SearchNode(
        valid=True,
        id=node_id,
        parent_id=0,
        state=ChainState(
            payload=VectorStatePayload(
                content=content,
                raster_data_url=None,
                raster_preview_data_url=None,
                origin=None,
            ),
        ),
    )


def _evaluate(nodes, scorer, built: list | None = None):
    def front_scorer():
        if built is not None:
            built.append(1)
        return scorer, object()

    return evaluate_front(
        nodes,
        front_scorer=front_scorer,
        format_plugin=FakePlugin(),
        out_w=8,
        out_h=8,
    )


def test_every_node_is_scored_the_first_time():
    scorer = CountingScorer()
    nodes = [_node(i, f"<svg id='{i}'/>") for i in range(1, 4)]

    ranked = _evaluate(nodes, scorer)

    assert scorer.calls == [3]
    assert all(FRONT_SCORE in n.metrics for n in ranked)


def test_a_second_look_at_the_same_nodes_costs_nothing():
    scorer = CountingScorer()
    nodes = [_node(i, f"<svg id='{i}'/>") for i in range(1, 4)]

    _evaluate(nodes, scorer)
    _evaluate(nodes, scorer)

    assert scorer.calls == [3]


def test_a_fully_cached_call_does_not_even_build_the_scorer():
    """The model is the expensive part, and a call the cache answers in full
    has no reason to load one."""
    scorer = CountingScorer()
    nodes = [_node(i, f"<svg id='{i}'/>") for i in range(1, 4)]
    _evaluate(nodes, scorer)

    built: list = []
    _evaluate(nodes, scorer, built=built)

    assert built == []


def test_only_the_unseen_nodes_are_scored():
    scorer = CountingScorer()
    old = [_node(i, f"<svg id='{i}'/>") for i in range(1, 4)]
    _evaluate(old, scorer)

    fresh = [_node(9, "<svg id='9'/>")]
    _evaluate([*old, *fresh], scorer)

    assert scorer.calls == [3, 1]


def test_a_recalled_score_orders_against_a_fresh_one():
    """The point of an absolute score. A value measured in an earlier call has
    to be comparable with one measured now, or the cache would order the field
    by when each candidate happened to be seen."""
    scorer = CountingScorer(values=[0.9])
    stale = _node(1, "<svg id='1'/>")
    _evaluate([stale], scorer)

    better = _node(2, "<svg id='2'/>")
    scorer._values = [0.1]
    ranked = _evaluate([stale, better], scorer)

    assert [n.id for n in ranked] == [2, 1]


def test_nodes_without_content_are_left_out_rather_than_failing_the_field():
    scorer = CountingScorer()
    nodes = [_node(1, "<svg id='1'/>"), _node(2, "")]

    ranked = _evaluate(nodes, scorer)

    assert scorer.calls == [1]
    assert [n.id for n in ranked] == [1]


def test_a_failing_evaluator_returns_the_nodes_it_was_given():
    class Exploding:
        def rank(self, _ref, _pngs):
            raise RuntimeError("no")

    nodes = [_node(1, "<svg id='1'/>"), _node(2, "<svg id='2'/>")]

    ranked = _evaluate(nodes, Exploding())

    assert ranked == nodes
    assert all(FRONT_SCORE not in n.metrics for n in ranked)


def test_a_scorer_without_rank_is_asked_one_candidate_at_a_time():
    """--scorer simple has no panel to put a field to, only a score per
    candidate."""

    class SingleOnly:
        def __init__(self):
            self.seen = 0

        def score(self, _ref, _png):
            self.seen += 1
            return 0.5

    scorer = SingleOnly()
    nodes = [_node(i, f"<svg id='{i}'/>") for i in range(1, 4)]

    _evaluate(nodes, scorer)

    assert scorer.seen == 3
