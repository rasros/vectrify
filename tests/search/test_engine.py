import time

import pytest

from vectrify.score.metrics import FRONT_SCORE
from vectrify.search import ChainState, Result, SearchNode
from vectrify.search.engine import MultiprocessSearchEngine


def _rank_of(node) -> float:
    """These fakes order on one measure, where the real strategy trades four
    off by dominance. A test that cares about order sets `edge` on its results.

    Without one the id breaks the tie, so a pool still has a strict order and a
    later candidate is the worse one.
    """
    return node.metrics.get("edge", float(node.id))


class _TierMixin:
    """The best-ranked tier the engine asks every strategy for."""

    def top_tier_ids(self, pool) -> set[int]:
        valid = [n for n in pool if n.valid]
        if not valid:
            return set()
        best = min(_rank_of(n) for n in valid)
        return {n.id for n in valid if _rank_of(n) == best}


class FakeStrategy(_TierMixin):
    def select_parent(
        self,
        nodes: list[SearchNode],
    ) -> tuple[int, int | None]:
        _ = nodes
        return 1, None

    def select_survivors(
        self, nodes: list[SearchNode], max_keep: int
    ) -> list[SearchNode]:
        return sorted(nodes, key=_rank_of)[:max_keep]

    def epoch_parents(
        self, pool: list[SearchNode], max_parents: int
    ) -> list[SearchNode]:
        return pool[:max_parents]


def test_engine_init():
    engine = MultiprocessSearchEngine(2, FakeStrategy())
    assert engine.workers == 2


def test_engine_run_loop_processes_result_and_saves():
    strat = FakeStrategy()

    engine = MultiprocessSearchEngine(workers=1, strategy=strat, max_total_tasks=1)

    res = Result(
        task_id=1,
        parent_id=1,
        valid=True,
        measured=True,
        payload="fake_payload",
    )
    # Put into the unscored queue so the ScorerThread can process it
    engine.unscored_q.put(res)

    initial_node = SearchNode(
        valid=True,
        id=1,
        parent_id=0,
        state=ChainState(payload=None),
    )

    outcome = engine.run(
        initial_nodes=[initial_node],
        max_wall_seconds=None,
    )

    assert any(n.id != 1 for n in outcome.pool)


def test_engine_respects_max_wall_seconds(monkeypatch):
    class TrackingStrategy(FakeStrategy):
        def __init__(self):
            self.select_calls = 0

        def select_parent(self, nodes: list[SearchNode]) -> tuple[int, int | None]:
            _ = nodes
            self.select_calls += 1
            return 1, None

    strat = TrackingStrategy()
    engine = MultiprocessSearchEngine(workers=1, strategy=strat)

    class FakeTime:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            return float(self.calls * 100)

    monkeypatch.setattr(time, "monotonic", FakeTime())

    dummy_node = SearchNode(
        valid=True,
        id=1,
        parent_id=0,
        state=ChainState(payload=None),
    )

    # The first wall-clock check already exceeds the limit, so the run loop
    # must exit before dispatching any task.
    engine.run(initial_nodes=[dummy_node], max_wall_seconds=50.0)
    assert strat.select_calls == 0


def test_engine_respects_max_total_tasks():
    class TrackingStrategy(FakeStrategy):
        def __init__(self):
            self.select_calls = 0

        def select_parent(self, nodes: list[SearchNode]) -> tuple[int, int | None]:
            _ = nodes
            self.select_calls += 1
            return 1, None

    strat = TrackingStrategy()
    engine = MultiprocessSearchEngine(workers=1, strategy=strat, max_total_tasks=0)
    dummy_node = SearchNode(
        valid=True,
        id=1,
        parent_id=0,
        state=ChainState(payload=None),
    )

    outcome = engine.run(initial_nodes=[dummy_node], max_wall_seconds=None)
    assert strat.select_calls == 0
    assert [n.id for n in outcome.pool] == [1]


def test_engine_active_pool_bounded():
    class TrackingStrategy(FakeStrategy):
        def __init__(self):
            self.max_seen = 0

        def select_parent(self, nodes: list[SearchNode]) -> tuple[int, int | None]:
            self.max_seen = max(self.max_seen, len(nodes))
            return nodes[0].id, None

    strat = TrackingStrategy()
    engine = MultiprocessSearchEngine(workers=1, strategy=strat, max_total_tasks=10)

    for i in range(10):
        engine.unscored_q.put(
            Result(
                task_id=i + 1,
                parent_id=1,
                valid=True,
                measured=True,
                payload="p",
            )
        )

    initial_node = SearchNode(
        valid=True, id=1, parent_id=0, state=ChainState(payload=None)
    )
    engine.run(
        initial_nodes=[initial_node],
        max_wall_seconds=None,
        active_pool_size=3,
    )
    assert strat.max_seen <= 4


def test_engine_score_fn_none_with_unscored_result_raises():
    engine = MultiprocessSearchEngine(workers=1, strategy=FakeStrategy())
    engine.unscored_q.put(
        Result(
            task_id=1,
            parent_id=1,
            valid=True,
            measured=False,
            payload="p",
        )
    )

    initial_node = SearchNode(
        valid=True, id=1, parent_id=0, state=ChainState(payload=None)
    )

    with pytest.raises(RuntimeError, match="never measured"):
        engine.run(
            initial_nodes=[initial_node],
            max_wall_seconds=None,
            score_fn=None,
        )


def test_front_is_ranked_by_the_evaluator_not_by_the_round_score():
    """The round optimises pixel L1 because it is ~300x cheaper than the real
    objective. If the evaluator's verdict did not decide the final pick, the
    cheap proxy's opinion would choose the artifact."""

    class TrackingStrategy(FakeStrategy):
        def select_parent(self, nodes):
            return nodes[0].id, None

        def epoch_parents(self, pool, max_parents):
            return pool[:max_parents]

    seen: list[list[int]] = []

    def rank_front(nodes):
        seen.append([n.id for n in nodes])
        return sorted(nodes, key=lambda n: -_rank_of(n))  # deliberately reversed

    strat = TrackingStrategy()
    engine = MultiprocessSearchEngine(
        workers=1,
        strategy=strat,
        max_total_tasks=4,
        rank_front=rank_front,
    )
    for tid in (1, 2, 3, 4):
        engine.unscored_q.put(
            Result(task_id=tid, parent_id=1, valid=True, measured=True, payload="p")
        )

    initial = SearchNode(valid=True, id=1, parent_id=0, state=ChainState(payload=None))
    engine.run(
        initial_nodes=[initial],
        max_wall_seconds=None,
        active_pool_size=2,
    )
    assert seen, "the evaluator was never consulted"


def _explode(_nodes):
    raise RuntimeError("no cuda")


def test_a_failing_evaluator_does_not_stop_the_run():
    """It runs a model at the end of the run; a load failure there must cost
    the ordering, not the run."""
    engine = MultiprocessSearchEngine(
        workers=1,
        strategy=FakeStrategy(),
        max_total_tasks=3,
        rank_front=_explode,
    )
    for tid in (1, 2, 3):
        engine.unscored_q.put(
            Result(task_id=tid, parent_id=1, valid=True, measured=True, payload="p")
        )
    initial = SearchNode(valid=True, id=1, parent_id=0, state=ChainState(payload=None))
    engine.run(
        initial_nodes=[initial],
        max_wall_seconds=None,
        active_pool_size=2,
    )


def test_children_join_the_pool_only_when_the_generation_closes():
    """Survival is a sort over the whole population, so it is paid once per
    generation rather than once per child; until then children are held back."""

    class TrackingStrategy(FakeStrategy):
        def __init__(self):
            self.pools = []

        def select_parent(self, nodes):
            self.pools.append({n.id for n in nodes})
            return nodes[0].id, None

    strat = TrackingStrategy()
    engine = MultiprocessSearchEngine(workers=1, strategy=strat, max_total_tasks=5)
    for tid in range(1, 6):
        engine.unscored_q.put(
            Result(task_id=tid, parent_id=1, valid=True, measured=True, payload="p")
        )

    initial = SearchNode(valid=True, id=1, parent_id=0, state=ChainState(payload=None))
    engine.run(
        initial_nodes=[initial],
        max_wall_seconds=None,
        active_pool_size=4,
        generation_size=4,
    )

    assert strat.pools[:4] == [{1}] * 4
    assert len(strat.pools[4]) > 1


def test_the_engine_picks_the_operator_and_hears_how_it_did():
    """Eight workers each running their own policy could never learn anything:
    none of them sees whether its own children survived."""

    class RecordingPolicy:
        def __init__(self):
            self.selects = 0
            self.updates = []

        def select(self):
            self.selects += 1
            return "op"

        def update(self, operator, reward):
            self.updates.append((operator, reward))

    policy = RecordingPolicy()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=FakeStrategy(), max_total_tasks=2
    )
    for tid, edge in ((1, 0.1), (2, 0.9)):
        engine.unscored_q.put(
            Result(
                task_id=tid,
                parent_id=1,
                valid=True,
                measured=True,
                payload="p",
                metrics={"edge": edge},
                operator="op",
            )
        )

    # The parent has to carry a measure of its own: the reward is how far the
    # child improved on it, so a parent that was never measured leaves nothing
    # to grade against.
    initial = SearchNode(
        valid=True,
        id=1,
        parent_id=0,
        state=ChainState(payload=None),
        metrics={"edge": 0.5},
    )
    engine.run(
        initial_nodes=[initial],
        max_wall_seconds=None,
        active_pool_size=1,
        generation_size=2,
        operator_policy=policy,
    )

    assert policy.selects == 2
    assert [engine.task_q.get().operator for _ in range(2)] == ["op", "op"]
    # The pool holds one node, so the better child survives and the worse does
    # not. The survivor's reward is graded rather than a flat 1: what it earns
    # is how far it improved on its parent, measured against the run's scale.
    assert [name for name, _ in policy.updates] == ["op", "op"]
    survivor, evicted = (reward for _, reward in policy.updates)
    assert survivor > 0.0
    assert evicted == 0.0


def test_an_operator_that_produced_nothing_is_charged_for_the_draw():
    """A blank draw spends a task and returns no candidate. Reporting nothing
    would leave the operator at its prior weight, still drawing its share to
    fail again; a zero is what the draw was actually worth."""

    class RecordingPolicy:
        def __init__(self):
            self.updates = []

        def select(self):
            return "op"

        def update(self, operator, reward):
            self.updates.append((operator, reward))

    policy = RecordingPolicy()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=FakeStrategy(), max_total_tasks=2
    )
    # One blank draw naming its operator, one ordinary failure naming none.
    engine.unscored_q.put(
        Result(
            task_id=1,
            parent_id=1,
            valid=False,
            measured=True,
            payload=None,
            operator="op",
        )
    )
    engine.unscored_q.put(
        Result(task_id=2, parent_id=1, valid=False, measured=True, payload=None)
    )

    engine.run(
        initial_nodes=[
            SearchNode(valid=True, id=1, parent_id=0, state=ChainState(payload=None))
        ],
        max_wall_seconds=None,
        active_pool_size=1,
        generation_size=2,
        operator_policy=policy,
    )

    assert policy.updates == [("op", 0.0)]


def _engine_with_evaluator(rank_front):
    return MultiprocessSearchEngine(
        workers=1,
        strategy=FakeStrategy(),
        max_total_tasks=2,
        rank_front=rank_front,
    )


def _run_two_children(engine):
    for tid in (1, 2):
        engine.unscored_q.put(
            Result(task_id=tid, parent_id=1, valid=True, measured=True, payload="p")
        )
    return engine.run(
        initial_nodes=[
            SearchNode(valid=True, id=1, parent_id=0, state=ChainState(payload=None))
        ],
        max_wall_seconds=None,
        active_pool_size=3,
        generation_size=1,
    )


def test_the_final_artifact_is_chosen_by_the_evaluator():
    """The round score is a proxy that ranks candidates at about rho 0.83, and
    the evaluator finds roughly a 2x spread inside one front -- so writing out
    the proxy's winner is close to picking arbitrarily among the good ones."""
    # An evaluator that prefers the worst round score, which the proxy never would.
    engine = _engine_with_evaluator(
        lambda nodes: sorted(nodes, key=lambda n: -_rank_of(n))
    )
    outcome = _run_two_children(engine)

    assert outcome.best is not None
    assert outcome.best.id == 3


def test_a_failing_evaluator_still_writes_a_top_tier_candidate():
    """Losing the run's single most important artifact to a scorer error at
    shutdown would be the worst possible time for it. With no blended score to
    fall back on, any unbeaten candidate is written instead."""

    def explode(_nodes):
        raise RuntimeError("no")

    outcome = _run_two_children(_engine_with_evaluator(explode))

    assert outcome.best is not None
    assert outcome.best.valid


def test_scorer_thread_scores_queued_results_together():
    """The point of batching: a scorer whose cost is per-call, not per-image,
    should see everything already queued in one call rather than one each."""

    class PoolStrategy(FakeStrategy):
        def select_parent(self, nodes: list[SearchNode]) -> tuple[int, int | None]:
            return nodes[0].id, None

    engine = MultiprocessSearchEngine(
        workers=1, strategy=PoolStrategy(), max_total_tasks=8
    )
    for task_id in range(1, 9):
        engine.unscored_q.put(
            Result(
                task_id=task_id, parent_id=1, valid=True, measured=False, payload="p"
            )
        )

    batch_sizes = []

    def score_fn(results):
        batch_sizes.append(len(results))
        for res in results:
            res.measured = True

    initial_node = SearchNode(
        valid=True, id=1, parent_id=0, state=ChainState(payload=None)
    )
    engine.run(
        initial_nodes=[initial_node],
        max_wall_seconds=None,
        score_fn=score_fn,
        active_pool_size=3,
    )

    assert sum(batch_sizes) == 8
    assert max(batch_sizes) > 1


def test_a_scoring_failure_loses_only_the_batch_it_belongs_to():
    """Scoring in batches must not let one bad candidate discard the candidates
    scored alongside it, which singly-scored candidates never risked."""
    engine = MultiprocessSearchEngine(
        workers=1, strategy=FakeStrategy(), max_total_tasks=4
    )
    for task_id in range(1, 5):
        engine.unscored_q.put(
            Result(
                task_id=task_id, parent_id=1, valid=True, measured=False, payload="p"
            )
        )

    scored = []

    def score_fn(results):
        # Half the batch is scored before the failure, as a real scorer that
        # fails partway through its work would leave it.
        for res in results[: len(results) // 2]:
            res.measured = True
            scored.append(res.task_id)
        raise ValueError("scorer exploded")

    initial_node = SearchNode(
        valid=True, id=1, parent_id=0, state=ChainState(payload=None)
    )
    engine.run(
        initial_nodes=[initial_node],
        max_wall_seconds=None,
        score_fn=score_fn,
        active_pool_size=3,
    )

    assert scored, "results scored before the failure should have survived it"


def test_the_evaluator_is_asked_during_the_run_not_only_at_its_end():
    """The cheap measures can be driven a long way without the drawing getting
    better, and the only way to notice is to ask the evaluator while it is
    happening."""
    seen: list[int] = []

    def rank(nodes):
        seen.append(len(nodes))
        for i, node in enumerate(nodes):
            node.metrics[FRONT_SCORE] = 0.5 - i * 0.1
        return nodes

    engine = MultiprocessSearchEngine(
        workers=1,
        strategy=FakeStrategy(),
        max_total_tasks=4,
        rank_front=rank,
    )
    for task_id in range(1, 5):
        engine.unscored_q.put(
            Result(task_id=task_id, parent_id=1, valid=True, measured=True, payload="p")
        )

    engine.run(
        initial_nodes=[
            SearchNode(valid=True, id=1, parent_id=0, state=ChainState(payload=None))
        ],
        max_wall_seconds=None,
        active_pool_size=3,
        generation_size=1,
        epoch_eval_interval=1,
    )

    assert seen, "the evaluator was never consulted mid-run"


def test_a_candidate_measuring_as_its_parent_is_rejected_and_charged():
    """An operator can rewrite the markup and leave the render untouched --
    reordering elements that do not overlap is the clearest case. The result
    differs in bytes and in nothing the search can perceive, so the byte
    comparison in the worker cannot see it.

    Those are worse than wasted: identical objectives cannot be ranked against
    the parent, so the candidate survives wherever the parent does and the
    policy is told the operator succeeded. Measured on one run, 58% of all
    candidates were of this kind and the operator producing them held 74% of the
    policy's weight."""

    class RecordingPolicy:
        def __init__(self):
            self.updates = []

        def select(self):
            return "reorder"

        def update(self, operator, reward):
            self.updates.append((operator, reward))

    policy = RecordingPolicy()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=FakeStrategy(), max_total_tasks=2
    )
    parent_metrics = {"edge": 0.4, "colour": 0.2}
    for task_id in (1, 2):
        engine.unscored_q.put(
            Result(
                task_id=task_id,
                parent_id=1,
                valid=True,
                measured=True,
                payload="p",
                metrics=dict(parent_metrics),
                operator="reorder",
            )
        )

    engine.run(
        initial_nodes=[
            SearchNode(
                valid=True,
                id=1,
                parent_id=0,
                state=ChainState(payload=None),
                metrics=dict(parent_metrics),
            )
        ],
        max_wall_seconds=None,
        active_pool_size=4,
        generation_size=1,
        operator_policy=policy,
    )

    # Neither child entered the pool, and the operator was charged for both.
    assert policy.updates == [("reorder", 0.0), ("reorder", 0.0)]


def test_a_candidate_that_moves_any_measure_is_kept():
    """The test above must not be catching everything: a real change on a single
    objective is still a real change."""

    class RecordingPolicy:
        def __init__(self):
            self.updates = []

        def select(self):
            return "nudge"

        def update(self, operator, reward):
            self.updates.append((operator, reward))

    policy = RecordingPolicy()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=FakeStrategy(), max_total_tasks=1
    )
    engine.unscored_q.put(
        Result(
            task_id=1,
            parent_id=1,
            valid=True,
            measured=True,
            payload="p",
            metrics={"edge": 0.4, "colour": 0.19},
            operator="nudge",
        )
    )

    engine.run(
        initial_nodes=[
            SearchNode(
                valid=True,
                id=1,
                parent_id=0,
                state=ChainState(payload=None),
                metrics={"edge": 0.4, "colour": 0.2},
            )
        ],
        max_wall_seconds=None,
        active_pool_size=4,
        generation_size=1,
        operator_policy=policy,
    )

    assert [name for name, _ in policy.updates] == ["nudge"]


def _measured(node_id: int, edge: float) -> SearchNode:
    return SearchNode(
        id=node_id,
        state=ChainState(payload=None),
        parent_id=0,
        metrics={"edge": edge, "colour": 0.1, "shape": 0.1, "detail": 0.1},
        valid=True,
    )
