from PIL import Image

from vectrify.score.ensemble import EnsembleScorer, PanelReference


class _FixedMember:
    """Stands in for an encoder, returning distances decided by the test."""

    def __init__(self, distances: list[float]):
        self._distances = distances

    def prepare_reference(self, original_rgb):
        return original_rgb

    def score(self, reference, candidate_png):
        _ = reference
        return self._distances[int(candidate_png)]

    def score_many(self, reference, candidate_pngs):
        _ = reference
        return [self._distances[int(png)] for png in candidate_pngs]


def _panel(*rows: list[float]) -> tuple[EnsembleScorer, PanelReference]:
    scorer = EnsembleScorer.__new__(EnsembleScorer)
    # Stand-ins rather than real encoders: the vote is what is under test, and
    # loading five models would make these tests take minutes.
    scorer._members = [_FixedMember(r) for r in rows]  # type: ignore[assignment]
    scorer._names = tuple(f"m{i}" for i in range(len(rows)))
    reference = PanelReference(image=Image.new("RGB", (4, 4)), references=list(rows))
    return scorer, reference


CANDIDATES = [b"0", b"1", b"2"]


def test_the_majority_outvotes_a_dissenting_member():
    """The property a single scorer cannot have: one member being idiosyncratic
    about a particular pair does not decide the pair."""
    agree = [0.1, 0.2, 0.3]
    dissent = [0.9, 0.2, 0.1]
    scorer, reference = _panel(agree, agree, agree, dissent, dissent)

    ranked = scorer.rank(reference, CANDIDATES)

    assert ranked[0] < ranked[1] < ranked[2]


def test_a_voting_cycle_leaves_every_candidate_ranked():
    """Rock-paper-scissors: each candidate beats the next by a majority. The
    relation cannot be sorted, and no candidate may be dropped or left without
    a position."""
    scorer, reference = _panel(
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3],
        [0.3, 0.1, 0.2],
        [0.3, 0.1, 0.2],
        [0.2, 0.3, 0.1],
    )

    ranked = scorer.rank(reference, CANDIDATES)

    assert len(ranked) == 3
    assert all(0.0 <= value <= 1.0 for value in ranked)


def test_ranking_is_not_decided_by_one_member_s_scale():
    """A member reporting distances an order of magnitude larger than the rest
    would dominate any averaging scheme. It gets one vote here."""
    small = [0.01, 0.02, 0.03]
    huge = [90.0, 60.0, 30.0]
    scorer, reference = _panel(small, small, small, huge, huge)

    ranked = scorer.rank(reference, CANDIDATES)

    assert ranked[0] < ranked[2], "the loud member overruled the majority"


def test_a_single_candidate_falls_back_to_the_mean_distance():
    """Nothing to compare against, so there is no vote to take."""
    scorer, reference = _panel([0.2, 0.4, 0.6], [0.4, 0.4, 0.4])

    assert scorer.rank(reference, [b"0"]) == [(0.2 + 0.4) / 2]
    assert scorer.rank(reference, []) == []
