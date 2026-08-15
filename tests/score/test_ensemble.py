from PIL import Image

from vectrify.score.ensemble import EnsembleScorer, PanelReference


def _panel(*rows: list[float]) -> tuple[EnsembleScorer, PanelReference]:
    """A panel whose members report the distances the test dictates.

    Stubbed at the per-member distances rather than at the encoders: the panel
    cuts every picture into tiles and compares the embeddings itself, so faking
    an encoder would mean faking image decoding and tiling too, none of which
    is what these tests are about. What is under test is the vote.
    """
    scorer = EnsembleScorer.__new__(EnsembleScorer)
    scorer._members = [None] * len(rows)  # type: ignore[assignment]
    scorer._names = tuple(f"m{i}" for i in range(len(rows)))
    reference = PanelReference(image=Image.new("RGB", (12, 12)), tiles=[])

    def distances(_reference, images):
        return [list(row[: len(images)]) for row in rows]

    scorer._distances = distances  # type: ignore[method-assign]
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


def test_a_tie_on_votes_is_settled_by_mean_rank():
    """Wins are integers, so a front of tens ties often, and the caller keeps
    only the best few as parents. Leaving ties to fall through to pool order
    would decide the next epoch arbitrarily."""
    # Every member ranks 0 and 1 adjacently but 2 last, so 0 and 1 tie on wins
    # against each other while every member places 0 ahead of 1.
    scorer, reference = _panel(
        [0.10, 0.11, 0.90],
        [0.10, 0.11, 0.90],
        [0.10, 0.11, 0.90],
    )

    ranked = scorer.rank(reference, CANDIDATES)

    assert ranked[0] < ranked[1] < ranked[2]
    assert len(set(ranked)) == 3, "candidates were left tied"


def test_the_tie_break_never_overrides_a_vote():
    """A candidate the majority puts ahead must stay ahead however lopsided the
    mean ranks are, or the tie-break has quietly become the ranking."""
    # 0 wins the majority (three members of five), while the two dissenting
    # members rank it dead last by a wide margin.
    scorer, reference = _panel(
        [0.10, 0.20, 0.30],
        [0.10, 0.20, 0.30],
        [0.10, 0.20, 0.30],
        [0.99, 0.01, 0.02],
        [0.99, 0.01, 0.02],
    )

    ranked = scorer.rank(reference, CANDIDATES)

    assert ranked[0] < ranked[1], "the tie-break reordered across a majority"
