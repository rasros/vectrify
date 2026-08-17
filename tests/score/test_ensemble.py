import statistics

from PIL import Image

from vectrify.score.ensemble import EnsembleScorer, PanelReference


def _panel(*rows: list[float]) -> tuple[EnsembleScorer, PanelReference]:
    """A panel whose members report the calibrated distances the test dictates.

    Stubbed at the per-member distances rather than at the encoders: the panel
    cuts every picture into tiles and compares the embeddings itself, so faking
    an encoder would mean faking image decoding and tiling too, none of which
    is what these tests are about.
    """
    scorer = EnsembleScorer.__new__(EnsembleScorer)
    scorer._members = [None] * len(rows)  # type: ignore[assignment]
    scorer._names = tuple(f"m{i}" for i in range(len(rows)))
    reference = PanelReference(
        image=Image.new("RGB", (12, 12)), tiles=[], blank=[1.0] * len(rows)
    )

    def distances(_reference, images):
        return [list(row[: len(images)]) for row in rows]

    scorer._distances = distances  # type: ignore[method-assign]
    return scorer, reference


CANDIDATES = [b"0", b"1", b"2"]


def test_the_majority_decides_and_a_dissenting_member_does_not():
    """The property a single scorer cannot have. With three members the median
    is the majority position: one member being idiosyncratic about a drawing
    cannot move the verdict, whatever value it reports."""
    agree = [0.1, 0.2, 0.3]
    dissent = [0.9, 0.9, 0.9]
    scorer, reference = _panel(agree, agree, dissent)

    ranked = scorer.rank(reference, CANDIDATES)

    assert ranked == agree


def test_a_score_does_not_depend_on_the_field_it_was_scored_with():
    """Why this replaced counting rivals beaten. The same drawing has to come
    out the same however it is grouped, or two checks cannot be compared and
    nothing can be cached from one to the next.
    """
    rows = [[0.1, 0.5, 0.9], [0.2, 0.5, 0.8], [0.3, 0.5, 0.7]]
    scorer, reference = _panel(*rows)

    alone = scorer.rank(reference, CANDIDATES[:1])
    with_others = scorer.rank(reference, CANDIDATES)

    assert alone[0] == with_others[0]


def test_scores_are_the_median_of_the_calibrated_members():
    rows = [[0.1, 0.4], [0.2, 0.5], [0.9, 0.6]]
    scorer, reference = _panel(*rows)

    ranked = scorer.rank(reference, CANDIDATES[:2])

    assert ranked[0] == statistics.median([0.1, 0.2, 0.9])
    assert ranked[1] == statistics.median([0.4, 0.5, 0.6])


def test_ranking_is_not_decided_by_one_member_s_scale():
    """Calibration is what earns this. Raw cosine distances come from three
    embedding spaces of different widths, and an uncalibrated combination is
    decided by whichever member spreads widest -- one model steering the run,
    which is the thing a panel exists to prevent."""
    narrow = [0.10, 0.11]
    also_narrow = [0.10, 0.11]
    wide_but_opposed = [0.90, 0.10]
    scorer, reference = _panel(narrow, also_narrow, wide_but_opposed)

    ranked = scorer.rank(reference, CANDIDATES[:2])

    # The wide member prefers the second candidate by a distance that dwarfs
    # the others; the majority still decides.
    assert ranked[0] < ranked[1]


def test_an_empty_field_ranks_to_nothing():
    scorer, reference = _panel([0.1], [0.2], [0.3])
    assert scorer.rank(reference, []) == []
