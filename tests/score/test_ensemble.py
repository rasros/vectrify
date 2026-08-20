import io
import statistics

from PIL import Image

from vectrify.score.ensemble import EnsembleScorer, PanelReference


def _panel(*rows: list[float]) -> tuple[EnsembleScorer, PanelReference]:
    """A panel whose members report the calibrated distances the test dictates.

    Stubbed at the per-member distances rather than at the encoders: the panel
    embeds every picture over two views and compares the embeddings itself, so
    faking an encoder would mean faking image decoding and cropping too, which
    is not what these tests are about.
    """
    scorer = EnsembleScorer.__new__(EnsembleScorer)
    scorer._members = [None] * len(rows)  # type: ignore[assignment]
    scorer._names = tuple(f"m{i}" for i in range(len(rows)))
    reference = PanelReference(
        image=Image.new("RGB", (12, 12)),
        box=(0, 0, 12, 12),
        targets=[],
        blank=[1.0] * len(rows),
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


def test_a_dissenting_member_cannot_decide_against_the_majority():
    """Comparing two candidates' medians reads one member per candidate, and
    they need not be the same one. On a real pair, two members preferred the
    fitted drawing (-0.013, -0.018) and the third did not (+0.010); the median
    landed on the dissenter both times and rejected it. Over 21 real pairs, 4
    were decided against the 2-1 majority.
    """
    png = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 255, 255)).save(png, format="PNG")
    frame = png.getvalue()

    # Two members improve on the second candidate, the middle one worsens.
    scorer, reference = _panel([0.085], [0.087], [0.116])
    before = scorer.score(reference, frame)

    scorer_after, reference_after = _panel([0.072], [0.096], [0.098])
    after = scorer_after.score(reference_after, frame)
    assert after < before, "the majority preferred the second candidate"


def test_the_ink_box_finds_the_drawing_rather_than_the_page():
    from vectrify.score.ensemble import ink_box

    page = Image.new("L", (200, 200), 255)
    page.paste(0, (80, 90, 100, 120))
    left, top, right, bottom = ink_box(page.convert("RGB"), pad=4)
    assert 70 <= left <= 80
    assert 80 <= top <= 90
    assert 100 <= right <= 110
    assert 118 <= bottom <= 128


def test_a_blank_page_has_no_ink_to_crop_to():
    from vectrify.score.ensemble import ink_box

    blank = Image.new("RGB", (64, 48), (255, 255, 255))
    assert ink_box(blank) == (0, 0, 64, 48)


def test_both_views_are_offered_and_a_tiny_crop_falls_back():
    page = Image.new("RGB", (64, 64), (255, 255, 255))
    wide = EnsembleScorer._views(page, (0, 0, 64, 64))
    assert len(wide) == 2
    assert wide[1].size == (64, 64)

    sliver = EnsembleScorer._views(page, (10, 10, 13, 13))
    assert sliver[1] is page, "a crop too small to embed falls back to the page"
