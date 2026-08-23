import numpy as np
from PIL import Image

from vectrify.score.compare import Comparison
from vectrify.score.segments import (
    Segment,
    save_segments,
    segment_error,
    segment_target,
)


def test_segment_error_only_reads_pixels_inside_its_mask():
    comparison = Comparison(
        colour=np.array([[0.0, 1.0], [0.0, 1.0]]),
        reference_edges=np.zeros((2, 2)),
        candidate_edges=np.zeros((2, 2)),
    )

    assert segment_error(comparison, np.array([[True, False], [True, False]])) == 0.0
    assert segment_error(comparison, np.array([[False, True], [False, True]])) == 0.5


def test_segment_error_is_worst_for_an_empty_mask():
    comparison = Comparison(
        colour=np.zeros((2, 2)),
        reference_edges=np.zeros((2, 2)),
        candidate_edges=np.zeros((2, 2)),
    )

    assert segment_error(comparison, np.zeros((2, 2), dtype=bool)) == 1.0


def test_detail_segment_prioritises_edges_over_colour():
    comparison = Comparison(
        colour=np.ones((2, 2)),
        reference_edges=np.zeros((2, 2)),
        candidate_edges=np.zeros((2, 2)),
    )
    mask = np.ones((2, 2), dtype=bool)

    assert segment_error(comparison, mask, detail=True) == 0.25


def test_segment_target_returns_the_requested_disjoint_partition():
    image = Image.new("RGB", (16, 16), "white")

    segments = segment_target(image, max_regions=8)

    assert len(segments) == 8
    coverage = np.zeros((16, 16), dtype=int)
    for segment in segments:
        coverage += segment.mask
    assert np.all(coverage == 1)


def test_save_segments_writes_directly_to_the_run_directory(tmp_path):
    save_segments(
        [Segment(index=0, label_id=3, mask=np.array([[True, False]]))], tmp_path
    )

    assert (tmp_path / "segments.png").is_file()
    assert not (tmp_path / "segments.json").exists()
