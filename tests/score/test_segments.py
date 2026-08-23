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


def test_segment_target_returns_ranked_sam_regions():
    image = Image.new("RGB", (64, 64), "white")
    for coordinate in range(8, 56):
        image.putpixel((coordinate, coordinate), (0, 0, 0))

    segments = segment_target(image, max_regions=8)

    assert 1 <= len(segments) <= 8
    assert all(segment.mask.dtype == bool for segment in segments)
    coverage = np.sum([segment.mask for segment in segments], axis=0)
    assert coverage.max() <= 1


def test_segment_target_returns_nonempty_clusters_for_sparse_targets():
    image = Image.new("RGB", (32, 32), "white")
    image.putpixel((0, 0), (255, 0, 0))

    segments = segment_target(image, max_regions=8)

    assert min(float(segment.mask.sum()) for segment in segments) > 0.0


def test_save_segments_writes_directly_to_the_run_directory(tmp_path):
    save_segments(
        [Segment(index=0, label_id=3, mask=np.array([[True, False]]))],
        tmp_path,
        Image.new("RGB", (2, 1), "white"),
    )

    assert (tmp_path / "segments.png").is_file()
    assert not (tmp_path / "segments.json").exists()
