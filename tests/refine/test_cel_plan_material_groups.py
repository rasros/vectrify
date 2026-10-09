"""Dynamic material grouping preserves supported differences and ownership."""

import numpy as np
import pytest

from tests.helpers import required
from vectrify.refine.cel_plan import material_groups
from vectrify.refine.cel_plan.material_groups import grouped, ink_paint_links
from vectrify.refine.cel_plan.model import Work


def test_updated_costs_do_not_chain_a_color_ramp_into_one_surface():
    colors = np.repeat(np.array([[0], [20], [40], [60]], float), 3, axis=1)
    sizes = np.full(4, 100)
    edges = [(0, 1), (1, 2), (2, 3)]
    groups, eligible, details = required(
        grouped(colors, sizes, edges, 2, Work.start(10))
    )
    np.testing.assert_array_equal(groups, [0, 0, 2, 2])
    assert eligible.all()
    assert details["remaining_materials"] == 2
    assert not groups.flags.writeable
    assert not eligible.flags.writeable
    np.testing.assert_array_equal(colors[:, 0], [0, 20, 40, 60])
    np.testing.assert_array_equal(sizes, 100)


def test_tiny_disconnected_supports_keep_original_owners_outside_material_budget():
    colors = np.repeat(np.array([[0], [20], [40], [60], [0], [0]], float), 3, axis=1)
    sizes = np.array([100, 100, 100, 100, 25, 25])
    groups, eligible, details = required(
        grouped(colors, sizes, [(0, 1), (1, 2), (2, 3), (4, 5)], 2, Work.start(10))
    )
    np.testing.assert_array_equal(groups, [0, 0, 2, 2, 4, 5])
    np.testing.assert_array_equal(eligible, [True, True, True, True, False, False])
    assert details["retained_disconnected_owners"] == 2
    assert details["substantial_components"] == 1
    assert not details["budget_unmet"]


def test_disconnected_large_materials_do_not_merge_to_satisfy_a_budget():
    colors, sizes = np.zeros((4, 3)), np.full(4, 150)
    groups, _, details = required(
        grouped(colors, sizes, [(0, 1), (2, 3)], 1, Work.start(10))
    )
    np.testing.assert_array_equal(groups, [0, 0, 2, 2])
    assert details["budget_unmet"]
    assert details["remaining_materials"] == 2


def test_supported_ink_cannot_merge_with_similar_material_to_meet_a_budget():
    colors = np.repeat(np.array([[100], [110], [120]], float), 3, axis=1)
    sizes = np.full(3, 1000)
    kinds = np.array([False, True, False])
    groups, _, details = required(
        grouped(colors, sizes, [(0, 1), (1, 2)], 1, Work.start(10), kinds=kinds)
    )
    np.testing.assert_array_equal(groups, [0, 1, 2])
    assert details["budget_unmet"]
    assert details["ink_materials"] == 1
    np.testing.assert_array_equal(kinds, [False, True, False])


def test_disconnected_matching_ink_can_share_paint_without_merging_material():
    colors = np.repeat(np.array([[11], [100], [12], [100], [24]], float), 3, axis=1)
    sizes, kinds = np.full(5, 1000), np.array([1, 0, 1, 0, 1])
    links = ink_paint_links(colors, kinds, Work.start(10))
    assert links == [(0, 2)]
    groups, _, details = required(
        grouped(
            colors,
            sizes,
            [(0, 1), (2, 3)],
            3,
            Work.start(10),
            kinds=kinds,
            paint_edges=links,
        )
    )
    np.testing.assert_array_equal(groups, [0, 1, 0, 3, 4])
    assert details["ink_paint_links"] == 1
    assert details["ink_materials"] == 2
    assert details["budget_unmet"]


def test_paint_links_do_not_promote_tiny_disconnected_support_into_eligibility():
    colors, kinds = np.full((3, 3), 10), np.ones(3, np.uint8)
    groups, eligible, details = required(
        grouped(
            colors,
            np.full(3, 100),
            [],
            1,
            Work.start(10),
            kinds=kinds,
            paint_edges=ink_paint_links(colors, kinds, Work.start(10)),
        )
    )
    np.testing.assert_array_equal(groups, [0, 1, 2])
    assert not eligible.any()
    assert details["ink_paint_links"] == 0


def test_ink_paint_link_discovery_is_bounded_and_interruption_is_atomic():
    colors, kinds = np.full((4096, 3), 10), np.ones(4096, np.uint8)
    assert len(required(ink_paint_links(colors, kinds, Work.start(10)))) == 4095
    work = Work.start(10)
    work.stop.set()
    assert ink_paint_links(colors, kinds, work) is None


@pytest.mark.parametrize("paint_edges", [[(0, 1)], [(0, 0)], [(0, 3)]])
def test_invalid_ink_paint_links_are_rejected(paint_edges):
    with pytest.raises(ValueError, match="Co-paint"):
        grouped(
            np.zeros((2, 3)),
            np.full(2, 1000),
            [],
            1,
            Work.start(10),
            kinds=np.array([1, 0]),
            paint_edges=paint_edges,
        )


def test_queue_rebuild_preserves_the_same_complete_connected_partition(monkeypatch):
    colors = np.repeat(np.arange(24)[:, None], 3, axis=1).astype(float)
    sizes = np.full(24, 100)
    edges = [(i, j) for i in range(24) for j in range(i + 1, min(24, i + 4))]
    expected, _, _ = required(grouped(colors, sizes, edges, 4, Work.start(10)))
    monkeypatch.setattr(material_groups, "REBUILD_QUEUE_AT", 90)
    actual, _, details = required(grouped(colors, sizes, edges, 4, Work.start(10)))
    np.testing.assert_array_equal(actual, expected)
    assert details["queue_rebuilds"] > 0
    assert details["queue_peak"] <= 90 + len(sizes)


def test_interruption_during_a_union_discards_the_whole_partial_hierarchy():
    colors = np.repeat(np.arange(16)[:, None], 3, axis=1).astype(float)
    sizes = np.full(16, 100)
    edges = [(i, i + 1) for i in range(15)]

    class Interrupted:
        calls = 0

        @property
        def interrupted(self):
            self.calls += 1
            return self.calls > 80

    work = Interrupted()
    assert grouped(colors, sizes, edges, 2, work) is None
    assert work.calls > 80
    np.testing.assert_array_equal(colors[:, 0], np.arange(16))
    np.testing.assert_array_equal(sizes, 100)


@pytest.mark.parametrize("scale", [0.5, 2, 4])
def test_uniform_sample_density_does_not_change_the_material_interpretation(scale):
    colors = np.repeat(np.array([[0], [20], [40], [60]], float), 3, axis=1)
    groups, _, _ = required(
        grouped(
            colors,
            np.full(4, 1000 * scale),
            [(0, 1), (1, 2), (2, 3)],
            2,
            Work.start(10),
        )
    )
    np.testing.assert_array_equal(groups, [0, 0, 2, 2])
