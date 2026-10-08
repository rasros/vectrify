"""Immutable, complete source-pixel atom splits and rebuilt branch graphs."""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import find_objects

from vectrify.refine.cel_plan.graph import build, shared
from vectrify.refine.cel_plan.model import Evidence, Graph, StageInterruptedError, Work

MAX_PIXELS = 1536**2
MAX_CUTS = 64
MAX_CHILDREN = 128
MAX_PARTS = 64
MAX_RUNS = 16_384
CHUNK_PIXELS = 65_536


def identity(labels: np.ndarray) -> str:
    digest = hashlib.sha256(repr(labels.shape).encode())
    digest.update(np.asarray(labels, dtype="<i4").tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Cut:
    parent: int
    left: tuple[tuple[int, int, int], ...]
    areas: tuple[int, int]


@dataclass(frozen=True)
class MultiCut:
    """One parent partitioned directly, with its last child's runs implicit."""

    parent: int
    groups: tuple[tuple[tuple[int, int, int], ...], ...]
    areas: tuple[int, ...]


def _blocks(cut):
    return (cut.left,) if isinstance(cut, Cut) else cut.groups


def _run_count(cuts):
    return sum(len(block) for cut in cuts for block in _blocks(cut))


@dataclass(frozen=True)
class Atoms:
    source: str
    shape: tuple[int, int]
    count: int
    cuts: tuple[Cut | MultiCut, ...] = ()

    def __post_init__(self):
        if (
            not isinstance(self.source, str)
            or len(self.source) != 64
            or any(c not in "0123456789abcdef" for c in self.source)
            or not isinstance(self.shape, tuple)
            or any(type(i) is not int for i in self.shape)
            or type(self.count) is not int
            or not isinstance(self.cuts, tuple)
            or any(not isinstance(c, (Cut, MultiCut)) for c in self.cuts)
        ):
            raise ValueError("Invalid source atom schema")
        if (
            len(self.shape) != 2
            or min(self.shape) <= 0
            or np.prod(self.shape) > MAX_PIXELS
            or self.count < 1
            or len(self.cuts) > MAX_CUTS
        ):
            raise ValueError("Source atom bounds exceeded")
        retired = set()
        runs = 0
        next_child = self.count
        height, width = self.shape
        for cut in self.cuts:
            blocks = _blocks(cut)
            if (
                type(cut.parent) is not int
                or not isinstance(cut.areas, tuple)
                or not 2 <= len(cut.areas) <= MAX_PARTS
                or (isinstance(cut, Cut) and len(cut.areas) != 2)
                or (isinstance(cut, MultiCut) and len(cut.areas) < 3)
                or any(type(i) is not int for i in cut.areas)
                or not isinstance(blocks, tuple)
                or len(blocks) != len(cut.areas) - 1
                or any(not isinstance(block, tuple) for block in blocks)
                or any(
                    not isinstance(r, tuple)
                    or len(r) != 3
                    or any(type(i) is not int for i in r)
                    for block in blocks
                    for r in block
                )
            ):
                raise ValueError("Invalid source atom split schema")
            runs += sum(len(block) for block in blocks)
            if (
                runs > MAX_RUNS
                or next_child + len(cut.areas) > self.count + MAX_CHILDREN
            ):
                raise ValueError("Source atom bounds exceeded")
            if (
                cut.parent < 0
                or cut.parent >= next_child
                or cut.parent in retired
                or min(cut.areas) <= 0
                or any(not block for block in blocks)
            ):
                raise ValueError("Invalid source atom split lineage")
            retired.add(cut.parent)
            for block, expected in zip(blocks, cut.areas[:-1], strict=True):
                previous = (-1, 0, 0)
                area = 0
                for y, start, end in block:
                    if (
                        not 0 <= y < height
                        or not 0 <= start < end <= width
                        or y < previous[0]
                        or (y == previous[0] and start < previous[2])
                    ):
                        raise ValueError("Source atom runs must be sorted and disjoint")
                    previous = (y, start, end)
                    area += end - start
                if area != expected:
                    raise ValueError("Source atom split area disagrees with runs")
            # Separate children cannot claim the same source pixel. Replay
            # additionally proves every run belongs to this actual parent.
            previous = (-1, 0, 0)
            for y, start, end in heapq.merge(*blocks):
                if y == previous[0] and start < previous[2]:
                    raise ValueError("Source atom runs must be sorted and disjoint")
                previous = (y, start, end)
            next_child += len(cut.areas)

    @property
    def namespace_count(self):
        return self.count + sum(len(c.areas) for c in self.cuts)

    @property
    def key(self) -> str:
        return hashlib.sha256(repr(self).encode()).hexdigest()

    @classmethod
    def original(cls, graph: Graph):
        return cls(identity(graph.labels), graph.labels.shape, len(graph.regions))

    def extends(self, previous: Atoms | None) -> bool:
        return previous is None or (
            self.source == previous.source
            and self.shape == previous.shape
            and self.count == previous.count
            and self.cuts[: len(previous.cuts)] == previous.cuts
        )

    def descendants(self, members, start=0) -> tuple[int, ...]:
        leaves = set(members)
        child = self.count + sum(len(c.areas) for c in self.cuts[:start])
        for cut in self.cuts[start:]:
            if cut.parent in leaves:
                leaves.remove(cut.parent)
                leaves.update(range(child, child + len(cut.areas)))
            child += len(cut.areas)
        return tuple(sorted(leaves))

    def labels(self, original: Graph, work: Work) -> np.ndarray:
        if work.interrupted:
            raise StageInterruptedError("Source atom rebuilding interrupted")
        if (
            original.labels.shape != self.shape
            or len(original.regions) != self.count
            or identity(original.labels) != self.source
        ):
            raise ValueError("Source atoms belong to a different original graph")
        if not self.cuts and not original.labels.flags.writeable:
            return original.labels
        labels = original.labels.copy()
        roots = list(range(self.count))
        low = self.count
        for cut in self.cuts:
            if work.interrupted:
                raise StageInterruptedError("Source atom rebuilding interrupted")
            root = roots[cut.parent]
            if root in original.hidden or original.regions[root].fixed:
                raise ValueError("Protected source atoms cannot be split")
            roots.extend([root] * len(cut.areas))
            implicit = low + len(cut.areas) - 1
            area = 0
            flat = labels.ravel()
            for start in range(0, flat.size, CHUNK_PIXELS):
                if work.interrupted:
                    raise StageInterruptedError("Source atom rebuilding interrupted")
                part = flat[start : start + CHUNK_PIXELS]
                own = part == cut.parent
                area += int(own.sum())
                part[own] = implicit
            if area != sum(cut.areas):
                raise ValueError("Source atom split does not retain complete support")
            for index, block in enumerate(_blocks(cut)):
                for y, start, end in block:
                    if work.interrupted:
                        raise StageInterruptedError(
                            "Source atom rebuilding interrupted"
                        )
                    if not np.all(labels[y, start:end] == implicit):
                        raise ValueError(
                            "Source atom run claims another owner's pixels"
                        )
                    labels[y, start:end] = low + index
            low += len(cut.areas)
        if work.interrupted:
            raise StageInterruptedError("Source atom rebuilding interrupted")
        labels.flags.writeable = False
        return labels

    def graph(self, evidence: Evidence, original: Graph, work: Work) -> Graph:
        labels = self.labels(original, work)
        if not self.cuts and all(
            not edge.points.flags.writeable for edge in original.boundaries
        ):
            return replace(original, labels=labels, source_atoms=self.key)
        result = build(evidence, labels, work=work)
        result = shared(result, original, work)
        return replace(result, source_atoms=self.key)

    def split(self, graph: Graph, members, left: np.ndarray, work: Work):
        if left.shape != self.shape or left.dtype != bool:
            raise ValueError("Source split requires a complete boolean classification")
        if (
            graph.labels.shape != self.shape
            or len(graph.regions) != self.namespace_count
        ):
            raise ValueError("Source split graph does not match its atom namespace")
        if (self.cuts and graph.source_atoms != self.key) or (
            not self.cuts and identity(graph.labels) != self.source
        ):
            raise ValueError("Source split graph does not match its atom namespace")
        cuts = list(self.cuts)
        run_count = _run_count(cuts)
        next_child = self.namespace_count
        boxes = find_objects(graph.labels + 1, max_label=len(graph.regions))
        left_members, right_members = [], []
        for member in sorted(set(members)):
            if work.interrupted:
                raise StageInterruptedError("Source atom splitting interrupted")
            if not 0 <= member < len(boxes) or boxes[member] is None:
                raise ValueError("Source split references an inactive atom")
            box = boxes[member]
            own = graph.labels[box] == member
            selected = own & left[box]
            low_area = int(selected.sum())
            high_area = int(own.sum()) - low_area
            if not low_area or not high_area:
                (left_members if low_area else right_members).append(member)
                continue
            if graph.regions[member].fixed or member in graph.hidden:
                raise ValueError("Protected source atoms cannot be split")
            if len(cuts) >= MAX_CUTS or next_child + 2 > self.count + MAX_CHILDREN:
                raise ValueError("Source atom bounds exceeded")
            runs = []
            for y in np.flatnonzero(selected.any(axis=1)):
                if work.interrupted:
                    raise StageInterruptedError("Source atom splitting interrupted")
                changes = np.diff(np.r_[False, selected[y], False].astype(np.int8))
                for start, end in zip(
                    np.flatnonzero(changes == 1),
                    np.flatnonzero(changes == -1),
                    strict=True,
                ):
                    if run_count + len(runs) >= MAX_RUNS:
                        raise ValueError("Source atom bounds exceeded")
                    runs.append(
                        (
                            int(y) + box[0].start,
                            int(start) + box[1].start,
                            int(end) + box[1].start,
                        )
                    )
            child = next_child
            next_child += 2
            cuts.append(Cut(member, tuple(runs), (low_area, high_area)))
            run_count += len(runs)
            left_members.append(child)
            right_members.append(child + 1)
        refined = Atoms(self.source, self.shape, self.count, tuple(cuts))
        return refined, tuple(sorted(left_members)), tuple(sorted(right_members))

    def partition(
        self,
        graph: Graph,
        members,
        classes: np.ndarray,
        count: int,
        work: Work,
        *,
        compact=False,
    ):
        """Partition complete atoms into several cells without rebuilding per cut.

        A crossing atom is retired through exact binary RLE cuts by default,
        or a direct multiway partition when compact is explicit. Every child
        retains its whole support. The existing lineage, run, pixel
        and protected-atom limits apply to the entire atomic operation.
        """
        if (
            classes.shape != self.shape
            or classes.dtype.kind not in "iu"
            or type(count) is not int
            or not 1 <= count <= 64
            or classes.min() < 0
            or classes.max() >= count
        ):
            raise ValueError("Source cells require a complete bounded classification")
        if (
            graph.labels.shape != self.shape
            or len(graph.regions) != self.namespace_count
            or (self.cuts and graph.source_atoms != self.key)
            or (not self.cuts and identity(graph.labels) != self.source)
        ):
            raise ValueError("Source cells do not match their atom namespace")
        if compact:
            return self._compact_partition(graph, members, classes, count, work)
        cuts = list(self.cuts)
        run_count = _run_count(cuts)
        next_child = self.namespace_count
        boxes = find_objects(graph.labels + 1, max_label=len(graph.regions))
        cells = [[] for _ in range(count)]
        for member in sorted(set(members)):
            if work.interrupted:
                raise StageInterruptedError("Source cell splitting interrupted")
            if not 0 <= member < len(boxes) or boxes[member] is None:
                raise ValueError("Source cells reference an inactive atom")
            box = boxes[member]
            remaining = graph.labels[box] == member
            values = classes[box]
            occupied = np.unique(values[remaining])
            if len(occupied) > 1 and (
                graph.regions[member].fixed or member in graph.hidden
            ):
                raise ValueError("Protected source atoms cannot be split")
            parent = member
            for index, cell in enumerate(occupied):
                if work.interrupted:
                    raise StageInterruptedError("Source cell splitting interrupted")
                if index == len(occupied) - 1:
                    cells[int(cell)].append(parent)
                    break
                if len(cuts) >= MAX_CUTS or next_child + 2 > self.count + MAX_CHILDREN:
                    raise ValueError("Source atom bounds exceeded")
                selected = remaining & (values == cell)
                low_area = int(selected.sum())
                high_area = int(remaining.sum()) - low_area
                runs = []
                for y in np.flatnonzero(selected.any(axis=1)):
                    if work.interrupted:
                        raise StageInterruptedError("Source cell splitting interrupted")
                    changes = np.diff(np.r_[False, selected[y], False].astype(np.int8))
                    for start, end in zip(
                        np.flatnonzero(changes == 1),
                        np.flatnonzero(changes == -1),
                        strict=True,
                    ):
                        if run_count + len(runs) >= MAX_RUNS:
                            raise ValueError("Source atom bounds exceeded")
                        runs.append(
                            (
                                int(y) + box[0].start,
                                int(start) + box[1].start,
                                int(end) + box[1].start,
                            )
                        )
                child = next_child
                next_child += 2
                cuts.append(Cut(parent, tuple(runs), (low_area, high_area)))
                run_count += len(runs)
                cells[int(cell)].append(child)
                remaining &= ~selected
                parent = child + 1
        refined = Atoms(self.source, self.shape, self.count, tuple(cuts))
        return refined, tuple(tuple(sorted(c)) for c in cells)

    def _compact_partition(self, graph, members, classes, count, work):
        """Retire each parent once, preserving the same entry/child/run limits.

        Binary staging introduces intermediate regions for multi-class parents.
        A direct partition removes those intermediate nodes rather than raising
        the 64-entry, 128-child or 16,384-run allowances. One largest child is
        implicit; all other source supports have exact, disjoint RLE evidence.
        """
        cuts = list(self.cuts)
        run_count = _run_count(cuts)
        next_child = self.namespace_count
        boxes = find_objects(graph.labels + 1, max_label=len(graph.regions))
        cells = [[] for _ in range(count)]
        for member in sorted(set(members)):
            if work.interrupted:
                raise StageInterruptedError("Source cell splitting interrupted")
            if not 0 <= member < len(boxes) or boxes[member] is None:
                raise ValueError("Source cells reference an inactive atom")
            box = boxes[member]
            own = graph.labels[box] == member
            values = classes[box]
            occupied, areas = np.unique(values[own], return_counts=True)
            if len(occupied) == 1:
                cells[int(occupied[0])].append(member)
                continue
            if graph.regions[member].fixed or member in graph.hidden:
                raise ValueError("Protected source atoms cannot be split")
            if (
                len(cuts) >= MAX_CUTS
                or next_child + len(occupied) > self.count + MAX_CHILDREN
            ):
                raise ValueError("Source atom bounds exceeded")
            implicit = int(np.argmax(areas))
            order = [i for i in range(len(occupied)) if i != implicit] + [implicit]
            blocks = []
            for index in order[:-1]:
                selected = own & (values == occupied[index])
                runs = []
                for y in np.flatnonzero(selected.any(axis=1)):
                    if work.interrupted:
                        raise StageInterruptedError("Source cell splitting interrupted")
                    changes = np.diff(np.r_[False, selected[y], False].astype(np.int8))
                    for start, end in zip(
                        np.flatnonzero(changes == 1),
                        np.flatnonzero(changes == -1),
                        strict=True,
                    ):
                        if run_count + len(runs) >= MAX_RUNS:
                            raise ValueError("Source atom bounds exceeded")
                        runs.append(
                            (
                                int(y) + box[0].start,
                                int(start) + box[1].start,
                                int(end) + box[1].start,
                            )
                        )
                block = tuple(runs)
                blocks.append(block)
                run_count += len(block)
            actual_areas = tuple(int(areas[i]) for i in order)
            cuts.append(
                Cut(member, blocks[0], (actual_areas[0], actual_areas[1]))
                if len(occupied) == 2
                else MultiCut(member, tuple(blocks), actual_areas)
            )
            for offset, index in enumerate(order):
                cells[int(occupied[index])].append(next_child + offset)
            next_child += len(occupied)
        if work.interrupted:
            raise StageInterruptedError("Source cell splitting interrupted")
        refined = Atoms(self.source, self.shape, self.count, tuple(cuts))
        return refined, tuple(tuple(sorted(c)) for c in cells)

    def metadata(self) -> dict:
        return {
            "version": 2 if any(isinstance(c, MultiCut) for c in self.cuts) else 1,
            "source": self.source,
            "shape": self.shape,
            "count": self.count,
            "cuts": [
                {"parent": c.parent, "left": c.left, "areas": c.areas}
                if isinstance(c, Cut)
                else {"parent": c.parent, "groups": c.groups, "areas": c.areas}
                for c in self.cuts
            ],
        }

    @classmethod
    def from_metadata(cls, data):
        if data is None:
            return None
        if (
            not isinstance(data, dict)
            or type(data.get("version")) is not int
            or data["version"] not in (1, 2)
        ):
            raise ValueError("Unknown source atom format")
        try:
            if len(data["cuts"]) > MAX_CUTS:
                raise ValueError("Source atom bounds exceeded")
            cuts: list[Cut | MultiCut] = []
            runs = children = 0
            for c in data["cuts"]:
                if ("left" in c) == ("groups" in c) or (
                    "groups" in c and data["version"] != 2
                ):
                    raise ValueError("Invalid source atom schema")
                blocks = [c["left"]] if "left" in c else c["groups"]
                if not 1 <= len(blocks) < MAX_PARTS:
                    raise ValueError("Source atom bounds exceeded")
                runs += sum(len(block) for block in blocks)
                children += len(blocks) + 1
                if runs > MAX_RUNS or children > MAX_CHILDREN:
                    raise ValueError("Source atom bounds exceeded")
                areas = tuple(c["areas"])
                encoded = tuple(tuple(tuple(r) for r in block) for block in blocks)
                if len(areas) != len(encoded) + 1 or ("groups" in c and len(areas) < 3):
                    raise ValueError("Invalid source atom schema")
                cuts.append(
                    Cut(c["parent"], encoded[0], (areas[0], areas[1]))
                    if "left" in c
                    else MultiCut(c["parent"], encoded, areas)
                )
            return cls(
                data["source"],
                tuple(data["shape"]),
                data["count"],
                tuple(cuts),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid source atom schema") from exc
