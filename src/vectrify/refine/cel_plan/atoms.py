"""Immutable, complete source-pixel atom splits and rebuilt branch graphs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import find_objects

from vectrify.refine.cel_plan.graph import build, shared
from vectrify.refine.cel_plan.model import Evidence, Graph, StageInterruptedError, Work

MAX_PIXELS = 1536**2
MAX_CUTS = 64
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
class Atoms:
    source: str
    shape: tuple[int, int]
    count: int
    cuts: tuple[Cut, ...] = ()

    def __post_init__(self):
        if (
            not isinstance(self.source, str)
            or len(self.source) != 64
            or any(c not in "0123456789abcdef" for c in self.source)
            or not isinstance(self.shape, tuple)
            or any(type(i) is not int for i in self.shape)
            or type(self.count) is not int
            or not isinstance(self.cuts, tuple)
            or any(not isinstance(c, Cut) for c in self.cuts)
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
        height, width = self.shape
        for index, cut in enumerate(self.cuts):
            if (
                type(cut.parent) is not int
                or not isinstance(cut.areas, tuple)
                or len(cut.areas) != 2
                or any(type(i) is not int for i in cut.areas)
                or not isinstance(cut.left, tuple)
                or any(
                    not isinstance(r, tuple)
                    or len(r) != 3
                    or any(type(i) is not int for i in r)
                    for r in cut.left
                )
            ):
                raise ValueError("Invalid source atom split schema")
            runs += len(cut.left)
            if runs > MAX_RUNS:
                raise ValueError("Source atom bounds exceeded")
            if (
                cut.parent < 0
                or cut.parent >= self.count + 2 * index
                or cut.parent in retired
                or min(cut.areas) <= 0
                or not cut.left
            ):
                raise ValueError("Invalid source atom split lineage")
            retired.add(cut.parent)
            previous = (-1, 0, 0)
            area = 0
            for y, start, end in cut.left:
                if (
                    not 0 <= y < height
                    or not 0 <= start < end <= width
                    or y < previous[0]
                    or (y == previous[0] and start < previous[2])
                ):
                    raise ValueError("Source atom runs must be sorted and disjoint")
                previous = (y, start, end)
                area += end - start
            if area != cut.areas[0]:
                raise ValueError("Source atom split area disagrees with runs")

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
        for index, cut in enumerate(self.cuts[start:], start=start):
            if cut.parent in leaves:
                leaves.remove(cut.parent)
                leaves.update((self.count + 2 * index, self.count + 2 * index + 1))
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
        for index, cut in enumerate(self.cuts):
            if work.interrupted:
                raise StageInterruptedError("Source atom rebuilding interrupted")
            low = self.count + 2 * index
            root = roots[cut.parent]
            if root in original.hidden or original.regions[root].fixed:
                raise ValueError("Protected source atoms cannot be split")
            roots.extend((root, root))
            area = 0
            flat = labels.ravel()
            for start in range(0, flat.size, CHUNK_PIXELS):
                if work.interrupted:
                    raise StageInterruptedError("Source atom rebuilding interrupted")
                part = flat[start : start + CHUNK_PIXELS]
                own = part == cut.parent
                area += int(own.sum())
                part[own] = low + 1
            if area != sum(cut.areas):
                raise ValueError("Source atom split does not retain complete support")
            for y, start, end in cut.left:
                if work.interrupted:
                    raise StageInterruptedError("Source atom rebuilding interrupted")
                if not np.all(labels[y, start:end] == low + 1):
                    raise ValueError("Source atom run claims another owner's pixels")
                labels[y, start:end] = low
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
        if graph.labels.shape != self.shape or len(
            graph.regions
        ) != self.count + 2 * len(self.cuts):
            raise ValueError("Source split graph does not match its atom namespace")
        if (self.cuts and graph.source_atoms != self.key) or (
            not self.cuts and identity(graph.labels) != self.source
        ):
            raise ValueError("Source split graph does not match its atom namespace")
        cuts = list(self.cuts)
        run_count = sum(len(c.left) for c in cuts)
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
            if len(cuts) >= MAX_CUTS:
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
            child = self.count + 2 * len(cuts)
            cuts.append(Cut(member, tuple(runs), (low_area, high_area)))
            run_count += len(runs)
            left_members.append(child)
            right_members.append(child + 1)
        refined = Atoms(self.source, self.shape, self.count, tuple(cuts))
        return refined, tuple(sorted(left_members)), tuple(sorted(right_members))

    def partition(
        self, graph: Graph, members, classes: np.ndarray, count: int, work: Work
    ):
        """Partition complete atoms into several cells without rebuilding per cut.

        A source atom crossing cells is retired through exact binary RLE cuts.
        Every child retains its whole support. The existing lineage, run, pixel
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
            or len(graph.regions) != self.count + 2 * len(self.cuts)
            or (self.cuts and graph.source_atoms != self.key)
            or (not self.cuts and identity(graph.labels) != self.source)
        ):
            raise ValueError("Source cells do not match their atom namespace")
        cuts = list(self.cuts)
        run_count = sum(len(c.left) for c in cuts)
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
                if len(cuts) >= MAX_CUTS:
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
                child = self.count + 2 * len(cuts)
                cuts.append(Cut(parent, tuple(runs), (low_area, high_area)))
                run_count += len(runs)
                cells[int(cell)].append(child)
                remaining &= ~selected
                parent = child + 1
        refined = Atoms(self.source, self.shape, self.count, tuple(cuts))
        return refined, tuple(tuple(sorted(c)) for c in cells)

    def metadata(self) -> dict:
        return {
            "version": 1,
            "source": self.source,
            "shape": self.shape,
            "count": self.count,
            "cuts": [
                {"parent": c.parent, "left": c.left, "areas": c.areas}
                for c in self.cuts
            ],
        }

    @classmethod
    def from_metadata(cls, data):
        if data is None:
            return None
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("Unknown source atom format")
        try:
            if (
                len(data["cuts"]) > MAX_CUTS
                or sum(len(c["left"]) for c in data["cuts"]) > MAX_RUNS
            ):
                raise ValueError("Source atom bounds exceeded")
            return cls(
                data["source"],
                tuple(data["shape"]),
                data["count"],
                tuple(
                    Cut(
                        c["parent"],
                        tuple(tuple(r) for r in c["left"]),
                        tuple(c["areas"]),
                    )
                    for c in data["cuts"]
                ),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid source atom schema") from exc
