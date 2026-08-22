"""Rank candidates with a panel of image encoders."""

import io
import logging
import statistics
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from vectrify.score.base import Scorer
from vectrify.score.embedding import EmbeddingScorer
from vectrify.score.utils import MAX_SCORE

log = logging.getLogger(__name__)

PANEL_MODELS: tuple[str, ...] = (
    "facebook/dinov2-small",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "google/siglip-base-patch16-224",
)


def ink_box(image: Image.Image, pad: int = 8, threshold: int = 250):
    """The drawing's own extent, so the subject can fill the frame.

    A page is mostly blank. Every member resizes what it is given to 224px, so
    a 700px drawing arrives with its subject a small fraction of the input and
    its strokes about a pixel wide -- and a stroke moved a few pixels is then
    below what the encoder can resolve. Measured by displacing a whole beak
    group and asking each member to order the result, two of the three called
    an 8px displacement BETTER than no displacement at all.
    """
    grey = np.asarray(image.convert("L"), dtype=np.float32)
    rows, columns = np.where(grey < threshold)
    if rows.size == 0:
        return (0, 0, image.width, image.height)
    return (
        max(0, int(columns.min()) - pad),
        max(0, int(rows.min()) - pad),
        min(image.width, int(columns.max()) + pad),
        min(image.height, int(rows.max()) + pad),
    )


@dataclass
class PanelReference:
    image: Image.Image
    box: tuple[int, int, int, int]
    targets: list[list[Any]]
    blank: list[float]


class EnsembleScorer(Scorer):
    """Ranks by majority vote across a panel of encoders."""

    def __init__(self, model_names: tuple[str, ...] = PANEL_MODELS):
        self._members = [EmbeddingScorer(model_name=name) for name in model_names]
        self._names = model_names

    def validate_environment(self) -> None:
        self._members[0].validate_environment()

    def prepare_reference(self, original_rgb: Image.Image) -> PanelReference:
        box = ink_box(original_rgb)
        views = self._views(original_rgb, box)
        reference = PanelReference(
            image=original_rgb,
            box=box,
            targets=[
                [m.embed_images([view])[0] for view in views] for m in self._members
            ],
            blank=[],
        )
        empty = Image.new("RGB", original_rgb.size, (255, 255, 255))
        reference.blank.extend(
            max(d[0], 1e-6) for d in self._raw_distances(reference, [empty])
        )
        return reference

    @staticmethod
    def _views(image: Image.Image, box: tuple[int, int, int, int]):
        """The whole page, and the drawing cropped to the target's ink."""
        cropped = image.crop(box)
        return [image, cropped if min(cropped.size) >= 8 else image]

    def _raw_distances(self, reference: PanelReference, images: list[Image.Image]):
        """Each member's distance to every image, averaged over its views.

        One batched forward pass per member per view: with a single embedding
        per image and view, the whole field fits in two calls.
        """
        per_view = [
            [self._views(image, reference.box)[index] for image in images]
            for index in range(len(reference.targets[0]))
        ]
        per_member = []
        for member, targets in zip(self._members, reference.targets, strict=True):
            sums = [0.0] * len(images)
            for target, batch in zip(targets, per_view, strict=True):
                got = member.embed_images(batch)
                values = [float(1.0 - (target * row).sum()) for row in got]
                sums = [a + b for a, b in zip(sums, values, strict=True)]
            per_member.append([v / len(targets) for v in sums])
        return per_member

    def _distances(self, reference: PanelReference, images: list[Image.Image]):
        """Per-member distances, each as a fraction of that member's distance
        from the target to a blank canvas. 0 is the target itself and about 1
        is as wrong as an empty drawing, on every member and every target."""
        raw = self._raw_distances(reference, images)
        return [
            [value / scale for value in member]
            for member, scale in zip(raw, reference.blank, strict=True)
        ]

    def score(self, reference: PanelReference, candidate_png: bytes) -> float:
        """The panel's verdict on one candidate: the median calibrated distance.

        The median rather than the mean, and that is the whole panel argument
        in absolute form. With three members the median is the majority
        position: for any standard you might hold a candidate to, "the panel
        says it meets this" is true exactly when the median says so, and a
        member that is idiosyncratic about this particular drawing cannot move
        it. The pairwise vote said the same thing about pairs; this says it
        about candidates, which is what lets two scores be compared at all.

        Absolute, so it means the same thing in every call and in every run --
        the property `rank` could not have, since counting rivals beaten only
        describes the field a candidate was ranked against.
        """
        image = self._decode(candidate_png)
        if image is None:
            return MAX_SCORE
        values = [d[0] for d in self._distances(reference, [image])]
        return statistics.fmean(values) if values else MAX_SCORE

    def rank(
        self, reference: PanelReference, candidate_pngs: list[bytes]
    ) -> list[float]:
        """Score every candidate, lower being better.

        One absolute number each, so the caller may compare them with anything
        else the panel has scored -- a candidate from an earlier check, from an
        earlier epoch, or from another run entirely.

        This used to put every pair to the panel and count rivals beaten, which
        ranked a field correctly and said nothing outside it: the same drawing
        scored differently depending on who it was ranked against, so two calls
        could not be compared and nothing could be cached between them.
        """
        if not candidate_pngs:
            return []

        images = [
            self._decode(png)
            or Image.new("RGB", reference.image.size, (255, 255, 255))
            for png in candidate_pngs
        ]

        per_member = self._distances(reference, images)
        return [
            statistics.median([member[i] for member in per_member])
            for i in range(len(images))
        ]

    @staticmethod
    def _decode(candidate_png: bytes) -> Image.Image | None:
        try:
            return Image.open(io.BytesIO(candidate_png)).convert("RGB")
        except Exception:
            return None
