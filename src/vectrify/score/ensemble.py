"""A panel of image encoders that ranks candidates by majority vote.

No single encoder can be shown to judge this task well. On MIEB's NIGHTS task,
the one leaderboard task built on human similarity judgements, the spread
across the whole field is narrow (0.2646 at the top, 0.2363 at the bottom of 55
models), and NIGHTS is diffusion-generated natural images rather than vector
art, so even that ordering is transfer evidence. Measured on our own pooled
candidates the encoders agree with each other only 47-63% of the time, where
50% is chance -- they are close to independent, not four views of one truth.

Picking one encoder therefore stakes the run's direction on a choice nothing
supports. A panel votes instead: each pair of candidates is compared by every
member, and the majority decides. A member that is idiosyncratic on some
particular pair is outvoted, which is the property no single scorer has.

The members are three cheap encoders spanning three training regimes:

    facebook/dinov2-small                        self-supervised
    laion/CLIP-ViT-B-32-laion2B-s34B-b79K        contrastive, LAION-2B
    google/siglip-base-patch16-224               sigmoid pairwise

Five were seated at first, on the reasoning that more voices outvote more
idiosyncrasy. Measured against the distortion screen -- damage of a known
severity in a known order, so no other scorer has to be trusted as the
reference -- that reasoning does not hold here. On every family the panel is
weak at, the five members score within 3 to 7 points of each other: they share
their blind spots rather than covering them, so the extra voices were averaging
noise, not insuring against error. Three score as well as five (96.2% against
96.2% on vector damage), and adding a 428M model to those three buys 0.4.

Several better-placed candidates cannot be used at all -- the DataComp
checkpoints ship open_clip weights with no HuggingFace preprocessor config,
ebind and jina-omni need custom code, and the BLIP retrieval checkpoints load
through AutoModel with a randomly initialised visual_projection, which is the
very layer their image embedding comes from.

Two of the cheap pixel measures selection uses, edge overlap and colour
distance, are deliberately not members. On the same hand-judged pairs they
agree with the human 40% and 44% of the time, the bottom of the field, and they
are already two of the three objectives the search optimises -- seating them
here would make the evaluator agree with those measures by construction.

Five is odd on purpose, so a pairwise majority always exists.
"""

import io
import logging
import statistics
from dataclasses import dataclass
from typing import Any

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


@dataclass
class PanelReference:
    image: Image.Image
    # Each member's embedding of the target, read whole.
    #
    # The panel used to cut every picture into a 5x5 lattice and average the 25
    # cell distances. It ordered deliberate damage slightly better that way --
    # 96.4% against 92.8% on the distortion screen -- but the average is over
    # cells, so a defect confined to one cell arrives divided by 25, and a
    # drawing is not judged by how much damage it holds. Measured on a real
    # run's eye, drawn with its pupil and highlight inverted: at 5 cells a side
    # the eye was 11.8% of one 140px cell and straddled a cell boundary, the
    # three members landed within 0.0005 of each other on distances of 0.036 to
    # 0.096, and the median vote settled that near-tie in favour of the wrong
    # polarity. Read whole, the same pair separates by 0.031 the right way.
    targets: list[Any]
    # Each member's distance from the target to a blank canvas, measured once.
    # It is what makes a member's distance mean something on its own: raw
    # cosine distances come from three different embedding spaces and span
    # different widths, so an uncalibrated average is decided by whichever
    # member happens to spread widest -- one model steering the run, which is
    # what a panel exists to prevent.
    blank: list[float]


class EnsembleScorer(Scorer):
    """Ranks by majority vote across a panel of encoders."""

    def __init__(self, model_names: tuple[str, ...] = PANEL_MODELS):
        self._members = [EmbeddingScorer(model_name=name) for name in model_names]
        self._names = model_names

    def validate_environment(self) -> None:
        self._members[0].validate_environment()

    def prepare_reference(self, original_rgb: Image.Image) -> PanelReference:
        reference = PanelReference(
            image=original_rgb,
            targets=[m.embed_images([original_rgb])[0] for m in self._members],
            blank=[],
        )
        empty = Image.new("RGB", original_rgb.size, (255, 255, 255))
        reference.blank.extend(
            max(d[0], 1e-6) for d in self._raw_distances(reference, [empty])
        )
        return reference

    def _raw_distances(self, reference: PanelReference, images: list[Image.Image]):
        """Each member's distance to every image.

        One batched forward pass per member rather than one per picture: with
        the lattice gone there is a single embedding per image, so the whole
        field fits in one call.
        """
        per_member = []
        for member, target in zip(self._members, reference.targets, strict=True):
            got = member.embed_images(images)
            per_member.append([float(1.0 - (target * row).sum()) for row in got])
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
        try:
            image = Image.open(io.BytesIO(candidate_png)).convert("RGB")
        except Exception:
            return MAX_SCORE
        values = [d[0] for d in self._distances(reference, [image])]
        return statistics.median(values) if values else MAX_SCORE

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

        images = []
        for png in candidate_pngs:
            try:
                images.append(Image.open(io.BytesIO(png)).convert("RGB"))
            except Exception:
                # A candidate that will not open is scored worst rather than
                # failing the field it was ranked with.
                images.append(Image.new("RGB", reference.image.size, (255, 255, 255)))

        per_member = self._distances(reference, images)
        return [
            statistics.median([member[i] for member in per_member])
            for i in range(len(images))
        ]

        images = []
        for png in candidate_pngs:
            try:
                images.append(Image.open(io.BytesIO(png)).convert("RGB"))
            except Exception:
                # A candidate that will not open is scored worst rather than
                # failing the field it was ranked with.
                images.append(Image.new("RGB", reference.image.size, (255, 255, 255)))
        votes = self._distances(reference, images)

        if len(images) == 1:
            # Nothing to compare against, so there is no vote to take; the
            # panel's mean distance is the most that can be said.
            return [sum(v[0] for v in votes) / len(votes)]

        count = len(candidate_pngs)
        wins = [0] * count
        for i in range(count):
            for j in range(i + 1, count):
                ahead = sum(1 for member in votes if member[i] < member[j])
                behind = sum(1 for member in votes if member[j] < member[i])
                if ahead > behind:
                    wins[i] += 1
                    wins[j] -= 1
                elif behind > ahead:
                    wins[j] += 1
                    wins[i] -= 1

        # Wins run from -(n-1) to +(n-1); map to a distance in [0, 1].
        span = 2 * (count - 1)
        tie_break = self._mean_ranks(votes, count)
        # Wins are integers, so ties are common on a front of tens, and the
        # caller keeps only the best few as parents -- leaving ties to fall
        # through to pool order decides the next epoch arbitrarily. Mean rank
        # settles them: scale-free, every member counting equally, which is the
        # same reason the panel votes rather than averaging distances. Scaled
        # to a quarter of the gap between adjacent win counts, so it orders
        # within a tie and can never reorder across one.
        return [
            0.5 - (w / span) + (rank - 0.5) / (span * 4)
            for w, rank in zip(wins, tie_break, strict=True)
        ]

    @staticmethod
    def _mean_ranks(votes: list[list[float]], count: int) -> list[float]:
        """Each candidate's average position across members, in [0, 1]."""
        totals = [0.0] * count
        for member in votes:
            order = sorted(range(count), key=lambda i: member[i])
            for position, index in enumerate(order):
                totals[index] += position / max(count - 1, 1)
        return [total / len(votes) for total in totals]
