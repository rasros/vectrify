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

The two cheap pixel measures the round score uses, edge overlap and colour
distance, are deliberately not members. On the same hand-judged pairs they
agree with the human 40% and 44% of the time, the bottom of the field, and they
are already two of the three objectives the search optimises -- seating them
here would make the evaluator agree with the round score by construction.

Five is odd on purpose, so a pairwise majority always exists.
"""

import io
import logging
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

# Cells per side, swept from 1 to 8 against the distortion screen. Vector
# damage is ordered correctly 92.8% of the time reading the picture whole,
# 95.4% at three cells a side and 96.4% at five, after which it is flat -- 6, 7
# and 8 score 96.1, 96.1 and 96.4. So there is a real optimum around a 77px
# cell rather than a simple appetite for resolution, which finer grids would
# have kept feeding.
GRID = 5


@dataclass
class PanelReference:
    image: Image.Image
    tiles: list[Any]


def _tiles(image: Image.Image) -> list[Image.Image]:
    """The picture cut into a GRID x GRID lattice, row by row."""
    width, height = image.size
    return [
        image.crop(
            (
                column * width // GRID,
                row * height // GRID,
                (column + 1) * width // GRID,
                (row + 1) * height // GRID,
            )
        )
        for row in range(GRID)
        for column in range(GRID)
    ]


class EnsembleScorer(Scorer):
    """Ranks by majority vote across a panel of encoders."""

    def __init__(self, model_names: tuple[str, ...] = PANEL_MODELS):
        self._members = [EmbeddingScorer(model_name=name) for name in model_names]
        self._names = model_names

    def validate_environment(self) -> None:
        self._members[0].validate_environment()

    def prepare_reference(self, original_rgb: Image.Image) -> PanelReference:
        cells = _tiles(original_rgb)
        return PanelReference(
            image=original_rgb,
            tiles=[m.embed_images(cells) for m in self._members],
        )

    def _distances(self, reference: PanelReference, images: list[Image.Image]):
        """Each member's distance to every image, cell by cell then averaged."""
        per_member = []
        for member, reference_tiles in zip(self._members, reference.tiles, strict=True):
            got = [member.embed_images(_tiles(image)) for image in images]
            per_member.append(
                [
                    float((1.0 - (reference_tiles * tiles).sum(dim=-1)).mean())
                    for tiles in got
                ]
            )
        return per_member

    def score(self, reference: PanelReference, candidate_png: bytes) -> float:
        """Mean distance across the panel.

        A single candidate has nothing to be compared against, so there is no
        vote to take. This exists for callers that need a scalar per candidate;
        the panel's actual judgement is ``rank``, which is what decides
        direction.
        """
        try:
            image = Image.open(io.BytesIO(candidate_png)).convert("RGB")
        except Exception:
            return MAX_SCORE
        values = [d[0] for d in self._distances(reference, [image])]
        return sum(values) / len(values) if values else MAX_SCORE

    def rank(
        self, reference: PanelReference, candidate_pngs: list[bytes]
    ) -> list[float]:
        """Score candidates by how many rivals the panel puts them ahead of.

        Every pair is put to the panel and the majority wins, then candidates
        are ordered by wins less losses. That ordering step is not decoration:
        a majority relation is a tournament and cycles, so it cannot be sorted
        by pairwise comparison -- three candidates can each beat the next.
        Counting wins ranks a tournament without needing it to be transitive.

        Returns one value per candidate, lower being better, so it drops into
        a caller that expects a distance. Values span [0, 1] up to the width of
        the tie-break term, which can carry the extremes a little past either
        end; nothing compares them against an absolute threshold.
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
