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

The members are five cheap-to-medium encoders, chosen for spread across
training regimes as much as for their NIGHTS placing:

    facebook/dinov2-small                        0.2599 (5th)   self-supervised
    laion/CLIP-ViT-B-32-laion2B-s34B-b79K        0.2579 (9th)   LAION-2B
    google/siglip-base-patch16-224               0.2575 (10th)  sigmoid pairwise
    laion/CLIP-ViT-L-14-laion2B-s32B-b82K        0.2569 (12th)  LAION-2B
    openai/clip-vit-base-patch32                 0.2305 (21st)  WIT-400M

Two members of one family vote together and cost the panel a voice, so a second
DINOv2 was passed over for a model trained on a different corpus even though it
places lower: a vote is worth having only from a member that can disagree.
Several better-placed candidates cannot be used -- the DataComp checkpoints
ship open_clip weights with no HuggingFace preprocessor config, ebind and
jina-omni need custom code, and the BLIP retrieval checkpoints load through
AutoModel with a randomly initialised visual_projection, which is the very
layer their image embedding comes from.

The two cheap pixel measures the round score uses, edge overlap and colour
distance, are deliberately not members. On the same hand-judged pairs they
agree with the human 40% and 44% of the time, the bottom of the field, and they
are already two of the three objectives the search optimises -- seating them
here would make the evaluator agree with the round score by construction.

Five is odd on purpose, so a pairwise majority always exists.
"""

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
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "openai/clip-vit-base-patch32",
)


@dataclass
class PanelReference:
    image: Image.Image
    references: list[Any]


class EnsembleScorer(Scorer):
    """Ranks by majority vote across a panel of encoders."""

    def __init__(self, model_names: tuple[str, ...] = PANEL_MODELS):
        self._members = [EmbeddingScorer(model_name=name) for name in model_names]
        self._names = model_names

    def validate_environment(self) -> None:
        self._members[0].validate_environment()

    def prepare_reference(self, original_rgb: Image.Image) -> PanelReference:
        return PanelReference(
            image=original_rgb,
            references=[m.prepare_reference(original_rgb) for m in self._members],
        )

    def score(self, reference: PanelReference, candidate_png: bytes) -> float:
        """Mean distance across the panel.

        A single candidate has nothing to be compared against, so there is no
        vote to take. This exists for callers that need a scalar per candidate;
        the panel's actual judgement is ``rank``, which is what decides
        direction.
        """
        values = [
            member.score(ref, candidate_png)
            for member, ref in zip(self._members, reference.references, strict=True)
        ]
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
        if len(candidate_pngs) == 1:
            return [self.score(reference, candidate_pngs[0])]

        # Each member scores the whole field once, batched.
        votes = [
            member.score_many(ref, candidate_pngs)
            for member, ref in zip(self._members, reference.references, strict=True)
        ]

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
