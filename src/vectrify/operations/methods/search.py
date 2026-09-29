"""Improve: hill climbing over the selected objects.

The search starts from the current drawing, mutates only the selected objects
with the operators the permissions allow, and keeps a change when the drawing
scores no worse against the reference in the target region. The score is the
simple scorer's blend of structure and colour -- plain colour error rewards
erasing small parts, which a drawing tool must not do. The best drawing it
saw is replayed as a transaction, so the proposal is an ordinary, lock- and
pin-checked edit. No LLM call and no gradient fitting is involved.
"""

from __future__ import annotations

import os
from typing import ClassVar

from vectrify.document import DocumentError
from vectrify.image_utils import png_bytes, preview_urls, resize_long_side
from vectrify.operations.candidates import (
    mutation_scope,
    region_svg,
    replay,
    restore_root,
)
from vectrify.operations.contract import (
    OperationRequest,
    OperationResult,
    Proposal,
    RunContext,
    register,
)
from vectrify.operations.generate import render_region, target_region
from vectrify.operations.settings import Setting, read_settings

DEFAULT_TASKS = 600
LABEL = "Search improvements"

SETTINGS = {
    "workers": Setting(int, 2, minimum=1, maximum=max(1, os.cpu_count() or 1)),
    "resolution": Setting(int, 256, minimum=64, maximum=1024, label="resolution"),
    "adaptive_operators": Setting(bool, True),
}


class Search:
    action: ClassVar[str] = "improve"
    name: ClassVar[str] = "search"
    background: ClassVar[bool] = True
    needs_reference: ClassVar[bool] = True
    resources: ClassVar[frozenset[str]] = frozenset()

    def validate(self, request: OperationRequest) -> None:
        read_settings(request.settings, SETTINGS, "search")
        if request.reference is None:
            raise DocumentError("Add a reference image to improve against")
        scope = mutation_scope(request)
        if not scope.kinds & {"geometry", "paint", "structure"}:
            raise DocumentError("Allow geometry, paint or stacking changes")
        target_region(request)

    def run(self, request: OperationRequest, context: RunContext) -> OperationResult:
        from vectrify.score.simple import SimpleFallbackScorer
        from vectrify.vector.search import SearchSettings, run_search
        from vectrify.vector.worker import WorkerContext

        settings = read_settings(request.settings, SETTINGS, "search")
        tasks = request.budget.steps or DEFAULT_TASKS
        region = target_region(request)
        target = resize_long_side(region.image, settings["resolution"])
        size = target.size
        context.progress(0, "Starting the workers…", total=tasks)
        svg, root_attributes = region_svg(request, region, size)
        scorer = SimpleFallbackScorer()
        reference = scorer.prepare_reference(target)

        def score(png: bytes) -> float:
            return scorer.score(reference, png)

        outcome = run_search(
            [svg],
            score,
            WorkerContext(
                scope=mutation_scope(request),
                original_png_bytes=png_bytes(target),
                original_w=size[0],
                original_h=size[1],
            ),
            SearchSettings(
                workers=settings["workers"],
                adaptive_operators=settings["adaptive_operators"],
                max_total_tasks=tasks,
                max_wall_seconds=request.budget.seconds,
                # One result is proposed. The runners-up are one climb's
                # previous steps, so they only stand in if it fails replay.
                keep=4,
            ),
            stop=context.stop,
            progress=lambda p: context.progress(
                p.tasks_completed,
                f"Searching · {p.tasks_completed:,}/{tasks:,} variants",
            ),
        )
        before = outcome.start.score
        proposals: list[Proposal] = []
        rejected = 0
        for candidate in outcome.ranked:
            if candidate.content == svg:
                continue
            if candidate.score > before or proposals:
                break
            tx = request.transaction(LABEL)
            try:
                restored = restore_root(candidate.content, root_attributes)
                edits = replay(tx, restored).edits
            except DocumentError:
                rejected += 1
                continue
            if not edits:
                continue
            proposals.append(
                Proposal(
                    tx,
                    candidate.score < before,
                    metrics={
                        "before": {"difference": before},
                        "after": {"difference": candidate.score},
                        "edits": edits,
                        "tasks": outcome.tasks_completed,
                        "accepted": outcome.accepted,
                    },
                    label=f"Difference {candidate.score:.4f}",
                )
            )
        if not proposals or not proposals[0].changed:
            proposals.insert(
                0,
                Proposal(
                    request.transaction(LABEL),
                    False,
                    metrics={
                        "before": {"difference": before},
                        "after": {"difference": before},
                        "tasks": outcome.tasks_completed,
                        "rejected": rejected,
                    },
                ),
            )
        previous = render_region(request.snapshot.document, region)
        for proposal in proposals:
            proposal.previews = preview_urls(
                region.image,
                previous,
                render_region(proposal.transaction.preview, region),
            )
        return OperationResult(
            proposals[0],
            proposals[1:],
            message=None
            if proposals[0].changed
            else "No variant beat the current drawing",
        )


register(Search())
