"""Improve: NSGA-II local search over the selected objects.

The search starts from the current drawing, mutates only the selected objects
with the operators the permissions allow, and trades the pixel objectives off
by Pareto dominance. The final pool is then ranked by one explicit policy --
mean squared error against the reference in the target region -- and the best
candidates are replayed as transactions, so each proposal is an ordinary,
lock- and pin-checked edit. No LLM call and no gradient fitting is involved.
"""

from __future__ import annotations

import io
import os
from typing import ClassVar

from PIL import Image

from vectrify.document import DocumentError
from vectrify.image_utils import (
    preview_urls,
    rasterize_svg_to_png_bytes,
    resize_long_side,
)
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
from vectrify.operations.generate import error, render_region, target_region
from vectrify.operations.settings import Setting, read_settings

DEFAULT_TASKS = 600
LABEL = "Improve with NSGA-II"

SETTINGS = {
    "workers": Setting(int, 2, minimum=1, maximum=max(1, os.cpu_count() or 1)),
    "pool_size": Setting(int, 12, minimum=2, maximum=64, label="pool size"),
    "resolution": Setting(int, 256, minimum=64, maximum=1024, label="resolution"),
    "adaptive_operators": Setting(bool, True),
    "alternatives": Setting(int, 3, minimum=0, maximum=8),
}


class Nsga:
    action: ClassVar[str] = "improve"
    name: ClassVar[str] = "nsga"
    background: ClassVar[bool] = True
    needs_reference: ClassVar[bool] = True
    resources: ClassVar[frozenset[str]] = frozenset()

    def validate(self, request: OperationRequest) -> None:
        read_settings(request.settings, SETTINGS, "NSGA-II")
        if request.reference is None:
            raise DocumentError("Add a reference image to improve against")
        scope = mutation_scope(request)
        if not scope.kinds & {"geometry", "paint", "structure"}:
            raise DocumentError("Allow geometry, paint or stacking changes")
        target_region(request)

    def run(self, request: OperationRequest, context: RunContext) -> OperationResult:
        from vectrify.vector.reference import Reference
        from vectrify.vector.search import SearchSettings, run_search, seed_node
        from vectrify.vector.worker import WorkerContext

        settings = read_settings(request.settings, SETTINGS, "NSGA-II")
        tasks = request.budget.steps or DEFAULT_TASKS
        region = target_region(request)
        target = resize_long_side(region.image, settings["resolution"])
        size = target.size
        context.progress(0, "Preparing the reference and workers…", total=tasks)
        svg, root_attributes = region_svg(request, region, size)
        reference = Reference.build(target, score_resolution=max(size), segment_count=4)
        scope = mutation_scope(request)

        def render(content: str) -> Image.Image:
            png = rasterize_svg_to_png_bytes(content, out_w=size[0], out_h=size[1])
            with Image.open(io.BytesIO(png)) as image:
                return image.convert("RGB")

        seed_png = rasterize_svg_to_png_bytes(svg, out_w=size[0], out_h=size[1])
        seed = seed_node(
            reference,
            svg,
            seed_png,
            node_id=1,
            origin="Current drawing",
        )
        worker_context = WorkerContext(
            scope=scope,
            original_png_bytes=reference.png,
            original_w=size[0],
            original_h=size[1],
            log_level="WARNING",
        )
        outcome = run_search(
            reference,
            [seed],
            worker_context,
            SearchSettings(
                workers=settings["workers"],
                pool_size=settings["pool_size"],
                adaptive_operators=settings["adaptive_operators"],
                epochs=1,
                max_total_tasks=tasks,
                max_wall_seconds=request.budget.seconds,
            ),
            stop=context.stop,
            progress=lambda p: context.progress(
                p.tasks_completed,
                f"NSGA-II · {p.tasks_completed:,}/{tasks:,} candidates",
            ),
        )
        context.progress(context.step, "Ranking the final pool…")

        # The explicit ranking policy: pixel MSE against the reference region.
        target_rgb = target.convert("RGB")
        before = error(render(svg), target_rgb)
        ranked = sorted(
            (
                (error(render(content), target_rgb), content)
                for content in dict.fromkeys(
                    n.state.payload.content
                    for n in outcome.pool
                    if n.state.payload.content and n.state.payload.content != svg
                )
            ),
            key=lambda pair: pair[0],
        )
        proposals: list[Proposal] = []
        rejected = 0
        for score, content in ranked:
            if score >= before and proposals:
                break
            tx = request.transaction(LABEL)
            try:
                edits = replay(tx, restore_root(content, root_attributes)).edits
            except DocumentError:
                rejected += 1
                continue
            if not edits:
                continue
            proposals.append(
                Proposal(
                    tx,
                    score < before,
                    metrics={
                        "before": {"error": before},
                        "after": {"error": score},
                        "edits": edits,
                        "tasks": outcome.tasks_completed,
                    },
                    label=f"Error {score:.5f}",
                )
            )
            if len(proposals) > settings["alternatives"]:
                break
        if not proposals or not proposals[0].changed:
            proposals.insert(
                0,
                Proposal(
                    request.transaction(LABEL),
                    False,
                    metrics={
                        "before": {"error": before},
                        "after": {"error": before},
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
            else "No candidate beat the current drawing",
        )


register(Nsga())
