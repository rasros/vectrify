"""Generate and Improve with a multimodal LLM.

Generate asks the model to draw the reference region from scratch; its SVG is
placed like any other generated trace. Improve (LLM editing) sends the current
drawing, a render of it and an instruction, then replays the reply leniently:
the prompt names the objects that may change, but the transaction is what
enforces it, and changes outside the selection or permissions are left out and
counted. Each reply becomes a proposal ranked by reference error.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import ClassVar

from vectrify.document import DocumentError, UnsupportedSvgError
from vectrify.image_utils import png_url, preview_urls, resize_long_side
from vectrify.llm.models import DEFAULT_MODELS, PROVIDERS, resolve_provider
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
from vectrify.operations.generate import (
    error,
    generated_result,
    render_region,
    target_region,
    validate_generate,
)
from vectrify.operations.settings import Setting, read_settings
from vectrify.svg.ownership import invisible_descriptions
from vectrify.svg.replies import apply_edits, extract_svg

log = logging.getLogger(__name__)
SVG_NS = "http://www.w3.org/2000/svg"

COMMON = {
    "provider": Setting(str, "auto", choices=("auto", *PROVIDERS)),
    "model": Setting(str, "", label="model"),
    "reasoning": Setting(str, "medium", choices=("low", "medium", "high")),
    "candidates": Setting(int, 1, minimum=1, maximum=4),
    "resolution": Setting(int, 512, minimum=128, maximum=2048, label="resolution"),
}
GENERATE = {**COMMON, "instruction": Setting(str, "", label="instruction")}
EDIT = {**COMMON, "instruction": Setting(str, "", label="instruction")}


def _client(settings):
    from vectrify.llm import LLMConfig, get_provider

    try:
        provider, key = resolve_provider(settings["provider"])
    except ValueError as exc:
        raise DocumentError(str(exc)) from exc
    model = settings["model"] or DEFAULT_MODELS[provider]
    return get_provider(provider, key), LLMConfig(
        model=model, reasoning=settings["reasoning"]
    )


def to_canvas(svg: str, size: tuple[int, int]) -> str:
    """Rescale a reply whose viewBox differs from the canvas it was asked for."""
    root = ET.fromstring(svg)
    width, height = size
    viewbox = root.get("viewBox")
    if viewbox:
        x, y, w, h = (float(v) for v in viewbox.replace(",", " ").split())
    else:
        x, y = 0.0, 0.0
        w = float((root.get("width") or str(width)).rstrip("px"))
        h = float((root.get("height") or str(height)).rstrip("px"))
    if (x, y, w, h) == (0, 0, width, height):
        return svg
    if w <= 0 or h <= 0:
        raise DocumentError("The model's SVG has an empty viewBox")
    sx, sy = width / w, height / h
    group = ET.Element(
        f"{{{SVG_NS}}}g",
        {"transform": f"matrix({sx!r} 0 0 {sy!r} {-x * sx!r} {-y * sy!r})"},
    )
    for child in list(root):
        root.remove(child)
        group.append(child)
    root.append(group)
    root.set("viewBox", f"0 0 {width} {height}")
    ET.register_namespace("", SVG_NS)
    return ET.tostring(root, encoding="unicode")


def _ask(client, config, prompt, context, count, label):
    replies = []
    for index in range(count):
        if context.stop.is_set():
            break
        context.progress(index, f"{label} {index + 1} of {count}…", total=count)
        replies.append(client.generate(prompt, config))
    return replies


def _ranked(proposals: list[Proposal]) -> OperationResult:
    proposals.sort(key=lambda p: (not p.changed, p.metrics["after"]["error"]))
    return OperationResult(proposals[0], proposals[1:])


class LlmGenerate:
    action: ClassVar[str] = "generate"
    name: ClassVar[str] = "llm"
    background: ClassVar[bool] = True
    needs_reference: ClassVar[bool] = True
    resources: ClassVar[frozenset[str]] = frozenset()

    def validate(self, request: OperationRequest) -> None:
        read_settings(request.settings, GENERATE, "LLM")
        validate_generate(request)

    def run(self, request: OperationRequest, context: RunContext) -> OperationResult:
        from vectrify.svg.prompts import build_svg_gen_prompt

        settings = read_settings(request.settings, GENERATE, "LLM")
        client, config = _client(settings)
        region = target_region(request)
        canvas = region.image.size
        prompt = build_svg_gen_prompt(
            png_url(resize_long_side(region.image, settings["resolution"])),
            1,
            goal=settings["instruction"] or None,
            canvas=canvas,
            source_name=request.source_name,
        )
        replies = _ask(
            client, config, prompt, context, settings["candidates"], "Drawing"
        )
        proposals, problems = [], []
        for raw in replies:
            try:
                svg = to_canvas(extract_svg(raw), canvas)
                result = generated_result(
                    request, svg, region, label="Generate with LLM", name="LLM drawing"
                )
            except (DocumentError, UnsupportedSvgError, ValueError) as exc:
                problems.append(str(exc))
                continue
            proposals.append(result.recommended)
        if not proposals:
            raise DocumentError(
                "The model's drawing could not be used: "
                + ("; ".join(problems) or "no reply")
            )
        context.progress(len(replies), "Preview ready")
        return _ranked(proposals)


class LlmEdit:
    action: ClassVar[str] = "improve"
    name: ClassVar[str] = "llm"
    background: ClassVar[bool] = True
    needs_reference: ClassVar[bool] = True
    resources: ClassVar[frozenset[str]] = frozenset()

    def validate(self, request: OperationRequest) -> None:
        settings = read_settings(request.settings, EDIT, "LLM")
        if not settings["instruction"].strip():
            raise DocumentError("Describe the change you want")
        if len(settings["instruction"]) > 2000:
            raise DocumentError("Keep the instruction under 2000 characters")
        mutation_scope(request)
        target_region(request)

    def run(self, request: OperationRequest, context: RunContext) -> OperationResult:
        from vectrify.svg.normalize import normalize_svg
        from vectrify.svg.prompts import build_svg_gen_prompt

        settings = read_settings(request.settings, EDIT, "LLM")
        client, config = _client(settings)
        region = target_region(request)
        scope = mutation_scope(request)
        size = region.image.size
        svg, root_attributes = region_svg(request, region, size)
        before_image = render_region(request.snapshot.document, region)
        whole = request.snapshot.selection.whole_document
        goal = settings["instruction"].strip() + (
            ""
            if whole
            else "\nOnly change these elements (by id) and what is inside them: "
            + ", ".join(sorted(scope.object_ids))
            + ". Keep every other element exactly as it is, with its id."
        )
        prompt = build_svg_gen_prompt(
            png_url(resize_long_side(region.image, settings["resolution"])),
            1,
            svg_prev=svg,
            rasterized_svg_data_url=png_url(
                resize_long_side(before_image, settings["resolution"])
            ),
            goal=goal,
            canvas=(round(region.width), round(region.height)),
            source_name=request.source_name,
            invisible=invisible_descriptions(svg),
        )
        replies = _ask(client, config, prompt, context, settings["candidates"], "Edit")
        target = region.image.convert("RGB")
        before = error(before_image, target)
        proposals, problems = [], []
        for raw in replies:
            try:
                candidates = apply_edits(svg, raw)
            except Exception as exc:
                problems.append(str(exc))
                continue
            for candidate in candidates:
                tx = request.transaction("Edit with LLM")
                try:
                    outcome = replay(
                        tx,
                        restore_root(candidate, root_attributes),
                        lenient=True,
                        baseline=restore_root(normalize_svg(svg), root_attributes),
                    )
                except DocumentError as exc:
                    problems.append(str(exc))
                    continue
                after_image = render_region(tx.preview, region)
                proposals.append(
                    Proposal(
                        tx,
                        outcome.edits > 0,
                        metrics={
                            "before": {"error": before},
                            "after": {"error": error(after_image, target)},
                            "edits": outcome.edits,
                            "skipped": outcome.skipped,
                        },
                        previews=preview_urls(region.image, before_image, after_image),
                    )
                )
        if not proposals:
            raise DocumentError(
                "The model's edit could not be applied: "
                + ("; ".join(problems) or "no reply")
            )
        context.progress(len(replies), "Preview ready")
        return _ranked(proposals)


register(LlmGenerate())
register(LlmEdit())
