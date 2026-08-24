import logging
import random
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from typing import Any

from vectrify.formats.base import (
    NoUsableOutputError,
    apply_search_replace,
    describe_unusable,
    split_alternatives,
)
from vectrify.formats.mutations import operator_weights
from vectrify.formats.svg.normalize import normalize_svg
from vectrify.formats.svg.operations import (
    MUTATIONS,
    apply_crossover,
    apply_mutation,
)
from vectrify.formats.svg.ownership import describe_invisible, invisible_elements
from vectrify.formats.svg.prompts import (
    build_svg_gen_prompt,
    extract_svg_fragment,
    is_valid_svg,
)
from vectrify.formats.svg.targets import element_targets
from vectrify.image_utils import rasterize_svg_to_png_bytes
from vectrify.refine.paths import (
    PATH_FIT,
    UnsupportedPathError,
    fit_available,
    fit_opaque_fills_locally,
    fit_random_group,
    fittable_opaque_fills,
)

log = logging.getLogger(__name__)


class SvgPlugin:
    name = "svg"
    file_extension = ".svg"
    # The path fitter is the only mutation that may allocate CUDA tensors.
    # Workers use this marker to join the runner-wide GPU admission gate.
    gpu_bound_mutation = True
    gpu_mutation_operator = PATH_FIT
    gpu_gate: Any = None
    # A fit costs about 0.5s on a GPU where an ordinary mutation costs about a
    # millisecond, so it opens on a small share of the draws and the policy
    # moves it from there on what it actually returns.
    PATH_FIT_WEIGHT = 0.03
    # Halved: a draw of the fit is 0.5s of GPU against ~1ms for the others.
    PATH_FIT_REWARD_SCALE = 0.5

    def rasterize(self, content: str, out_w: int, out_h: int) -> bytes:
        return rasterize_svg_to_png_bytes(content, out_w=out_w, out_h=out_h)

    def validate(self, content: str) -> tuple[bool, str | None]:
        return is_valid_svg(content)

    @staticmethod
    def _require_svg(fragment: str, raw: str) -> str:
        if "<svg" not in fragment.lower():
            raise NoUsableOutputError(
                f"no <svg> in the reply and no diff blocks: {describe_unusable(raw)}"
            )
        return fragment

    def extract_from_llm(self, raw: str) -> str:
        # Normalised on the way in, so local search meets one form of markup
        # rather than whichever the model reached for. Which forms it reaches
        # for is a property of the model: one model's seeds carried 147
        # elements in relative path commands, which describe an offset from
        # wherever the pen already is and so cannot be moved at all.
        return normalize_svg(self._require_svg(extract_svg_fragment(raw), raw))

    def apply_edit(self, parent: str, raw: str) -> str:
        patched = apply_search_replace(parent, raw)
        if patched is None:
            patched = self._require_svg(extract_svg_fragment(raw), raw)
        return normalize_svg(patched)

    def apply_edits(self, parent: str, raw: str) -> list[str]:
        """Every attempt the reply offers, each a candidate of its own.

        A section that will not apply is dropped rather than failing the rest:
        a reply offering three attempts should not be discarded because one of
        them misquoted the markup. If none apply, the single-edit path runs
        again so the caller sees the same error it always did.
        """
        candidates: list[str] = []
        for section in split_alternatives(raw):
            try:
                candidates.append(self.apply_edit(parent, section))
            except Exception as exc:
                log.debug(f"Dropping one alternative: {exc}")
        if not candidates:
            return [self.apply_edit(parent, raw)]
        return candidates

    def build_generate_prompt(
        self,
        image_data_url: str,
        node_index: int,
        content_prev: str | None,
        raster_preview_url: str | None,
        goal: str | None,
        canvas: tuple[int, int],
        source_name: str | None = None,
        invisible: list[str] | None = None,
    ) -> list[dict]:
        return build_svg_gen_prompt(
            image_data_url,
            node_index,
            svg_prev=content_prev,
            rasterized_svg_data_url=raster_preview_url if content_prev else None,
            goal=goal,
            canvas=canvas,
            source_name=source_name,
            invisible=invisible,
        )

    def mutation_weights(self) -> Mapping[str, float]:
        weights = dict(operator_weights(MUTATIONS))
        if fit_available():
            weights[PATH_FIT] = self.PATH_FIT_WEIGHT
        return weights

    def operator_reward_scale(self) -> Mapping[str, float]:
        """The path fit's reward, discounted for what a draw of it costs."""
        return {PATH_FIT: self.PATH_FIT_REWARD_SCALE}

    def mutate(
        self,
        content: str,
        operator: str | None = None,
        targets: dict[int, float] | None = None,
        reference_png: bytes | None = None,
    ) -> tuple[str, str]:
        """Apply one operator, which may be the target-aware path fit.

        The fit is dispatched here rather than from the shared mutation table
        because every entry in that table is a pure markup transform -- it never
        sees the picture -- and the fit needs the reference and a render of the
        rest of the drawing. Keeping it out of the table leaves that contract
        intact for the other two backends.
        """
        wants_fit = operator == PATH_FIT or (
            operator is None
            and reference_png is not None
            and fit_available()
            and random.random() < self.PATH_FIT_WEIGHT
        )
        if wants_fit:
            # Handing back the content unchanged is how an operator reports that
            # it found nothing to do: the worker recognises it and charges the
            # draw to this operator by name. Raising would need the exception
            # from vector.worker, which formats must not depend on.
            if reference_png is None or not fit_available():
                return content, PATH_FIT
            try:
                # SAMVG seeds are opaque closed fills, so they use the exact
                # analytic CUDA fitter.  The older sampled operator remains
                # the style-specific path for stroked cubic drawings.
                if fittable_opaque_fills(content):
                    return (
                        fit_opaque_fills_locally(
                            content,
                            reference_png,
                            gpu_gate=self.gpu_gate,
                        ),
                        PATH_FIT,
                    )
                return (
                    fit_random_group(
                        content,
                        reference_png,
                        rasterize=lambda svg, w, h: self.rasterize(svg, w, h),
                        weights=targets,
                        gpu_gate=self.gpu_gate,
                    ),
                    PATH_FIT,
                )
            except UnsupportedPathError as exc:
                log.debug(f"Nothing to fit: {exc}")
                return content, PATH_FIT
            except Exception as exc:
                # A device that has run out is a reason to skip the operator,
                # not to fail the task: one run turned an exhausted GPU into
                # thousands of failed candidates. Anything else is a real bug
                # and is left to propagate.
                message = str(exc)
                if "CUDA" in message or "out of memory" in message.lower():
                    log.debug(f"No room to fit: {exc}")
                    return content, PATH_FIT
                raise
        return apply_mutation(content, operator, targets)

    def element_targets(self, content: str, reference_png: bytes) -> dict[int, float]:
        return element_targets(content, reference_png)

    def invisible_elements(self, content: str) -> list[str]:
        """Elements that paint nothing, for the edit prompt to name.

        The model cannot see this: on screen the element simply is not there,
        and in the markup it looks like any other. An LLM edit is also the only
        operator that can fix it, since making an occluded element visible needs
        its order changed and its position moved in one move, and the mutation
        operators each do one of those.
        """
        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            return []
        return describe_invisible(root, invisible_elements(root))

    def crossover(self, content_a: str, content_b: str) -> tuple[str, str]:
        return apply_crossover(content_a, content_b)
