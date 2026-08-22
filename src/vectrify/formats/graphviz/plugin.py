import io
import logging
import re
from collections.abc import Mapping

from vectrify.formats.base import BaseFormatPlugin
from vectrify.formats.graphviz.operations import (
    MUTATIONS,
    apply_crossover,
    apply_mutation,
)
from vectrify.formats.graphviz.prompts import build_dot_gen_prompt
from vectrify.formats.mutations import operator_weights

log = logging.getLogger(__name__)

_DOT_FENCE = re.compile(r"```dot\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_DOT_RAW = re.compile(
    r'(strict\s+)?(di)?graph\s+(?:\w+|"[^"]*")?\s*\{.*\}',
    re.DOTALL | re.IGNORECASE,
)

_HTML_TAGS = re.compile(r"<[^>]*>")
# Matches an attribute value like =<TAG ...>content</TAG> (paired open/close tag).
# LLMs often emit this when they mean =<<TAG>content</TAG>> (the valid DOT form).
_PAIRED_TAG_LABEL = re.compile(
    r"=\s*<([A-Za-z][^>/\s]*)(?:[^>]*)>(.*?)</\1\s*>",
    re.DOTALL | re.IGNORECASE,
)


def _plain_from_html_label(content: str) -> str:
    """Strip HTML tags from DOT HTML-label content, return a plain string."""
    plain = _HTML_TAGS.sub(" ", content)
    plain = " ".join(plain.split())
    return plain.replace('"', "'")


def _fix_html_labels(dot: str) -> str:
    """Convert malformed HTML attribute values (=<...>) to quoted plain strings.

    DOT HTML labels require =<<TAG>...</TAG>> (double angle brackets at the
    boundary). LLMs often emit =<TAG>...</TAG> (single angle bracket), which
    makes the DOT parser choke on every > inside the HTML. This function
    detects those attribute values, strips the HTML tags, and re-emits them
    as plain quoted strings. Properly doubled HTML labels (=<<...>>) are left untouched.
    """

    def _strip_paired(m: re.Match) -> str:
        inner = _plain_from_html_label(m.group(2))
        return f'="{inner}"'

    dot = _PAIRED_TAG_LABEL.sub(_strip_paired, dot)

    # Handle remaining single-level HTML labels using depth-tracking
    out: list[str] = []
    i = 0
    n = len(dot)
    while i < n:
        if dot[i] != "=":
            out.append(dot[i])
            i += 1
            continue

        j = i + 1
        while j < n and dot[j] in " \t\n\r":
            j += 1

        if j >= n or dot[j] != "<":
            out.append(dot[i])
            i += 1
            continue

        if j + 1 < n and dot[j + 1] == "<":
            out.append(dot[i])
            i += 1
            continue

        depth = 0
        k = j
        while k < n:
            if dot[k] == "<":
                depth += 1
            elif dot[k] == ">":
                depth -= 1
                if depth == 0:
                    html_content = dot[j + 1 : k]
                    plain = _plain_from_html_label(html_content)
                    out.append(f'="{plain}"')
                    i = k + 1
                    break
            k += 1
        else:
            out.append(dot[i])
            i += 1

    return "".join(out)


def _sanitize_dot(dot: str) -> str:
    """Fix common LLM-generated DOT mistakes before validation."""
    if "->" in dot and not re.search(r"\bdigraph\b", dot, re.IGNORECASE):
        dot = re.sub(r"\bgraph\b", "digraph", dot, count=1, flags=re.IGNORECASE)
    return _fix_html_labels(dot)


class GraphvizPlugin(BaseFormatPlugin):
    name = "graphviz"
    file_extension = ".dot"

    def _render_png(self, content: str) -> bytes:
        import graphviz

        # Select the engine explicitly.  DOT's `layout=` graph attribute is a
        # request, not a reliable renderer selection, and letting the wrapper
        # choose makes identical diagrams differ across callers.
        return graphviz.Source(content, engine="dot").pipe(format="png", quiet=True)

    def _compile(self, content: str) -> None:
        import graphviz

        graphviz.Source(content, engine="dot").pipe(format="svg", quiet=True)

    def rasterize(self, content: str, out_w: int, out_h: int) -> bytes:
        """Fit a layout-managed diagram onto the scoring canvas without warp.

        Graphviz computes a graph's natural aspect ratio.  Stretching that PNG
        to a target canvas changes circles to ellipses and makes pixel scoring
        reward distortion.  Keep that geometry and composite on white instead.
        """
        if out_w <= 0 or out_h <= 0:
            raise ValueError(f"Invalid raster target size: {out_w}x{out_h}")
        from PIL import Image
        from PIL.Image import Resampling

        rendered = Image.open(io.BytesIO(self._render_png(content))).convert("RGBA")
        scale = min(out_w / rendered.width, out_h / rendered.height)
        size = (
            max(1, round(rendered.width * scale)),
            max(1, round(rendered.height * scale)),
        )
        rendered = rendered.resize(size, Resampling.LANCZOS)
        canvas = Image.new("RGBA", (out_w, out_h), "white")
        canvas.alpha_composite(
            rendered, ((out_w - size[0]) // 2, (out_h - size[1]) // 2)
        )
        out = io.BytesIO()
        canvas.convert("RGB").save(out, format="PNG")
        return out.getvalue()

    def extract_from_llm(self, raw: str) -> str:
        m = _DOT_FENCE.search(raw)
        if m:
            return _sanitize_dot(m.group(1).strip())
        m = _DOT_RAW.search(raw)
        if m:
            return _sanitize_dot(m.group(0).strip())
        return _sanitize_dot(raw.strip())

    def build_generate_prompt(
        self,
        image_data_url: str,
        node_index: int,
        content_prev: str | None,
        raster_preview_url: str | None,
        goal: str | None,
        # DOT positions are computed by the layout engine, not written in the
        # source, so there is no coordinate space to pin and grafting between
        # parents cannot misplace anything.
        canvas: tuple[int, int],  # noqa: ARG002 - layout engine owns positions
        source_name: str | None = None,
        # Accepted and ignored: reporting elements that paint nothing needs a
        # renderer that can draw one element at a time, which these backends
        # hand off to an external tool. Same limit as `element_targets`.
        invisible: list[str] | None = None,  # noqa: ARG002
    ) -> list[dict]:
        return build_dot_gen_prompt(
            image_data_url=image_data_url,
            node_index=node_index,
            source_name=source_name,
            dot_prev=content_prev,
            rasterized_dot_data_url=raster_preview_url,
            goal=goal,
        )

    def mutation_weights(self) -> Mapping[str, float]:
        return operator_weights(MUTATIONS)

    def mutate(
        self,
        content: str,
        operator: str | None = None,
        targets: dict[int, float] | None = None,
        reference_png: bytes | None = None,
    ) -> tuple[str, str]:
        _ = targets, reference_png
        return apply_mutation(content, operator)

    def crossover(self, content_a: str, content_b: str) -> tuple[str, str]:
        return apply_crossover(content_a, content_b)
