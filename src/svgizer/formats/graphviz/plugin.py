from __future__ import annotations

import io
import logging
import re

import PIL.Image

from svgizer.formats.graphviz.operations import (
    crossover_with_micro_search,
    mutate_with_micro_search,
)
from svgizer.formats.graphviz.prompts import (
    build_dot_gen_prompt,
    build_dot_summarize_prompt,
)

log = logging.getLogger(__name__)

_DOT_FENCE = re.compile(r"```dot\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
# Graph name may be unquoted (\w+), quoted ("..."), or absent.
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
    boundary).  LLMs often emit =<TAG>...</TAG> (single angle bracket), which
    makes the DOT parser choke on every > inside the HTML.  This function
    detects those attribute values, strips the HTML tags, and re-emits them
    as plain quoted strings.

    Properly doubled HTML labels (=<<...>>) are left untouched.
    """

    # Pass 1: handle paired open/close HTML tags, e.g. =<B>text</B>.
    # These are the most common LLM mistake and can be caught with a regex.
    def _strip_paired(m: re.Match) -> str:
        inner = _plain_from_html_label(m.group(2))
        return f'="{inner}"'

    dot = _PAIRED_TAG_LABEL.sub(_strip_paired, dot)

    # Pass 2: handle remaining single-level HTML labels, e.g. =<plain text>,
    # using a depth-tracking scanner so we respect < > nesting correctly.
    out: list[str] = []
    i = 0
    n = len(dot)
    while i < n:
        if dot[i] != "=":
            out.append(dot[i])
            i += 1
            continue

        # Peek past optional whitespace to find '<'
        j = i + 1
        while j < n and dot[j] in " \t\n\r":
            j += 1

        if j >= n or dot[j] != "<":
            out.append(dot[i])
            i += 1
            continue

        # Leave the valid doubled form =<<...>> untouched.
        if j + 1 < n and dot[j + 1] == "<":
            out.append(dot[i])
            i += 1
            continue

        # Find the closing '>' tracking nesting depth.
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
    # Upgrade undirected `graph` to `digraph` when directed edges are present.
    if "->" in dot and not re.search(r"\bdigraph\b", dot, re.IGNORECASE):
        dot = re.sub(r"\bgraph\b", "digraph", dot, count=1, flags=re.IGNORECASE)
    # Convert malformed HTML labels to plain quoted strings.
    return _fix_html_labels(dot)


class GraphvizPlugin:
    name = "graphviz"
    file_extension = ".dot"

    def rasterize(self, content: str, out_w: int, out_h: int) -> bytes:
        import graphviz

        src = graphviz.Source(content)
        png = src.pipe(format="png", quiet=True)
        # Resize to exact dimensions
        img = PIL.Image.open(io.BytesIO(png)).convert("RGB")
        img = img.resize((out_w, out_h), PIL.Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def rasterize_fast(self, content: str, long_side: int) -> bytes | None:
        try:
            import graphviz

            from svgizer.image_utils import resize_long_side

            src = graphviz.Source(content)
            png = src.pipe(format="png", quiet=True)
            img = PIL.Image.open(io.BytesIO(png)).convert("RGB")
            img = resize_long_side(img, long_side)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return None

    def validate(self, content: str) -> tuple[bool, str | None]:
        try:
            import graphviz

            graphviz.Source(content).pipe(format="svg", quiet=True)
            return True, None
        except Exception as e:
            return False, str(e)

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
        change_summary: str | None,
        diff_data_url: str | None,
    ) -> list[dict]:
        return build_dot_gen_prompt(
            image_data_url=image_data_url,
            node_index=node_index,
            dot_prev=content_prev,
            rasterized_dot_data_url=raster_preview_url,
            change_summary=change_summary,
            diff_data_url=diff_data_url,
        )

    def build_summarize_prompt(
        self,
        image_data_url: str,
        raster_preview_url: str | None,
        custom_goal: str | None,
        previous_summary: str | None,
    ) -> list[dict]:
        return build_dot_summarize_prompt(
            image_data_url=image_data_url,
            raster_preview_url=raster_preview_url,
            custom_goal=custom_goal,
            previous_summary=previous_summary,
        )

    def mutate(self, content: str, orig_img_fast: PIL.Image.Image) -> tuple[str, str]:
        return mutate_with_micro_search(
            parent_dot=content, orig_img_fast=orig_img_fast, num_trials=15
        )

    def crossover(
        self, content_a: str, content_b: str, orig_img_fast: PIL.Image.Image
    ) -> tuple[str, str]:
        return crossover_with_micro_search(
            dot_a=content_a,
            dot_b=content_b,
            orig_img_fast=orig_img_fast,
            num_trials=15,
        )
