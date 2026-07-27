"""Common behavior of the code-format (graphviz/typst) generation prompts.

Format-specific assertions (syntax rules, fence language quirks) live in the
per-format test_prompts modules; everything both builders must do is here.
"""

from typing import NamedTuple

import pytest

from vectrify.formats.graphviz.prompts import build_dot_gen_prompt
from vectrify.formats.typst.prompts import build_typst_gen_prompt
from vectrify.tests.helpers import image_urls, text_blocks

_IMG_URL = "data:image/png;base64,abc"
_RENDER_URL = "data:image/png;base64,def"
_DIFF_URL = "data:image/png;base64,ghi"


def _build_dot(img, node_index, prev, render, goal, diff):
    return build_dot_gen_prompt(
        img,
        node_index=node_index,
        dot_prev=prev,
        rasterized_dot_data_url=render,
        goal=goal,
        diff_data_url=diff,
    )


def _build_typst(img, node_index, prev, render, goal, diff):
    return build_typst_gen_prompt(
        img,
        node_index=node_index,
        typst_prev=prev,
        rasterized_data_url=render,
        goal=goal,
        diff_data_url=diff,
    )


class Case(NamedTuple):
    build: object
    sample: str
    fence: str


CASES = [
    pytest.param(Case(_build_dot, "digraph G { A -> B }", "```dot"), id="dot"),
    pytest.param(Case(_build_typst, "#rect(width: 10pt)", "```typst"), id="typst"),
]


@pytest.mark.parametrize("case", CASES)
def test_first_iteration_asks_for_fenced_code(case):
    blocks = case.build(_IMG_URL, 1, None, None, None, None)
    text = "\n".join(text_blocks(blocks))
    assert "iteration #1" in text.lower()
    assert case.fence in text
    assert _IMG_URL in image_urls(blocks)


@pytest.mark.parametrize("case", CASES)
def test_refinement_includes_previous_code(case):
    blocks = case.build(_IMG_URL, 3, case.sample, None, None, None)
    text = "\n".join(text_blocks(blocks))
    assert case.sample in text
    assert "Iteration #3" in text


@pytest.mark.parametrize("case", CASES)
def test_render_url_included(case):
    blocks = case.build(_IMG_URL, 2, case.sample, _RENDER_URL, None, None)
    assert _RENDER_URL in image_urls(blocks)


@pytest.mark.parametrize("case", CASES)
def test_diff_url_included(case):
    blocks = case.build(_IMG_URL, 2, case.sample, None, None, _DIFF_URL)
    assert _DIFF_URL in image_urls(blocks)
    text = "\n".join(text_blocks(blocks))
    assert "Difference map" in text


@pytest.mark.parametrize("case", CASES)
def test_goal_included(case):
    blocks = case.build(_IMG_URL, 2, case.sample, None, "match the target", None)
    text = "\n".join(text_blocks(blocks))
    assert "match the target" in text


@pytest.mark.parametrize("case", CASES)
def test_diff_format_instructions_in_edit(case):
    blocks = case.build(_IMG_URL, 2, case.sample, None, None, None)
    text = "\n".join(text_blocks(blocks))
    assert "<<<SEARCH>>>" in text
    assert "<<<REPLACE>>>" in text
    assert "<<<END>>>" in text
    assert "search/replace diff blocks" in text
