import io
from xml.etree import ElementTree as ET

import pytest
from PIL import Image

from tests.helpers import TEST_MODEL as _MODEL
from vectrify.formats.base import NoUsableOutputError
from vectrify.formats.svg.plugin import SvgPlugin
from vectrify.formats.svg.prompts import (
    build_svg_gen_prompt,
    extract_svg_fragment,
    is_valid_svg,
)

NS = "http://www.w3.org/2000/svg"
SVG = f'<svg xmlns="{NS}" viewBox="0 0 32 32"><rect width="32" height="32"/></svg>'


def _make_image_data_url(color: str = "blue", size: int = 32) -> str:
    import base64

    img = Image.new("RGB", (size, size), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def test_plugin_identity_matches_the_files_it_writes():
    plugin = SvgPlugin()
    assert plugin.name == "svg"
    assert plugin.file_extension == ".svg"


def test_rasterize_renders_at_the_requested_size():
    png = SvgPlugin().rasterize(SVG, out_w=24, out_h=16)
    assert Image.open(io.BytesIO(png)).size == (24, 16)


def test_validate_accepts_svg_and_rejects_other_roots():
    plugin = SvgPlugin()
    assert plugin.validate(SVG) == (True, None)
    ok, err = plugin.validate("<html></html>")
    assert not ok
    assert err


def test_validate_reports_a_parse_error():
    ok, err = SvgPlugin().validate("<svg><rect></svg>")
    assert not ok
    assert err is not None
    assert "parse error" in err


def test_extract_from_llm_strips_prose_and_fences():
    """Compared as markup, not as text: what comes back has been normalised for
    local search, so its serialisation differs from the model's."""
    raw = f"Sure, here you go:\n```xml\n{SVG}\n```\nHope that helps!"

    got = ET.fromstring(SvgPlugin().extract_from_llm(raw))
    want = ET.fromstring(SVG)

    assert [el.tag for el in got.iter()] == [el.tag for el in want.iter()]


def test_apply_edit_patches_the_parent_with_a_diff_block():
    raw = '<<<SEARCH>>>\n<rect width="32"\n<<<REPLACE>>>\n<rect width="16"\n<<<END>>>'
    result = SvgPlugin().apply_edit(SVG, raw)
    assert 'width="16"' in result
    assert result.startswith("<svg")


def test_apply_edit_falls_back_to_a_whole_document():
    whole = f'<svg xmlns="{NS}"><circle r="4"/></svg>'

    got = ET.fromstring(SvgPlugin().apply_edit(SVG, f"Here it is:\n{whole}"))

    want = ET.fromstring(whole)
    assert [el.tag for el in got.iter()] == [el.tag for el in want.iter()]
    circle = next(el for el in got.iter() if el.tag.endswith("circle"))
    assert circle.get("r") == "4"


def test_generate_prompt_carries_the_target_image_and_canvas():
    url = _make_image_data_url()
    blocks = SvgPlugin().build_generate_prompt(
        url,
        node_index=1,
        content_prev=None,
        raster_preview_url=None,
        goal="make it blue",
        canvas=(64, 48),
    )
    text = "\n".join(b["text"] for b in blocks if b["type"] == "input_text")
    assert "viewBox='0 0 64 48'" in text
    assert "make it blue" in text
    assert [b["image_url"] for b in blocks if b["type"] == "input_image"] == [url]


def test_the_render_preview_is_only_sent_with_a_parent():
    plugin = SvgPlugin()
    target, preview = _make_image_data_url("red"), _make_image_data_url("green")
    fresh = plugin.build_generate_prompt(
        target,
        node_index=1,
        content_prev=None,
        raster_preview_url=preview,
        goal=None,
        canvas=(32, 32),
    )
    refine = plugin.build_generate_prompt(
        target,
        node_index=2,
        content_prev=SVG,
        raster_preview_url=preview,
        goal=None,
        canvas=(32, 32),
    )
    assert [b["image_url"] for b in fresh if b["type"] == "input_image"] == [target]
    assert [b["image_url"] for b in refine if b["type"] == "input_image"] == [
        target,
        preview,
    ]


def test_mutate_returns_valid_svg_and_a_summary():
    content, summary = SvgPlugin().mutate(SVG)
    assert SvgPlugin().validate(content)[0]
    assert summary.strip()


def test_crossover_returns_valid_svg_and_a_summary():
    other = f'<svg xmlns="{NS}" viewBox="0 0 32 32"><circle r="8"/></svg>'
    content, summary = SvgPlugin().crossover(SVG, other)
    assert SvgPlugin().validate(content)[0]
    assert summary.strip()


@pytest.mark.llm
def test_llm_svg_generation_produces_valid_svg():
    from vectrify.llm import LLMConfig, get_provider

    client = get_provider("openai")
    prompt = build_svg_gen_prompt(_make_image_data_url("blue"), iter_index=1)
    raw = client.generate(prompt, LLMConfig(model=_MODEL))
    svg = extract_svg_fragment(raw)
    valid, err = is_valid_svg(svg)
    assert valid, f"LLM did not produce valid SVG: {err}\nRaw: {raw[:200]}"


@pytest.mark.llm
def test_llm_svg_refinement_produces_valid_svg():
    from vectrify.llm import LLMConfig, get_provider

    ns = "http://www.w3.org/2000/svg"
    parent_svg = f'<svg xmlns="{ns}"><rect width="32" height="32" fill="blue"/></svg>'
    prompt = build_svg_gen_prompt(
        _make_image_data_url("red"),
        iter_index=2,
        svg_prev=parent_svg,
        goal="Make the fill color match the target image.",
    )
    client = get_provider("openai")
    raw = client.generate(prompt, LLMConfig(model=_MODEL))
    svg = SvgPlugin().apply_edit(parent_svg, raw)
    valid, err = is_valid_svg(svg)
    assert valid, f"LLM refinement did not produce valid SVG: {err}\nRaw: {raw[:200]}"


_PARENT = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
    '<circle cx="5" cy="5" r="2" /></svg>'
)


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("", id="empty"),
        pytest.param("I cannot see the image clearly, so I am skipping.", id="prose"),
        pytest.param("```svg\n```", id="empty-fence"),
    ],
)
def test_a_reply_with_nothing_drawn_in_it_says_so(raw):
    """Not an XML complaint: nothing was drawn, so there is no markup to blame.

    The extractor falls back to returning the whole reply, so prose used to
    reach the parser and come back as "not well-formed (invalid token): line 1,
    column 1", which reads as a broken drawing.
    """
    plugin = SvgPlugin()
    for call in (
        lambda: plugin.apply_edit(_PARENT, raw),
        lambda: plugin.extract_from_llm(raw),
    ):
        with pytest.raises(NoUsableOutputError) as caught:
            call()
        assert "no diff blocks" in str(caught.value)


def test_the_error_quotes_what_came_back_instead():
    plugin = SvgPlugin()
    with pytest.raises(NoUsableOutputError, match="rate limit"):
        plugin.apply_edit(_PARENT, "Sorry, rate limit reached.")
    with pytest.raises(NoUsableOutputError, match="the reply was empty"):
        plugin.apply_edit(_PARENT, "   \n  ")


def test_a_usable_reply_is_still_applied():
    plugin = SvgPlugin()
    edited = plugin.apply_edit(
        _PARENT, '<<<SEARCH>>>\nr="2"\n<<<REPLACE>>>\nr="3"\n<<<END>>>'
    )
    assert 'r="3"' in edited
    assert "<svg" in plugin.extract_from_llm("here it is\n" + _PARENT)


_STROKED = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
    '<g id="curve" fill="none" stroke="#111111" stroke-width="3">'
    '<path d="M 8 32 C 20 20 44 20 56 32" /></g></svg>'
)


def test_the_path_fit_is_only_offered_where_it_is_cheap_enough():
    """It costs ~0.5s on a GPU against ~9s on one CPU thread, where an ordinary
    mutation costs about a millisecond. Off the GPU it is not worth a worker.
    """
    from vectrify.refine.paths import PATH_FIT, fit_available

    weights = SvgPlugin().mutation_weights()
    assert (PATH_FIT in weights) == fit_available()
    assert sum(weights.values()) > 0


def test_asking_for_the_fit_without_a_reference_changes_nothing():
    """Handing back the content is how an operator reports a blank draw; the
    worker charges it to this operator by name rather than scoring a clone.
    """
    from vectrify.refine.paths import PATH_FIT

    content, origin = SvgPlugin().mutate(_STROKED, operator=PATH_FIT)
    assert content == _STROKED
    assert origin == PATH_FIT


def test_a_drawing_with_nothing_fittable_reports_a_blank_draw():
    from vectrify.refine.paths import PATH_FIT

    dots = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<g id="dots" fill="#000000"><circle cx="10" cy="10" r="3" /></g></svg>'
    )
    png = SvgPlugin().rasterize(dots, 64, 64)
    content, origin = SvgPlugin().mutate(dots, operator=PATH_FIT, reference_png=png)
    assert content == dots
    assert origin == PATH_FIT


def test_a_width_on_the_path_is_found_as_readily_as_one_on_the_group():
    """The width can be declared on the element, a group above it, or the root.
    One model wrote it on every path and none on their groups; a lookup that
    checked only the group and the root found nothing fittable in an entire
    run, so the operator never fired once.
    """
    import xml.etree.ElementTree as ET

    from vectrify.refine.paths import fittable_groups

    on_path = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<g id="curve"><path d="M 8 32 C 20 20 44 20 56 32" fill="none" '
        'stroke="#111111" stroke-width="3.2" /></g></svg>'
    )
    groups = fittable_groups(ET.fromstring(on_path))
    assert len(groups) == 1
    _group, paths, widths = groups[0]
    assert len(paths) == 1
    assert widths == [3.2]


def test_paths_of_differing_width_in_one_group_keep_their_own():
    """Real output mixes widths inside a group -- 3.2 and 4 on two halves of one
    outline -- so a single width for the group would render one of them wrong.
    """
    import xml.etree.ElementTree as ET

    from vectrify.refine.paths import fittable_groups

    mixed = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<g id="outline" fill="none" stroke="#111111">'
        '<path d="M 8 32 C 20 20 44 20 56 32" stroke-width="3.2" />'
        '<path d="M 8 40 C 20 52 44 52 56 40" stroke-width="4" /></g></svg>'
    )
    _group, paths, widths = fittable_groups(ET.fromstring(mixed))[0]
    assert len(paths) == 2
    assert widths == [3.2, 4.0]


def test_an_unstroked_group_is_not_fittable():
    import xml.etree.ElementTree as ET

    from vectrify.refine.paths import fittable_groups

    filled = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<g id="blob" fill="#000000" stroke="none">'
        '<path d="M 8 32 C 20 20 44 20 56 32" /></g></svg>'
    )
    assert fittable_groups(ET.fromstring(filled)) == []


def test_a_full_device_skips_the_fit_instead_of_failing_the_task():
    """One run turned an exhausted GPU into thousands of failed candidates.
    Every worker that fits holds a context of a few hundred MB and there are as
    many workers as cores, so running out is a normal condition, not a bug.
    """
    import vectrify.refine.paths as paths
    from vectrify.refine.paths import PATH_FIT

    plugin = SvgPlugin()
    png = plugin.rasterize(_STROKED, 64, 64)
    original = paths.fit_random_group

    def out_of_memory(*_args, **_kwargs):
        raise RuntimeError("CUDA error: out of memory")

    from vectrify.formats.svg import plugin as plugin_module

    plugin_module.fit_random_group = out_of_memory
    plugin_module.fit_available = lambda: True
    try:
        content, origin = plugin.mutate(_STROKED, operator=PATH_FIT, reference_png=png)
    finally:
        plugin_module.fit_random_group = original
        plugin_module.fit_available = paths.fit_available

    assert content == _STROKED
    assert origin == PATH_FIT


def test_an_unrelated_failure_in_the_fit_is_not_swallowed():
    import vectrify.refine.paths as paths
    from vectrify.formats.svg import plugin as plugin_module
    from vectrify.refine.paths import PATH_FIT

    plugin = SvgPlugin()
    png = plugin.rasterize(_STROKED, 64, 64)
    original = plugin_module.fit_random_group

    def bug(*_args, **_kwargs):
        raise ValueError("something genuinely wrong")

    plugin_module.fit_random_group = bug
    plugin_module.fit_available = lambda: True
    try:
        with pytest.raises(ValueError, match="genuinely wrong"):
            plugin.mutate(_STROKED, operator=PATH_FIT, reference_png=png)
    finally:
        plugin_module.fit_random_group = original
        plugin_module.fit_available = paths.fit_available


def test_path_fit_receives_the_shared_gpu_gate():
    from vectrify.formats.svg import plugin as plugin_module
    from vectrify.refine.paths import PATH_FIT

    class Gate:
        pass

    plugin = SvgPlugin()
    plugin.gpu_gate = gate = Gate()
    png = plugin.rasterize(_STROKED, 64, 64)
    original_fit = plugin_module.fit_random_group
    original_available = plugin_module.fit_available
    seen = {}

    def fit(*args, **kwargs):
        seen["gpu_gate"] = kwargs["gpu_gate"]
        return args[0]

    plugin_module.fit_random_group = fit
    plugin_module.fit_available = lambda: True
    try:
        plugin.mutate(_STROKED, operator=PATH_FIT, reference_png=png)
    finally:
        plugin_module.fit_random_group = original_fit
        plugin_module.fit_available = original_available

    assert seen["gpu_gate"] is gate
