from svgizer.svg.adapter import SvgResultPayload, SvgStatePayload, make_is_svg_stale


def create_payload(svg_text: str | None) -> SvgStatePayload:
    return SvgStatePayload(
        svg=svg_text,
        raster_data_url=None,
        raster_preview_data_url=None,
        change_summary=None,
        invalid_msg=None,
    )


def create_result(svg_text: str | None) -> SvgResultPayload:
    return SvgResultPayload(svg=svg_text, raster_png=None, change_summary=None)


def test_is_svg_stale_identical():
    is_stale = make_is_svg_stale(0.995)
    prev = create_payload("<svg><rect/></svg>")
    new_res = create_result("<svg><rect/></svg>")
    assert is_stale(prev, new_res) is True


def test_is_svg_stale_completely_different():
    is_stale = make_is_svg_stale(0.995)
    prev = create_payload("<svg><rect/></svg>")
    new_res = create_result("<svg><circle r='10'/></svg>")
    assert is_stale(prev, new_res) is False


def test_is_svg_stale_above_threshold():
    is_stale = make_is_svg_stale(0.90)
    prev = create_payload("<svg><rect width='10'/></svg>")
    new_res = create_result("<svg><rect width='11'/></svg>")
    assert is_stale(prev, new_res) is True


def test_is_svg_stale_handles_none():
    is_stale = make_is_svg_stale()
    prev = create_payload(None)
    new_res = create_result("<svg><rect/></svg>")
    assert is_stale(prev, new_res) is False


def test_is_svg_stale_handles_new_none():
    is_stale = make_is_svg_stale()
    prev = create_payload("<svg><rect/></svg>")
    new_res = create_result(None)
    assert is_stale(prev, new_res) is True
