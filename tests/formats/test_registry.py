import pytest

from vectrify.formats import FORMAT_NAMES, FORMAT_PLUGINS, get_plugin


def test_registry_covers_the_cli_formats():
    assert set(FORMAT_NAMES) == {"svg", "graphviz", "typst"}


@pytest.mark.parametrize("name", FORMAT_NAMES)
def test_get_plugin_returns_a_matching_plugin(name):
    plugin = get_plugin(name)
    assert plugin.name == name
    assert plugin.file_extension.startswith(".")


@pytest.mark.parametrize("name", FORMAT_NAMES)
def test_plugins_implement_the_protocol_surface(name):
    plugin = get_plugin(name)
    for method in (
        "rasterize",
        "validate",
        "extract_from_llm",
        "apply_edit",
        "build_generate_prompt",
        "mutate",
        "crossover",
    ):
        assert callable(getattr(plugin, method)), f"{name} missing {method}"


def test_file_extensions_are_unique():
    extensions = [get_plugin(n).file_extension for n in FORMAT_NAMES]
    assert len(set(extensions)) == len(extensions)


def test_unknown_format_raises():
    with pytest.raises(ValueError, match="Unknown format"):
        get_plugin("tikz")


def test_registry_entries_are_lazy():
    assert all(callable(v) for v in FORMAT_PLUGINS.values())
