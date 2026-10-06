"""File compression is exact, backwards compatible, and bounded when reading."""

import gzip

import pytest

from vectrify import project_file
from vectrify.document import DocumentError


@pytest.mark.parametrize("compressed", [False, True])
def test_project_file_keeps_exact_utf8_and_reference_data(compressed):
    source = '{"name":"Räven", "reference":"' + "abcdef012345" * 1000 + '"}'
    data = project_file.encode_project(source) if compressed else source.encode()
    assert project_file.decode_source(data) == source
    if compressed:
        assert len(data) < len(source.encode()) / 10


@pytest.mark.parametrize(
    "data",
    [b"\x1f\x8bgarbage", gzip.compress(b"project")[:-3], b"\xff\xfe"],
)
def test_invalid_files_give_document_errors(data):
    with pytest.raises(DocumentError):
        project_file.decode_source(data)


@pytest.mark.parametrize("compressed", [False, True])
def test_limits_apply_to_decompressed_size(monkeypatch, compressed):
    monkeypatch.setattr(project_file, "MAX_SOURCE", 1024)
    at_limit = "x" * 1024
    over_limit = at_limit + "x"
    encode = project_file.encode_project if compressed else str.encode
    assert project_file.decode_source(encode(at_limit)) == at_limit
    with pytest.raises(DocumentError, match="editor limit"):
        project_file.decode_source(encode(over_limit))


def test_concatenated_gzip_members_share_the_size_limit(monkeypatch):
    monkeypatch.setattr(project_file, "MAX_SOURCE", 1024)
    data = gzip.compress(b"x" * 600) + gzip.compress(b"y" * 600)
    with pytest.raises(DocumentError, match="editor limit"):
        project_file.decode_source(data)
