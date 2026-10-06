"""Lossless project compression at file boundaries; legacy JSON stays readable."""

from __future__ import annotations

import gzip
import io
import zlib

from vectrify.document import DocumentError

MAX_SOURCE = 128 * 1024 * 1024


def encode_project(source: str) -> bytes:
    """Save the complete project, including the reference, without losing data."""
    return gzip.compress(source.encode("utf-8"), compresslevel=6, mtime=0)


def decode_source(source: str | bytes) -> str:
    """Read plain SVG/JSON or gzip, bounding the decompressed size as well."""
    if isinstance(source, str):
        if len(source.encode("utf-8")) > MAX_SOURCE:
            raise DocumentError("File exceeds the 128 MB editor limit")
        return source
    if len(source) > MAX_SOURCE:
        raise DocumentError("File exceeds the 128 MB editor limit")
    if source.startswith(b"\x1f\x8b"):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(source)) as stream:
                source = stream.read(MAX_SOURCE + 1)
        except (OSError, EOFError, zlib.error) as exc:
            raise DocumentError("Invalid compressed project") from exc
        if len(source) > MAX_SOURCE:
            raise DocumentError("File exceeds the 128 MB editor limit")
    try:
        return source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentError("Expected an SVG or Vectrify project file") from exc
