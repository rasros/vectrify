import binascii

_NGRAM_SIZE = 4
_BITS = 64


def simhash(text: str | None) -> int | None:
    """
    Compute a 64-bit SimHash fingerprint for a string.

    SimHash is a locality-sensitive hash: similar texts produce similar
    fingerprints with small Hamming distance, while unrelated texts diverge.
    Identical texts always produce the same fingerprint, making it suitable
    for both exact deduplication (fingerprint equality) and diversity
    estimation (normalised Hamming distance).
    """
    if not text:
        return None
    encoded = text.encode()
    if len(encoded) < _NGRAM_SIZE:
        ngrams: set[bytes] = {encoded}
    else:
        ngrams = {
            encoded[i : i + _NGRAM_SIZE] for i in range(len(encoded) - _NGRAM_SIZE + 1)
        }

    v = [0] * _BITS
    for ng in ngrams:
        # Two CRC32s give us 64 independent bits cheaply.
        lo = binascii.crc32(b"\x00\x00\x00\x00" + ng) & 0xFFFFFFFF
        hi = binascii.crc32(b"\x01\x00\x00\x00" + ng) & 0xFFFFFFFF
        h = lo | (hi << 32)
        for i in range(_BITS):
            v[i] += 1 if (h >> i) & 1 else -1

    result = 0
    for i in range(_BITS):
        if v[i] > 0:
            result |= 1 << i
    return result


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two 64-bit SimHash values."""
    return (a ^ b).bit_count()
