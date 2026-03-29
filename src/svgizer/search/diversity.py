import binascii
import random

from svgizer.search.models import INVALID_SCORE, SearchNode

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

    # Accumulate per-bit signed votes across all n-grams.
    # Each n-gram hashes to a 64-bit value; bit i votes +1 if set, -1 if not.
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


def pool_diversity(nodes: list[SearchNode], sample_pairs: int = 100) -> float:
    """
    Estimate pool diversity via sampled pairwise normalised Hamming distance.

    Returns a value in [0, 1] where 1.0 = maximally diverse (all fingerprints
    differ in every bit) and 0.0 = all nodes are identical.  Returns 1.0 when
    there is too little data to measure.
    """
    sigs: list[int] = [
        n.signature
        for n in nodes
        if n.signature is not None and n.score < INVALID_SCORE
    ]
    if len(sigs) < 2:
        return 1.0

    n = len(sigs)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pairs = (
        all_pairs
        if len(all_pairs) <= sample_pairs
        else random.sample(all_pairs, sample_pairs)
    )

    total = sum(hamming_distance(sigs[i], sigs[j]) for i, j in pairs)
    return total / (len(pairs) * _BITS)
