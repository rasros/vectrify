from svgizer.search.base import compute_signature, estimate_jaccard

# ---------------------------------------------------------------------------
# compute_signature
# ---------------------------------------------------------------------------


def test_compute_signature_none_returns_none():
    assert compute_signature(None) is None


def test_compute_signature_empty_string_returns_none():
    assert compute_signature("") is None


def test_compute_signature_returns_tuple():
    sig = compute_signature("hello world")
    assert isinstance(sig, tuple)
    assert len(sig) == 64  # default num_perms


def test_compute_signature_is_deterministic():
    text = "<svg><rect width='100'/></svg>"
    assert compute_signature(text) == compute_signature(text)


def test_compute_signature_short_text_below_ngram_size():
    # Text shorter than ngram_size=4 still produces a valid signature
    sig = compute_signature("ab", ngram_size=4)
    assert isinstance(sig, tuple)
    assert len(sig) == 64


def test_compute_signature_different_texts_differ():
    sig_a = compute_signature("<svg><rect/></svg>")
    sig_b = compute_signature("<svg><circle/></svg>")
    assert sig_a != sig_b


# ---------------------------------------------------------------------------
# estimate_jaccard
# ---------------------------------------------------------------------------


def test_estimate_jaccard_both_none_returns_zero():
    assert estimate_jaccard(None, None) == 0.0


def test_estimate_jaccard_one_none_returns_zero():
    sig = compute_signature("hello")
    assert estimate_jaccard(sig, None) == 0.0
    assert estimate_jaccard(None, sig) == 0.0


def test_estimate_jaccard_mismatched_lengths_returns_zero():
    assert estimate_jaccard((1, 2, 3), (1, 2)) == 0.0


def test_estimate_jaccard_identical_signatures_returns_one():
    sig = compute_signature("some svg text")
    assert estimate_jaccard(sig, sig) == 1.0


def test_estimate_jaccard_different_texts_between_zero_and_one():
    sig_a = compute_signature("<svg><rect width='100'/></svg>")
    sig_b = compute_signature("<svg><circle r='50'/></svg>")
    result = estimate_jaccard(sig_a, sig_b)
    assert 0.0 <= result <= 1.0
