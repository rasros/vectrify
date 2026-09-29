from vectrify.search.diversity import hamming_distance, simhash


def test_simhash_none_returns_none():
    assert simhash(None) is None


def test_simhash_empty_returns_none():
    assert simhash("") is None


def test_simhash_deterministic():
    text = "<svg><rect width='100'/></svg>"
    assert simhash(text) == simhash(text)


def test_simhash_different_texts_differ():
    assert simhash("<svg><rect/></svg>") != simhash("<svg><circle/></svg>")


def test_simhash_short_text_below_ngram_size():
    assert isinstance(simhash("ab"), int)


def test_hamming_distance_identical():
    assert hamming_distance(0b1010, 0b1010) == 0


def test_hamming_distance_single_bit():
    assert hamming_distance(0b1010, 0b1011) == 1


def test_hamming_distance_all_bits():
    assert hamming_distance(0, (1 << 64) - 1) == 64
