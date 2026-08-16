def test_shape_distance_ignores_where_a_shape_sits():
    """The property ten other formulations lacked. A feature drawn correctly in
    the wrong place is not damaged, and every measure that counted pieces or
    compared contours scored it worse than a feature that had come apart."""
    import numpy as np

    from vectrify.score.moments import shape_distance

    ring = np.zeros((80, 80), dtype=bool)
    yy, xx = np.mgrid[0:80, 0:80]
    r = np.sqrt((yy - 40) ** 2 + (xx - 40) ** 2)
    ring[(r > 22) & (r < 28)] = True

    moved = np.roll(np.roll(ring, 7, axis=0), 5, axis=1)
    broken = ring.copy()
    for cut in range(0, 80, 10):
        broken[:, cut : cut + 4] = False

    assert shape_distance(ring, moved) < 0.5 * shape_distance(ring, broken)


def test_shape_distance_is_silent_on_an_identical_drawing():
    import numpy as np

    from vectrify.score.moments import shape_distance

    blob = np.zeros((40, 40), dtype=bool)
    blob[10:30, 12:28] = True

    assert shape_distance(blob, blob.copy()) == 0.0
