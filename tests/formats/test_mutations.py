import random

from vectrify.formats.mutations import operator_weights, pick_operator

TABLE = (
    (str.upper, "upper", 1.0),
    (str.lower, "lower", 1.0),
)


def test_operator_weights_maps_names_to_weights():
    assert operator_weights(TABLE) == {"upper": 1.0, "lower": 1.0}


def test_pick_operator_returns_the_named_one():
    fn, name = pick_operator(TABLE, "lower")
    assert name == "lower"
    assert fn("AB") == "ab"


def test_pick_operator_falls_back_to_a_weighted_draw():
    """A policy carried over from another format names operators this backend
    does not have; losing the task would cost more than losing the choice."""
    random.seed(0)
    assert pick_operator(TABLE, "not-a-real-operator")[1] in {"upper", "lower"}
    assert pick_operator(TABLE, None)[1] in {"upper", "lower"}
