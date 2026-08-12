from vectrify.search.operators import FixedWeightPolicy


def test_fixed_weight_policy_only_returns_known_operators():
    policy = FixedWeightPolicy({"a": 1.0, "b": 1.0})
    assert {policy.select() for _ in range(50)} <= {"a", "b"}


def test_fixed_weight_policy_respects_a_zero_weight():
    policy = FixedWeightPolicy({"a": 1.0, "never": 0.0})
    assert all(policy.select() == "a" for _ in range(50))


def test_fixed_weight_policy_without_operators_selects_nothing():
    assert FixedWeightPolicy({}).select() is None


def test_fixed_weight_policy_ignores_feedback():
    policy = FixedWeightPolicy({"a": 1.0, "b": 1.0})
    for _ in range(100):
        policy.update("a", survived=False)
    assert "a" in {policy.select() for _ in range(50)}
