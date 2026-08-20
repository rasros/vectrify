from vectrify.search.operators import DEFAULT_GAMMA, Exp3Policy, FixedWeightPolicy


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


def test_exp3_shifts_towards_what_survives():
    policy = Exp3Policy({"good": 1.0, "bad": 1.0}, gamma=0.05)
    for _ in range(400):
        policy.update("good", survived=True)
        policy.update("bad", survived=False)

    probs = policy.probabilities()
    assert probs["good"] > probs["bad"] * 5


def test_exp3_tracks_a_change_in_which_operator_works():
    """What works changes within a run, so a policy whose weights converge is
    stuck on whatever won the opening phase."""
    policy = Exp3Policy({"early": 1.0, "late": 1.0}, gamma=0.05)
    for _ in range(400):
        policy.update("early", survived=True)
        policy.update("late", survived=False)
    first_phase = policy.probabilities()

    for _ in range(400):
        policy.update("early", survived=False)
        policy.update("late", survived=True)
    second_phase = policy.probabilities()

    assert first_phase["early"] > first_phase["late"] * 5
    assert second_phase["late"] > second_phase["early"] * 5


def test_exp3_keeps_a_floor_of_gamma_over_k_under_every_operator():
    """An operator that is useless early would otherwise be starved before the
    phase it is good for arrives, with no way back."""
    policy = Exp3Policy({"good": 1.0, "hopeless": 1.0}, gamma=0.2)
    for _ in range(2000):
        policy.update("hopeless", survived=False)

    assert policy.probabilities()["hopeless"] >= 0.2 / 2


def test_exp3_weights_a_reward_by_the_probability_it_was_drawn_with():
    """Two operators that each survived once are equally good however often
    each was drawn. Without the importance weighting the rare one looks worse,
    which is the correction a discounted-counts scheme lacks.

    Reaches into the recorded draw probabilities because that is the quantity
    under test and select() cannot be made to produce a chosen one.
    """
    policy = Exp3Policy({"common": 1.0, "rare": 1.0}, gamma=0.5, alpha=0.0)

    policy._drawn_with["common"].append(0.9)
    policy.update("common", survived=True)
    policy._drawn_with["rare"].append(0.1)
    policy.update("rare", survived=True)

    probs = policy.probabilities()
    assert probs["rare"] > probs["common"]


def test_exp3_ignores_operators_it_does_not_know():
    policy = Exp3Policy({"a": 1.0})
    policy.update("crossover", survived=True)
    policy.update(None, survived=False)
    assert list(policy.probabilities()) == ["a"]


def test_exp3_survives_an_outcome_for_an_operator_it_never_drew():
    """The worker can substitute an operator for the one the task named."""
    policy = Exp3Policy({"a": 1.0, "b": 1.0})
    policy.update("a", survived=True)
    assert policy.probabilities()["a"] > 0


def test_exp3_without_operators_selects_nothing():
    assert Exp3Policy({}).select() is None


def test_the_opening_split_follows_the_weights_it_was_given():
    """`operator_weights` exists to supply a prior, and a flat opening is wrong
    when operators cost different amounts: drawing a 0.5s GPU fit as often as a
    1ms nudge took one run from 25 tasks a second to 4.
    """
    policy = Exp3Policy({"cheap": 0.5, "dear": 0.03})
    probabilities = policy.probabilities()
    assert probabilities["cheap"] > probabilities["dear"] * 3


def test_no_operator_is_starved_below_the_exploration_floor():
    policy = Exp3Policy({"cheap": 0.97, "dear": 0.03})
    floor = DEFAULT_GAMMA / 2
    assert policy.probabilities()["dear"] >= floor * 0.99


def test_a_bare_list_still_opens_flat():
    policy = Exp3Policy(["a", "b", "c"])
    values = list(policy.probabilities().values())
    assert max(values) - min(values) < 1e-9
