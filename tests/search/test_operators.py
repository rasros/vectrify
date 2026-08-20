from vectrify.search.operators import (
    DEFAULT_GAMMA,
    Exp3Policy,
    FixedWeightPolicy,
    GradedReward,
)


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
        policy.update("a", reward=0.0)
    assert "a" in {policy.select() for _ in range(50)}


def test_exp3_shifts_towards_what_survives():
    policy = Exp3Policy({"good": 1.0, "bad": 1.0}, gamma=0.05)
    for _ in range(400):
        policy.update("good", reward=1.0)
        policy.update("bad", reward=0.0)

    probs = policy.probabilities()
    assert probs["good"] > probs["bad"] * 5


def test_exp3_tracks_a_change_in_which_operator_works():
    """What works changes within a run, so a policy whose weights converge is
    stuck on whatever won the opening phase."""
    policy = Exp3Policy({"early": 1.0, "late": 1.0}, gamma=0.05)
    for _ in range(400):
        policy.update("early", reward=1.0)
        policy.update("late", reward=0.0)
    first_phase = policy.probabilities()

    for _ in range(400):
        policy.update("early", reward=0.0)
        policy.update("late", reward=1.0)
    second_phase = policy.probabilities()

    assert first_phase["early"] > first_phase["late"] * 5
    assert second_phase["late"] > second_phase["early"] * 5


def test_exp3_keeps_a_floor_of_gamma_over_k_under_every_operator():
    """An operator that is useless early would otherwise be starved before the
    phase it is good for arrives, with no way back."""
    policy = Exp3Policy({"good": 1.0, "hopeless": 1.0}, gamma=0.2)
    for _ in range(2000):
        policy.update("hopeless", reward=0.0)

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
    policy.update("common", reward=1.0)
    policy._drawn_with["rare"].append(0.1)
    policy.update("rare", reward=1.0)

    probs = policy.probabilities()
    assert probs["rare"] > probs["common"]


def test_exp3_ignores_operators_it_does_not_know():
    policy = Exp3Policy({"a": 1.0})
    policy.update("crossover", reward=1.0)
    policy.update(None, reward=0.0)
    assert list(policy.probabilities()) == ["a"]


def test_exp3_survives_an_outcome_for_an_operator_it_never_drew():
    """The worker can substitute an operator for the one the task named."""
    policy = Exp3Policy({"a": 1.0, "b": 1.0})
    policy.update("a", reward=1.0)
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


def _grader():
    return GradedReward(names=("edge", "colour"))


def test_a_child_that_changes_nothing_perceptible_earns_nothing():
    """The reason the reward is graded at all. A binary survived-or-not reward
    pays an operator about half the time for changing nothing, since a candidate
    indistinguishable from its parent is admitted wherever the parent sits. On
    one run that let a colour nudge of 8 channel steps take 65% of the weight on
    a black-and-white line drawing.
    """
    grade = _grader()
    # Establish what a step in this run looks like.
    for _ in range(200):
        grade({"edge": 0.5, "colour": 0.5}, {"edge": 0.49, "colour": 0.5})

    parent = {"edge": 0.4, "colour": 0.4}
    inert = grade(parent, {"edge": 0.4 - 3e-6, "colour": 0.4 - 3e-6})
    real = grade(parent, {"edge": 0.39, "colour": 0.39})
    assert inert < 0.01
    assert real > 0.3
    assert real > inert * 20


def test_a_child_that_got_worse_earns_nothing():
    grade = _grader()
    for _ in range(200):
        grade({"edge": 0.5}, {"edge": 0.49})
    assert grade({"edge": 0.4}, {"edge": 0.6}) == 0.0


def test_the_reward_survives_a_run_converging():
    """Improvements shrink by orders of magnitude as a run converges. An
    absolute reward would decay to zero and stop telling the operators apart,
    which is the same failure as a policy that has stopped learning.
    """
    early = _grader()
    for _ in range(200):
        early({"edge": 0.5}, {"edge": 0.49})
    late = _grader()
    for _ in range(200):
        late({"edge": 0.05}, {"edge": 0.0499})

    # A typical step for each regime earns a comparable reward, a hundredfold
    # difference in absolute size notwithstanding.
    early_reward = early({"edge": 0.5}, {"edge": 0.49})
    late_reward = late({"edge": 0.05}, {"edge": 0.0499})
    assert abs(early_reward - late_reward) < 0.05


def test_a_parent_with_no_measures_leaves_nothing_to_grade():
    assert _grader()({}, {"edge": 0.4}) == 0.0


def test_the_first_child_of_a_run_opens_neutral():
    """No scale exists yet, and both alternatives are worse than a neutral
    opening: zero punishes an operator for going first, and a scale creeping up
    from zero saturates whatever the second child does.
    """
    assert _grader()({"edge": 0.5}, {"edge": 0.4}) == 0.5


def test_a_bigger_improvement_earns_more():
    grade = _grader()
    for _ in range(200):
        grade({"edge": 0.5}, {"edge": 0.49})
    small = grade({"edge": 0.4}, {"edge": 0.398})
    bigger = grade({"edge": 0.4}, {"edge": 0.395})
    assert bigger > small > 0.0
