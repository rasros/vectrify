"""Policies for choosing which mutation operator a task should apply.

The choice lives here rather than in the worker so that one policy sees every
outcome: eight worker processes each running their own copy could never learn
anything, since none of them observes whether its own children survived.
"""

import math
import random
from collections import deque
from collections.abc import Mapping
from typing import Protocol

from vectrify.score.metrics import SCORER_METRICS


class OperatorPolicy(Protocol):
    def select(self) -> str | None:
        """Name the operator the next mutation task should apply."""
        ...

    def update(self, operator: str | None, reward: float) -> None:
        """Report what an operator's child earned, in [0, 1].

        *operator* is what the worker actually applied, which may be None or a
        name this policy does not know: crossover falls back to mutation, and
        a backend may not have the operator the task named. Ignore those.
        """
        ...


class FixedWeightPolicy:
    """Draws from static weights and learns nothing.

    One weight table for every image and every stage of every run, which is
    what the backends did on their own before the choice moved here.
    """

    def __init__(self, weights: Mapping[str, float]):
        self._names = list(weights)
        self._weights = [weights[name] for name in self._names]

    def select(self) -> str | None:
        if not self._names:
            return None
        return random.choices(self._names, weights=self._weights, k=1)[0]

    def update(self, operator: str | None, reward: float) -> None:
        _ = operator, reward


# Exploration rate. EXP3 mixes this much uniform noise into every draw, which
# also fixes the floor under each operator at gamma / len(operators): one that
# looks useless early must survive to the phase it is good for. It also bounds
# how rare an expensive operator can be made -- with nine operators no prior
# can push one below about 1.7% of draws.
DEFAULT_GAMMA = 0.15

# No operator opens on exactly zero weight, so a table that lists one is always
# reachable.
_MIN_WEIGHT = 1e-3

# Per-update share of total weight redistributed uniformly (the .S in EXP3.S).
# This is the forgetting: without it the weights converge and the policy stops
# tracking, which is the same failure as the fixed table it replaces. 0.001
# keeps roughly the last thousand outcomes, a few dozen generations.
DEFAULT_ALPHA = 0.001


class Exp3Policy:
    """EXP3.S over the mutation operators.

    Survival is not an i.i.d. draw per operator: a child competes against the
    pool the policy itself just filled, so the payoff for nudging a number
    depends on what the other operators have been producing, and it drifts as
    the drawing gets closer. That is an adaptive adversary rather than a fixed
    stochastic bandit, which is what EXP3 assumes and what a Beta-Bernoulli
    posterior does not.

    Rewards are importance-weighted by the probability the arm was drawn with,
    so evidence about a rarely-picked operator is not diluted by how often the
    others were picked -- the correction a discounted-counts scheme lacks.
    """

    def __init__(
        self,
        operators: Mapping[str, float] | list[str],
        gamma: float = DEFAULT_GAMMA,
        alpha: float = DEFAULT_ALPHA,
    ):
        self._names = list(operators)
        # Start from the weights the table supplied rather than flat, which is
        # what `operator_weights` exists to provide. Flat is wrong whenever the
        # operators cost different amounts: the path fit takes about 0.5s of GPU
        # against a millisecond for a mutation, and an even opening split drew
        # it every ninth task and cut a run's throughput from 25 tasks a second
        # to 4. The weights are a prior on where to spend draws, not a verdict
        # -- EXP3 moves away from them as results arrive.
        supplied = operators if isinstance(operators, Mapping) else {}
        self._weights = {
            name: max(float(supplied.get(name, 1.0)), _MIN_WEIGHT)
            for name in self._names
        }
        self._gamma = gamma
        self._alpha = alpha
        # The probability each outstanding draw was made with. Results come
        # back long after selection and out of order, so the probability has to
        # be remembered per draw rather than recomputed at update time.
        self._drawn_with: dict[str, deque[float]] = {
            name: deque() for name in self._names
        }

    def probabilities(self) -> dict[str, float]:
        """The current selection distribution, for reporting and for drawing."""
        count = len(self._names)
        if count == 0:
            return {}
        total = sum(self._weights.values()) or 1.0
        return {
            name: (1.0 - self._gamma) * weight / total + self._gamma / count
            for name, weight in self._weights.items()
        }

    def select(self) -> str | None:
        if not self._names:
            return None
        probs = self.probabilities()
        name = random.choices(self._names, weights=[probs[n] for n in self._names])[0]
        self._drawn_with[name].append(probs[name])
        return name

    def update(self, operator: str | None, reward: float) -> None:
        if operator not in self._weights:
            return

        drawn = self._drawn_with[operator]
        # Empty when the worker substituted an operator for the one drawn, so
        # this outcome belongs to no draw of ours. Fall back to the current
        # probability rather than discarding the observation.
        probability = drawn.popleft() if drawn else self.probabilities()[operator]

        count = len(self._names)
        reward = max(0.0, min(1.0, reward))
        self._weights[operator] *= math.exp(
            self._gamma * (reward / probability) / count
        )

        # Redistribute a share of the total weight uniformly: an operator whose
        # weight has collapsed can climb back when its phase arrives.
        total = sum(self._weights.values())
        share = math.e * self._alpha / count * total
        for name in self._names:
            self._weights[name] += share

        # Rescale to keep exp() away from overflow. Only the ratios matter.
        largest = max(self._weights.values())
        if largest > 1e6:
            for name in self._names:
                self._weights[name] /= largest


# How much of the running scale a child has to beat to earn the whole reward.
# At 1.0 every second child saturates, since about half of them improve by more
# than the typical step and the reward stops telling those apart; at 2.0 a step
# twice the size of what the run is currently managing is what full credit
# costs, which leaves the ordinary improvements spread across the range.
_SATURATION = 2.0

# EMA weight per observation for the per-objective scale, so the scale tracks
# roughly the last hundred children. Long enough to be a stable denominator,
# short enough to follow a run that is converging.
_SCALE_MEMORY = 0.01

# Below this a scale is not yet established -- the first children of a run,
# before anything has been observed -- and dividing by it would turn the first
# improvement of any size into full credit.
_SCALE_FLOOR = 1e-12


class GradedReward:
    """How far a child improved on its parent, in [0, 1].

    A binary survived-or-not reward pays an operator that changes nothing
    about half the time, because a candidate the search cannot tell apart from
    its parent survives wherever the parent does. Guarding that with a no-op
    test only moves the problem: whatever the test calls a no-op, a change one
    step past it collects full price. Nudging a colour channel by 8 clears an
    exact-equality test while leaving the picture identical, and over one run it
    took 65% of the policy's weight on a black-and-white line drawing having
    moved the edge measure by 0.000003 on the one occasion it entered the
    winning lineage. Grading by magnitude removes the line to walk past.

    Improvements shrink as a run converges -- an early path fit moves the edge
    measure by 0.01 and a late one by 0.0001 -- so absolute improvement would
    decay until every operator scored zero and the policy stopped choosing.
    Each objective is therefore divided by a running scale of how big a step
    currently is, which keeps the operators comparable at every stage.
    """

    def __init__(
        self,
        names: tuple[str, ...] = SCORER_METRICS,
        memory: float = _SCALE_MEMORY,
        saturation: float = _SATURATION,
    ):
        self._names = names
        self._memory = memory
        self._saturation = saturation
        self._scale = dict.fromkeys(names, 0.0)
        self._seen = dict.fromkeys(names, 0)

    def scales(self) -> dict[str, float]:
        """The current per-objective step size, for reporting."""
        return dict(self._scale)

    def __call__(
        self, parent: Mapping[str, float], child: Mapping[str, float]
    ) -> float:
        """Reward for *child*, whose parent measured *parent*.

        The objectives are minimised, so a positive delta is an improvement.
        """
        deltas = {
            name: float(parent[name]) - float(child[name])
            for name in self._names
            if name in parent and name in child
        }
        if not deltas:
            return 0.0

        # The first child of a run has no scale to be measured against, and an
        # average that starts at zero and creeps up is worse than no scale at
        # all: the second child would be divided by a hundredth of the first
        # one's step and saturate whatever it did. So the first observation of
        # an objective seeds the scale outright, which grades that one child
        # against its own size -- exactly half the reward for an improvement of
        # any magnitude, which is the neutral opening.
        for name, delta in deltas.items():
            if self._seen[name] == 0:
                self._scale[name] = abs(delta)

        total = 0.0
        for name, delta in deltas.items():
            scale = self._scale[name]
            if scale > _SCALE_FLOOR:
                total += delta / (scale * self._saturation)

        for name, delta in deltas.items():
            self._seen[name] += 1
            self._scale[name] += self._memory * (abs(delta) - self._scale[name])

        return max(0.0, min(1.0, total / len(deltas)))
