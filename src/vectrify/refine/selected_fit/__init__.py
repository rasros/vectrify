"""Explicit mechanics and acceptance boundaries for selected-path fitting.

Controls and stroke coverage provide differentiable mechanics. ExactObjective
renders candidates in the frozen FitContext and scores fairness in reference
units. RetainedFit alone owns the published recommendation and its exact error
allowance; the observer and both cleanup passes submit proposals through it.
CrossingRollback owns only the coordinate rollback budget, while FitObserver
keeps descent's stall baseline separate from the retained edge proposal.
Imports do not require optional Torch until fitting starts.
"""
