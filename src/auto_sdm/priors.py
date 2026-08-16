"""Methodological priors.

A prior is a rule keyed on data characteristics: *"presence-only, fewer than 50 records →
favour regularized MaxEnt with target-group background"*. It does not change what the
system **can** execute — that is the registry's job — only which candidates are worth
spending compute on. A search over forty configurations with no prior wastes most of its
compute; a literature-informed proposal distribution is what makes the search tractable.

Priors are **data, not code**. The initial set below is hand-authored, but it is authored
in the same structure `lit-review` will emit into, so extracting priors from papers in
Phase 4 adds rows rather than replacing the mechanism. Every prior carries a ``rationale``
and a ``source``; a prior whose provenance cannot be stated is one nobody can later
check.

Weights are additive in log space, so several weak priors compose into one strong
preference without any of them being able to zero a candidate out. That is deliberate:
priors *condition* the search, they never filter it. A prior that could drive a weight to
zero would silently convert the search into a heuristic and remove the system's ability to
discover that the prior was wrong.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .models import FrozenModel

Operator = Literal["<", "<=", ">", ">=", "==", "!=", "is_null", "not_null"]


class Condition(FrozenModel):
    """One test against a dotted path in the facts about a dataset.

    Paths address the profile (``profile.volume.n_unique_cells``) or the request
    (``objective``, ``occurrence_type``), so a prior can key on the data, the user's
    stated intent, or both.
    """

    path: str
    op: Operator
    value: float | int | str | bool | None = None

    def holds(self, facts: dict[str, Any]) -> bool:
        actual = resolve_path(facts, self.path)
        if self.op == "is_null":
            return actual is None
        if self.op == "not_null":
            return actual is not None
        if actual is None:
            # An absent measurement cannot satisfy a comparison. Treating missing as zero
            # would let "fewer than 50 records" fire on data whose count was never taken.
            return False
        if self.op == "==":
            return bool(actual == self.value)
        if self.op == "!=":
            return bool(actual != self.value)
        if not isinstance(actual, (int, float)) or not isinstance(self.value, (int, float)):
            return False
        if self.op == "<":
            return actual < self.value
        if self.op == "<=":
            return actual <= self.value
        if self.op == ">":
            return actual > self.value
        return actual >= self.value


class Prior(FrozenModel):
    """A weight adjustment for one method, conditional on the data and the objective."""

    prior_id: str
    stage: str
    method: str
    conditions: list[Condition] = Field(default_factory=list)
    log_weight: float
    rationale: str
    source: str | None = None  # DOI, or a run id for a prior learned from history

    def applies(self, facts: dict[str, Any]) -> bool:
        """All conditions must hold. An empty condition list is an unconditional prior."""
        return all(condition.holds(facts) for condition in self.conditions)


def resolve_path(facts: dict[str, Any], path: str) -> Any:
    """Walk a dotted path through dicts and pydantic models, returning None if absent."""
    current: Any = facts
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
    return current


# --------------------------------------------------------------------------------------
# The hand-authored starting set
#
# Small on purpose. These encode uncontroversial methodological consensus, and each one is
# a hypothesis the calibration harness will eventually be able to confirm or refute by
# measuring whether conditioning on it reduced regret.
# --------------------------------------------------------------------------------------

DEFAULT_PRIORS: tuple[Prior, ...] = (
    Prior(
        prior_id="small_sample_favours_simple_models",
        stage="algorithm",
        method="logistic",
        conditions=[Condition(path="profile.volume.n_unique_cells", op="<", value=50)],
        log_weight=1.0,
        rationale="With few unique records a flexible learner fits sampling noise; a "
        "regularized linear response is the standard small-sample recommendation.",
    ),
    Prior(
        prior_id="small_sample_penalises_boosting",
        stage="algorithm",
        method="gbm",
        conditions=[Condition(path="profile.volume.n_unique_cells", op="<", value=50)],
        log_weight=-1.0,
        rationale="Boosted trees need more records than this to estimate interactions "
        "rather than memorise them.",
    ),
    Prior(
        prior_id="ample_data_favours_flexible_models",
        stage="algorithm",
        method="gbm",
        conditions=[Condition(path="profile.volume.n_unique_cells", op=">=", value=300)],
        log_weight=0.7,
        rationale="With enough records a flexible learner can recover non-linear responses "
        "and interactions that a linear model cannot represent.",
    ),
    Prior(
        prior_id="collinear_predictors_favour_trees",
        stage="algorithm",
        method="gbm",
        conditions=[Condition(path="profile.environmental.max_vif", op=">", value=10)],
        log_weight=0.5,
        rationale="Strong collinearity destabilises linear coefficients; tree ensembles are "
        "far less sensitive to it.",
    ),
    Prior(
        prior_id="clustered_sampling_requires_spatial_blocks",
        stage="split",
        method="spatial_block",
        conditions=[Condition(path="profile.spatial.nearest_neighbour_index", op="<", value=0.8)],
        log_weight=1.5,
        rationale="Clustered records make random folds share neighbourhoods, which inflates "
        "held-out scores by an amount that differs between candidates and so biases the "
        "ranking, not merely the scores.",
    ),
    Prior(
        prior_id="covariate_shift_requires_spatial_blocks",
        stage="split",
        method="spatial_block",
        conditions=[Condition(path="profile.sampling.covariate_shift", op=">", value=0.5)],
        log_weight=1.0,
        rationale="A sample that does not resemble the landscape is being asked to "
        "extrapolate; random folds cannot measure that.",
    ),
    Prior(
        prior_id="spatial_transfer_requires_spatial_blocks",
        stage="split",
        method="spatial_block",
        conditions=[Condition(path="objective", op="==", value="spatial_transfer")],
        log_weight=1.5,
        rationale="Transfer to unsampled geography is exactly what block cross-validation "
        "estimates and what random cross-validation cannot.",
    ),
    Prior(
        prior_id="interpolation_tolerates_random_folds",
        stage="split",
        method="random_kfold",
        conditions=[Condition(path="objective", op="==", value="interpolation")],
        log_weight=0.5,
        rationale="Predicting within the sampled region is the one case where random folds "
        "answer the question actually being asked.",
    ),
)
