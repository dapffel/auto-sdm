"""The proposer.

Turns a data source plus a stated objective into candidate plans. The executable search
space comes from the registry — the proposer can never propose a method nothing
implements — and the *ordering* over that space comes from the profile and the priors.

The separation matters. What is possible is a fact about the registry; what is promising
is a hypothesis about the data. Conflating them would let a bad prior make a capability
invisible rather than merely unlikely.

Every candidate keeps a nonzero probability. Ranking is what the priors do; exclusion is
what they must not do, because a search that cannot propose the option a prior disfavours
can never discover that the prior was wrong.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from .backends import Backend, BackendRegistry
from .models import (
    Assumption,
    CandidatePlan,
    DataSource,
    MethodologySpec,
    MethodSpec,
    Objective,
    UnsupportedStep,
)
from .priors import DEFAULT_PRIORS, Prior
from .profile import DataProfile

# The stages a methodology must fill. ``background`` is conditional on the data regime,
# which is why it is not in this list.
REQUIRED_STAGES = ("cleaning", "accessible_area", "predictors", "split", "algorithm")


def _facts(profile: DataProfile | None, objective: Objective, occurrence_type: str) -> dict:
    return {"profile": profile, "objective": objective, "occurrence_type": occurrence_type}


def _weight_for(
    stage: str, method: str, facts: dict[str, Any], priors: tuple[Prior, ...]
) -> tuple[float, list[str]]:
    """Additive log weight for one method, plus the ids of the priors that moved it."""
    total = 0.0
    applied = []
    for prior in priors:
        if prior.stage == stage and prior.method == method and prior.applies(facts):
            total += prior.log_weight
            applied.append(prior.prior_id)
    return total, applied


def _derive_params(
    stage: str, method: str, profile: DataProfile | None, spec_params: dict[str, Any]
) -> dict[str, Any]:
    """Fill parameters the profile can determine, leaving anything explicit untouched.

    Block size in particular has no defensible fixed default: the right value is a
    property of the landscape's autocorrelation, which is exactly what the profile
    measures.
    """
    params = dict(spec_params)
    if profile is None:
        return params
    if stage == "split" and method == "spatial_block":
        params.setdefault("block_size", profile.suggested_block_size())
    if stage == "background" and method == "random":
        # A background sample far larger than the accessible area cannot be drawn without
        # replacement, and one far smaller under-describes what was available.
        params.setdefault("n_background", min(10_000, max(1_000, profile.spatial.extent_cells)))
    return params


def propose(
    data_source: DataSource,
    registry: BackendRegistry,
    *,
    profile: DataProfile | None = None,
    objective: Objective = "spatial_transfer",
    max_candidates: int | None = None,
    random_seed: int = 42,
    priors: tuple[Prior, ...] = DEFAULT_PRIORS,
) -> tuple[list[CandidatePlan], list[Assumption], list[UnsupportedStep]]:
    """Enumerate the executable search space, ranked by the priors.

    Returns the plans, whatever had to be assumed, and any stage the registry cannot fill
    at all — an empty stage is a capability gap, not an excuse to skip it.
    """
    assumptions: list[Assumption] = []
    gaps: list[UnsupportedStep] = []
    facts = _facts(profile, objective, data_source.occurrence_type)

    stages = list(REQUIRED_STAGES)
    if data_source.occurrence_type == "presence_only":
        stages.insert(3, "background")

    options: dict[str, list[Backend]] = {}
    for stage in stages:
        backends = registry.backends_for(stage)
        if not backends:
            gaps.append(
                UnsupportedStep(
                    code=f"no_backend_for_{stage}",
                    message=f"no registered backend can fill the required stage {stage!r}",
                    feature=stage,
                )
            )
        options[stage] = backends
    if gaps:
        return [], assumptions, gaps

    if profile is None:
        assumptions.append(
            Assumption(
                code="proposed_without_a_profile",
                message="no data profile was supplied, so candidates are unranked and "
                "profile-derived parameters fall back to backend defaults",
                field="profile",
            )
        )

    scored: list[tuple[float, CandidatePlan]] = []
    for combination in itertools.product(*(options[stage] for stage in stages)):
        chosen = dict(zip(stages, combination))
        total = 0.0
        specs: dict[str, MethodSpec] = {}

        for stage, backend in chosen.items():
            method = registry.canonical_method(backend)
            weight, applied = _weight_for(stage, method, facts, priors)
            total += weight
            specs[stage] = MethodSpec(
                method=method,
                params=_derive_params(stage, method, profile, {}),
                source=",".join(applied) or None,
            )

        candidate_id = "+".join(specs[stage].method for stage in ("algorithm", "split"))
        scored.append(
            (
                total,
                CandidatePlan(
                    candidate_id=candidate_id,
                    data_source=data_source,
                    methodology=MethodologySpec(**specs),
                    objective=objective,
                    random_seed=random_seed,
                ),
            )
        )

    ordered = _rank(scored, max_candidates, random_seed)
    if max_candidates is not None and max_candidates < len(scored):
        assumptions.append(
            Assumption(
                code="search_space_sampled",
                message=f"{len(scored)} candidates are executable but {max_candidates} were "
                "drawn, weighted by the priors; every candidate kept a nonzero chance",
                field="max_candidates",
            )
        )
    return ordered, assumptions, gaps


def _rank(
    scored: list[tuple[float, CandidatePlan]], max_candidates: int | None, random_seed: int
) -> list[CandidatePlan]:
    """Order by prior weight; sample without replacement when the budget binds.

    Sampling rather than truncating is the mechanism that keeps a disfavoured candidate
    reachable. Truncation would make the prior a filter, and a filter cannot be found to
    be wrong.
    """
    if not scored:
        return []
    by_weight = sorted(scored, key=lambda pair: (-pair[0], pair[1].candidate_id))
    if max_candidates is None or max_candidates >= len(scored):
        return [plan for _, plan in by_weight]

    weights = np.array([weight for weight, _ in by_weight], dtype=float)
    probabilities = np.exp(weights - weights.max())
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(random_seed)
    picked = rng.choice(len(by_weight), size=max_candidates, replace=False, p=probabilities)
    return [by_weight[index][1] for index in sorted(picked, key=lambda i: -weights[i])]


def explain(
    registry: BackendRegistry,
    profile: DataProfile | None,
    objective: Objective = "spatial_transfer",
    occurrence_type: str = "presence_absence",
    priors: tuple[Prior, ...] = DEFAULT_PRIORS,
) -> list[tuple[str, str, float, list[str]]]:
    """Per-method weights and the priors behind them — the proposer's reasoning, legible."""
    facts = _facts(profile, objective, occurrence_type)
    rows = []
    stages = list(REQUIRED_STAGES)
    if occurrence_type == "presence_only":
        stages.insert(3, "background")
    for stage in stages:
        for backend in registry.backends_for(stage):
            method = registry.canonical_method(backend)
            weight, applied = _weight_for(stage, method, facts, priors)
            rows.append((stage, method, weight, applied))
    return rows
