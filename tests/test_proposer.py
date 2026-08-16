"""The proposer.

The property that matters most is not that the ranking is good — calibration will decide
that later — but that ranking never becomes filtering. A search that cannot propose the
candidate its priors disfavour can never discover that its priors were wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from auto_sdm import (
    ArtifactStore,
    Condition,
    Prior,
    VirtualDataSource,
    build_profile,
    default_registry,
    execute_run,
    explain,
    load_data_source,
    propose,
)
from auto_sdm.backends import BackendRegistry
from auto_sdm.priors import resolve_path


def profile_for(source: VirtualDataSource, seed: int = 7):
    context, _ = load_data_source(source, random_seed=seed)
    profile, _ = build_profile(context, random_seed=seed)
    return profile


def weights(rows, stage):
    return {method: weight for s, method, weight, _ in rows if s == stage}


# -- the executable space -------------------------------------------------------------


def test_every_proposal_is_executable(virtual_source: VirtualDataSource) -> None:
    """The registry decides what is possible; the proposer may not exceed it."""
    registry = default_registry()
    plans, _, gaps = propose(virtual_source, registry, profile=profile_for(virtual_source))
    assert gaps == []
    assert plans
    for plan in plans:
        for stage, spec in plan.methodology.steps():
            assert registry.resolve(stage, spec).ok


def test_method_aliases_do_not_duplicate_the_search_space(
    virtual_source: VirtualDataSource,
) -> None:
    """'gbm' and 'brt' are one backend; proposing both would halve the useful compute."""
    plans, _, _ = propose(virtual_source, default_registry(), profile=profile_for(virtual_source))
    # 2 algorithms x 2 splits, with one backend each for the other three stages.
    assert len(plans) == 4
    assert len({plan.candidate_id for plan in plans}) == 4


def test_presence_only_data_gets_a_background_stage(predictor_paths: list[str]) -> None:
    source = VirtualDataSource(
        predictor_paths=predictor_paths,
        true_coefficients=[1.5, -0.8],
        occurrence_type="presence_only",
        n_presences=120,
    )
    plans, _, _ = propose(source, default_registry(), profile=profile_for(source))
    assert plans
    assert all(plan.methodology.background is not None for plan in plans)


def test_an_unfillable_stage_is_a_gap_not_a_skipped_step(
    virtual_source: VirtualDataSource,
) -> None:
    """An empty registry must not yield a plan with stages quietly missing."""
    plans, _, gaps = propose(virtual_source, BackendRegistry())
    assert plans == []
    assert {gap.feature for gap in gaps} >= {"algorithm", "split", "predictors"}


# -- priors rank, they do not filter ---------------------------------------------------


def test_priors_rank_the_space_without_removing_anything(
    virtual_source: VirtualDataSource,
) -> None:
    """The load-bearing property: a disfavoured candidate stays reachable."""
    registry = default_registry()
    unranked, _, _ = propose(virtual_source, registry)
    ranked, _, _ = propose(virtual_source, registry, profile=profile_for(virtual_source))
    assert len(ranked) == len(unranked)
    assert {plan.candidate_id for plan in ranked} == {plan.candidate_id for plan in unranked}


def test_spatial_transfer_ranks_spatial_blocks_first(
    virtual_source: VirtualDataSource,
) -> None:
    plans, _, _ = propose(
        virtual_source,
        default_registry(),
        profile=profile_for(virtual_source),
        objective="spatial_transfer",
    )
    assert plans[0].methodology.split.method == "spatial_block"


def test_the_objective_changes_the_ranking(virtual_source: VirtualDataSource) -> None:
    """Different intended uses select genuinely different methodologies."""
    registry = default_registry()
    profile = profile_for(virtual_source)
    transfer = explain(registry, profile, objective="spatial_transfer")
    interpolation = explain(registry, profile, objective="interpolation")
    assert weights(transfer, "split")["spatial_block"] > weights(transfer, "split")["random_kfold"]
    assert (
        weights(interpolation, "split")["random_kfold"] > weights(transfer, "split")["random_kfold"]
    )


def test_small_samples_shift_the_ranking_toward_simpler_models(
    predictor_paths: list[str],
) -> None:
    small = VirtualDataSource(
        predictor_paths=predictor_paths,
        true_coefficients=[1.5, -0.8],
        occurrence_type="presence_only",
        n_presences=25,
    )
    large = VirtualDataSource(
        predictor_paths=predictor_paths,
        true_coefficients=[1.5, -0.8],
        n_presences=400,
        n_absences=400,
    )
    registry = default_registry()
    small_rows = explain(registry, profile_for(small), occurrence_type="presence_only")
    large_rows = explain(registry, profile_for(large))
    assert weights(small_rows, "algorithm")["logistic"] > weights(small_rows, "algorithm")["gbm"]
    assert weights(large_rows, "algorithm")["gbm"] > weights(small_rows, "algorithm")["gbm"]


def test_applied_priors_are_recorded_on_the_plan(virtual_source: VirtualDataSource) -> None:
    """A ranking nobody can audit is a ranking nobody should trust."""
    plans, _, _ = propose(virtual_source, default_registry(), profile=profile_for(virtual_source))
    blocked = next(p for p in plans if p.methodology.split.method == "spatial_block")
    assert blocked.methodology.split.source is not None
    assert "spatial_transfer_requires_spatial_blocks" in blocked.methodology.split.source


# -- budgeted sampling -----------------------------------------------------------------


def test_a_budget_samples_rather_than_truncating(virtual_source: VirtualDataSource) -> None:
    """Truncation would make a prior a filter, and a filter cannot be found to be wrong."""
    registry = default_registry()
    profile = profile_for(virtual_source)
    seen = set()
    for seed in range(40):
        plans, _, _ = propose(
            virtual_source, registry, profile=profile, max_candidates=1, random_seed=seed
        )
        assert len(plans) == 1
        seen.add(plans[0].candidate_id)
    # The disfavoured combination must still turn up rather than being unreachable.
    assert len(seen) > 1


def test_sampling_is_deterministic_given_the_seed(virtual_source: VirtualDataSource) -> None:
    registry = default_registry()
    profile = profile_for(virtual_source)
    first, _, _ = propose(
        virtual_source, registry, profile=profile, max_candidates=2, random_seed=5
    )
    second, _, _ = propose(
        virtual_source, registry, profile=profile, max_candidates=2, random_seed=5
    )
    assert [p.candidate_id for p in first] == [p.candidate_id for p in second]


def test_a_binding_budget_is_recorded(virtual_source: VirtualDataSource) -> None:
    _, assumptions, _ = propose(
        virtual_source,
        default_registry(),
        profile=profile_for(virtual_source),
        max_candidates=2,
    )
    assert any(a.code == "search_space_sampled" for a in assumptions)


def test_proposing_without_a_profile_is_recorded(virtual_source: VirtualDataSource) -> None:
    _, assumptions, _ = propose(virtual_source, default_registry())
    assert any(a.code == "proposed_without_a_profile" for a in assumptions)


# -- derived parameters ----------------------------------------------------------------


def test_block_size_comes_from_the_profile_not_a_magic_number(
    virtual_source: VirtualDataSource,
) -> None:
    """The measurement the profile exists to supply."""
    profile = profile_for(virtual_source)
    plans, _, _ = propose(virtual_source, default_registry(), profile=profile)
    blocked = next(p for p in plans if p.methodology.split.method == "spatial_block")
    assert blocked.methodology.split.params["block_size"] == profile.suggested_block_size()


def test_without_a_profile_no_parameters_are_invented(
    virtual_source: VirtualDataSource,
) -> None:
    plans, _, _ = propose(virtual_source, default_registry())
    blocked = next(p for p in plans if p.methodology.split.method == "spatial_block")
    assert "block_size" not in blocked.methodology.split.params


# -- the prior schema ------------------------------------------------------------------


def test_conditions_read_dotted_paths_through_profile_and_request() -> None:
    facts = {"objective": "interpolation", "profile": None}
    assert resolve_path(facts, "objective") == "interpolation"
    assert resolve_path(facts, "profile.volume.n_records") is None


def test_a_missing_measurement_never_satisfies_a_comparison() -> None:
    """Treating absent as zero would fire 'fewer than 50 records' on uncounted data."""
    condition = Condition(path="profile.volume.n_unique_cells", op="<", value=50)
    assert not condition.holds({"profile": None})


def test_priors_are_serialisable_so_lit_review_can_emit_them() -> None:
    prior = Prior(
        prior_id="p1",
        stage="algorithm",
        method="maxent",
        conditions=[Condition(path="profile.volume.n_records", op="<", value=50)],
        log_weight=1.0,
        rationale="because the paper said so",
        source="10.1111/example",
    )
    assert Prior.model_validate_json(prior.model_dump_json()) == prior


def test_an_unknown_method_in_a_prior_is_simply_inert(
    virtual_source: VirtualDataSource,
) -> None:
    """Priors must never conjure a capability the registry does not have."""
    invented = (
        Prior(
            prior_id="favour_maxent",
            stage="algorithm",
            method="maxent",
            log_weight=99.0,
            rationale="a prior for a method with no backend",
        ),
    )
    plans, _, gaps = propose(
        virtual_source,
        default_registry(),
        profile=profile_for(virtual_source),
        priors=invented,
    )
    assert gaps == []
    assert all(plan.methodology.algorithm.method != "maxent" for plan in plans)


# -- end to end ------------------------------------------------------------------------


def test_proposed_plans_execute(virtual_source: VirtualDataSource, tmp_path) -> None:
    registry = default_registry()
    plans, _, _ = propose(
        virtual_source, registry, profile=profile_for(virtual_source), random_seed=7
    )
    result = execute_run(plans, registry, ArtifactStore(tmp_path / "store"), run_id="proposed")
    assert result.unsupported_steps == []
    assert all(not candidate.failed for candidate in result.candidates), [
        c.failure_reason for c in result.candidates if c.failed
    ]
    assert all(np.isfinite(c.metrics["auc_mean"]) for c in result.candidates)
