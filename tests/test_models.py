from __future__ import annotations

import pytest
from pydantic import ValidationError

from auto_sdm import (
    CandidatePlan,
    MethodologySpec,
    MethodSpec,
    RealDataSource,
    UnsupportedStep,
    VirtualDataSource,
)


def methodology(*, background: MethodSpec | None = MethodSpec(method="random")) -> MethodologySpec:
    return MethodologySpec(
        cleaning=MethodSpec(method="spatial_thin", params={"distance_km": 10}),
        accessible_area=MethodSpec(method="buffer", params={"radius_km": 200}),
        predictors=MethodSpec(method="vif", params={"threshold": 10}),
        background=background,
        split=MethodSpec(method="spatial_block", params={"n_blocks": 8}),
        algorithm=MethodSpec(method="maxent"),
    )


def real_source(occurrence_type: str = "presence_absence") -> RealDataSource:
    return RealDataSource(
        occurrences_path="occ.csv",
        predictor_paths=["bio1.tif", "bio12.tif"],
        occurrence_type=occurrence_type,  # type: ignore[arg-type]
    )


def test_presence_only_requires_a_background_strategy() -> None:
    with pytest.raises(ValidationError, match="background"):
        CandidatePlan(
            candidate_id="c1",
            data_source=real_source("presence_only"),
            methodology=methodology(background=None),
        )


def test_presence_absence_may_omit_background() -> None:
    plan = CandidatePlan(
        candidate_id="c1",
        data_source=real_source(),
        methodology=methodology(background=None),
    )
    assert plan.methodology.background is None


def test_a_plan_with_a_capability_gap_is_not_executable() -> None:
    plan = CandidatePlan(
        candidate_id="c1",
        data_source=real_source(),
        methodology=methodology(),
        unsupported_steps=[
            UnsupportedStep(code="unsupported_algorithm", message="no backend", feature="algorithm")
        ],
    )
    assert not plan.is_executable


def test_the_same_methodology_runs_against_real_and_virtual_data() -> None:
    """The split that makes selector calibration possible without a parallel path."""
    shared = methodology()
    virtual = VirtualDataSource(
        predictor_paths=["bio1.tif", "bio12.tif"],
        true_coefficients=[1.5, -0.8],
    )
    real = CandidatePlan(candidate_id="c1", data_source=real_source(), methodology=shared)
    simulated = CandidatePlan(candidate_id="c1", data_source=virtual, methodology=shared)
    assert real.methodology == simulated.methodology


def test_virtual_truth_must_cover_every_predictor() -> None:
    with pytest.raises(ValidationError, match="true_coefficients"):
        VirtualDataSource(predictor_paths=["bio1.tif", "bio12.tif"], true_coefficients=[1.0])


def test_data_source_is_discriminated_on_kind() -> None:
    plan = CandidatePlan(
        candidate_id="c1",
        data_source=VirtualDataSource(predictor_paths=["bio1.tif"], true_coefficients=[1.0]),
        methodology=methodology(),
    )
    restored = CandidatePlan.model_validate(plan.model_dump())
    assert isinstance(restored.data_source, VirtualDataSource)


def test_plans_are_immutable() -> None:
    plan = CandidatePlan(candidate_id="c1", data_source=real_source(), methodology=methodology())
    with pytest.raises(ValidationError):
        plan.candidate_id = "c2"  # type: ignore[misc]


def test_spatial_transfer_is_the_default_objective() -> None:
    plan = CandidatePlan(candidate_id="c1", data_source=real_source(), methodology=methodology())
    assert plan.objective == "spatial_transfer"


def test_methodology_steps_are_ordered_and_skip_omitted_stages() -> None:
    stages = [stage for stage, _ in methodology(background=None).steps()]
    assert stages == ["cleaning", "accessible_area", "predictors", "split", "algorithm"]
