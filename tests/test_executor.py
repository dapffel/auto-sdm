from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import methodology

from auto_sdm import (
    ArtifactStore,
    CandidatePlan,
    MethodSpec,
    RealDataSource,
    VirtualDataSource,
    default_registry,
    execute_candidate,
    execute_run,
)


def test_a_candidate_runs_end_to_end(plan: CandidatePlan, store: ArtifactStore) -> None:
    result = execute_candidate(plan, default_registry(), store, "run1")
    assert not result.failed, result.failure_reason
    assert [record.stage for record in result.step_records] == [
        "data_source",
        "cleaning",
        "accessible_area",
        "predictors",
        "split",
        "algorithm",
    ]
    assert 0.5 < result.metrics["auc_mean"] <= 1.0
    assert "auc_sd" in result.metrics


def test_every_step_is_recorded_with_its_backend_and_seed(
    plan: CandidatePlan, store: ArtifactStore
) -> None:
    result = execute_candidate(plan, default_registry(), store, "run1")
    for record in result.step_records:
        assert record.backend.startswith("python-")
        assert record.random_seed == plan.random_seed
        assert record.inputs_digest
        assert record.duration_seconds >= 0


def test_step_records_are_persisted(plan: CandidatePlan, store: ArtifactStore) -> None:
    execute_candidate(plan, default_registry(), store, "run1")
    steps = store.root / "runs" / "run1" / "candidates" / "c1" / "steps"
    assert sorted(path.name for path in steps.iterdir()) == [
        "00-data_source.json",
        "01-cleaning.json",
        "02-accessible_area.json",
        "03-predictors.json",
        "04-split.json",
        "05-algorithm.json",
    ]


def test_a_stage_that_changes_nothing_re_emits_the_same_artifact(
    plan: CandidatePlan, store: ArtifactStore
) -> None:
    """Content addressing makes 'this stage was a no-op' auditable, not asserted."""
    result = execute_candidate(plan, default_registry(), store, "run1")
    by_stage = {record.stage: record for record in result.step_records}
    cleaning_coords = {
        artifact.name.split(".")[-1]: artifact.digest for artifact in by_stage["cleaning"].artifacts
    }
    predictor_coords = {
        artifact.name.split(".")[-1]: artifact.digest
        for artifact in by_stage["predictors"].artifacts
    }
    # No record fell on a missing cell, so predictors passes coords through untouched.
    assert cleaning_coords["coords"] == predictor_coords["coords"]


def test_execution_is_deterministic_given_the_seed(plan: CandidatePlan, tmp_path: Path) -> None:
    """The run record claims reproducibility; this is the claim being tested."""
    first = execute_candidate(plan, default_registry(), ArtifactStore(tmp_path / "a"), "r")
    second = execute_candidate(plan, default_registry(), ArtifactStore(tmp_path / "b"), "r")
    assert first.metrics == second.metrics
    assert [record.inputs_digest for record in first.step_records] == [
        record.inputs_digest for record in second.step_records
    ]


def test_a_different_seed_changes_the_result(plan: CandidatePlan, tmp_path: Path) -> None:
    other = plan.model_copy(update={"random_seed": 99})
    first = execute_candidate(plan, default_registry(), ArtifactStore(tmp_path / "a"), "r")
    second = execute_candidate(other, default_registry(), ArtifactStore(tmp_path / "b"), "r")
    assert first.metrics != second.metrics


def test_one_methodology_runs_unchanged_against_real_and_virtual_sources(
    real_source: RealDataSource, virtual_source: VirtualDataSource, tmp_path: Path
) -> None:
    """The split in CandidatePlan is load-bearing precisely because of this."""
    shared = methodology()
    real = execute_candidate(
        CandidatePlan(
            candidate_id="real", data_source=real_source, methodology=shared, random_seed=7
        ),
        default_registry(),
        ArtifactStore(tmp_path / "a"),
        "r",
    )
    virtual = execute_candidate(
        CandidatePlan(
            candidate_id="virtual", data_source=virtual_source, methodology=shared, random_seed=7
        ),
        default_registry(),
        ArtifactStore(tmp_path / "b"),
        "r",
    )
    assert not real.failed and not virtual.failed
    assert real.metrics == virtual.metrics


def test_presence_only_data_runs_through_the_background_stage(
    predictor_paths: list[str], store: ArtifactStore
) -> None:
    source = VirtualDataSource(
        predictor_paths=predictor_paths,
        true_coefficients=[1.6, -0.9],
        occurrence_type="presence_only",
        n_presences=120,
    )
    plan = CandidatePlan(
        candidate_id="po",
        data_source=source,
        methodology=methodology(background="random"),
        random_seed=7,
    )
    result = execute_candidate(plan, default_registry(), store, "run1")
    assert not result.failed, result.failure_reason
    assert "background" in [record.stage for record in result.step_records]
    assert result.metrics["n_background"] > 0


def test_an_unresolved_method_fails_the_candidate_and_records_a_gap(
    virtual_source: VirtualDataSource, store: ArtifactStore
) -> None:
    plan = CandidatePlan(
        candidate_id="gap",
        data_source=virtual_source,
        methodology=methodology(algorithm="maxent"),
        random_seed=7,
    )
    result = execute_candidate(plan, default_registry(), store, "run1")
    assert result.failed
    assert [gap.requested_value for gap in result.plan.unsupported_steps] == ["maxent"]
    assert result.step_records[-1].stage == "algorithm"
    assert result.step_records[-1].backend == "<unresolved>"


def test_gaps_from_a_failed_candidate_reach_the_run_level_backlog(
    virtual_source: VirtualDataSource, store: ArtifactStore
) -> None:
    """Failed candidates are the whole input to the capability backlog."""
    plans = [
        CandidatePlan(
            candidate_id=f"c{index}",
            data_source=virtual_source,
            methodology=methodology(algorithm=algorithm),
            random_seed=7,
        )
        for index, algorithm in enumerate(("logistic", "maxent", "gam"))
    ]
    result = execute_run(plans, default_registry(), store, run_id="run1")
    assert sorted(gap.requested_value or "" for gap in result.unsupported_steps) == [
        "gam",
        "maxent",
    ]
    assert sum(not candidate.failed for candidate in result.candidates) == 1


def test_an_unreadable_data_source_fails_the_candidate_rather_than_the_run(
    tmp_path: Path, store: ArtifactStore
) -> None:
    raster = tmp_path / "bio1.tif"
    raster.write_bytes(b"II*\0")
    plan = CandidatePlan(
        candidate_id="bad",
        data_source=VirtualDataSource(predictor_paths=[str(raster)], true_coefficients=[1.0]),
        methodology=methodology(),
        random_seed=7,
    )
    result = execute_candidate(plan, default_registry(), store, "run1")
    assert result.failed
    assert result.plan.unsupported_steps[0].code == "unreadable_raster"


def test_a_backend_raising_fails_only_its_candidate(
    plan: CandidatePlan, store: ArtifactStore
) -> None:
    registry = default_registry()

    class Exploding:
        name = "python-explode"
        stage = "algorithm"
        methods = ("explode",)
        runtime = "python"

        def run(self, spec, inputs, random_seed):  # type: ignore[no-untyped-def]
            raise RuntimeError("numerical meltdown")

    registry.register(Exploding())
    exploding = plan.model_copy(update={"methodology": methodology(algorithm="explode")}, deep=True)
    result = execute_candidate(exploding, registry, store, "run1")
    assert result.failed
    assert "numerical meltdown" in (result.failure_reason or "")
    assert not result.plan.unsupported_steps  # a crash is not a capability gap


def test_a_plan_carrying_gaps_is_never_executed(
    virtual_source: VirtualDataSource, store: ArtifactStore
) -> None:
    from auto_sdm import UnsupportedStep

    plan = CandidatePlan(
        candidate_id="prescreened",
        data_source=virtual_source,
        methodology=methodology(),
        unsupported_steps=[
            UnsupportedStep(code="unsupported_algorithm", message="nope", feature="algorithm")
        ],
    )
    result = execute_candidate(plan, default_registry(), store, "run1")
    assert result.failed
    assert result.step_records == []


def test_a_run_writes_a_reproducible_manifest(plan: CandidatePlan, store: ArtifactStore) -> None:
    result = execute_run([plan], default_registry(), store, run_id="run1")
    assert result.manifest.is_reproducible
    assert (store.root / "runs" / "run1" / "manifest.json").exists()
    assert (store.root / "runs" / "run1" / "result.json").exists()


def test_spatial_and_random_cv_disagree_on_the_same_candidate(
    virtual_source: VirtualDataSource, tmp_path: Path
) -> None:
    """The reason the CV design is part of the search space rather than a fixed default."""
    scores = {}
    for split in ("random_kfold", "spatial_block"):
        plan = CandidatePlan(
            candidate_id=split,
            data_source=virtual_source,
            methodology=methodology(algorithm="gbm", split=split),
            random_seed=7,
        )
        result = execute_candidate(plan, default_registry(), ArtifactStore(tmp_path / split), "r")
        assert not result.failed, result.failure_reason
        scores[split] = result.metrics["auc_mean"]
    assert scores["random_kfold"] != scores["spatial_block"]


def test_assumptions_are_recorded_rather_than_applied_silently(
    predictor_paths: list[str], store: ArtifactStore
) -> None:
    """Asking for more background than the extent holds is a capped, recorded fallback."""
    source = VirtualDataSource(
        predictor_paths=predictor_paths,
        true_coefficients=[1.6, -0.9],
        occurrence_type="presence_only",
        n_presences=60,
    )
    spec = methodology(background="random")
    spec = spec.model_copy(
        update={"background": MethodSpec(method="random", params={"n_background": 999_999})}
    )
    plan = CandidatePlan(candidate_id="capped", data_source=source, methodology=spec, random_seed=7)
    result = execute_candidate(plan, default_registry(), store, "run1")
    background = next(r for r in result.step_records if r.stage == "background")
    assert [a.code for a in background.assumptions] == ["background_capped_by_extent"]


@pytest.mark.parametrize("algorithm", ["logistic", "glm", "gbm", "brt"])
def test_every_registered_algorithm_alias_executes(
    algorithm: str, virtual_source: VirtualDataSource, tmp_path: Path
) -> None:
    plan = CandidatePlan(
        candidate_id=algorithm,
        data_source=virtual_source,
        methodology=methodology(algorithm=algorithm),
        random_seed=7,
    )
    result = execute_candidate(plan, default_registry(), ArtifactStore(tmp_path / algorithm), "r")
    assert not result.failed, result.failure_reason
    assert np.isfinite(result.metrics["auc_mean"])
