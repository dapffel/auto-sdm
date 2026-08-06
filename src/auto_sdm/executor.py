"""The executor.

Walks a candidate's stages in declaration order, resolving each against the registry and
threading one stage's outputs into the next stage's inputs. It knows nothing about what
any stage means — that is the point. Adding a method is registering a backend, never
editing this file.

Two behaviours here are load-bearing rather than incidental:

* An unresolved method ends the candidate and is recorded as a typed ``UnsupportedStep``
  on the returned plan, so ``RunResult.unsupported_steps`` aggregates it into the
  capability backlog. It is never skipped and never substituted.
* Every array a stage emits is persisted content-addressed. Because a stage that leaves
  the data alone re-emits identical bytes, "this stage changed nothing" becomes a
  checkable fact about the run record instead of a claim in prose.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .backends import BackendRegistry
from .digest import digest_obj
from .models import (
    Artifact,
    CandidatePlan,
    CandidateResult,
    Objective,
    RunManifest,
    RunResult,
    StepRecord,
    UnsupportedStep,
)
from .sources import UnsupportedDataSource, load_data_source
from .store import ArtifactStore

# A stage returns its outputs under arbitrary keys; this one is lifted into the step
# record instead of being threaded onward, so metrics never masquerade as data.
METRICS_KEY = "metrics"


def _coerce_metrics(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): float(value) for key, value in raw.items()}


def _persist(store: ArtifactStore, stage: str, outputs: dict[str, Any]) -> list[Artifact]:
    artifacts = []
    for key, value in sorted(outputs.items()):
        if isinstance(value, np.ndarray):
            artifacts.append(store.put_array(f"{stage}.{key}", value))
    return artifacts


def _failed(
    plan: CandidatePlan,
    records: list[StepRecord],
    reason: str,
    gaps: list[UnsupportedStep] | None = None,
) -> CandidateResult:
    """A candidate that stopped early, with any gaps attached to the plan itself.

    The plan is where gaps live, so a failed candidate still feeds the backlog.
    """
    if gaps:
        plan = plan.model_copy(
            update={"unsupported_steps": [*plan.unsupported_steps, *gaps]},
        )
    return CandidateResult(plan=plan, step_records=records, failed=True, failure_reason=reason)


def execute_candidate(
    plan: CandidatePlan,
    registry: BackendRegistry,
    store: ArtifactStore,
    run_id: str,
) -> CandidateResult:
    """Run one candidate end to end. Never raises for an ordinary execution failure."""
    records: list[StepRecord] = []

    if not plan.is_executable:
        return _failed(plan, records, "plan carries unsupported steps and was not executed")

    started = time.perf_counter()
    try:
        context, load_assumptions = load_data_source(plan.data_source, plan.random_seed)
    except UnsupportedDataSource as exc:
        return _failed(plan, records, exc.gap.message, [exc.gap])

    # Loading gets its own step record. Reprojections and dropped records are decisions
    # about the user's data, and burying them in the first methodology stage would make
    # "how many of my occurrences actually made it in" unanswerable from the run record.
    records.append(
        StepRecord(
            stage="data_source",
            candidate_id=plan.candidate_id,
            method=plan.data_source.kind,
            backend="python-source-loader",
            inputs_digest=digest_obj(plan.data_source),
            artifacts=_persist(store, "data_source", context),
            metrics={"n_records": float(context["labels"].size)},
            assumptions=list(load_assumptions),
            random_seed=plan.random_seed,
            duration_seconds=time.perf_counter() - started,
        )
    )
    store.write_step_record(run_id, 0, records[-1])

    metrics: dict[str, float] = {"n_records": float(context["labels"].size)}

    for offset, (stage, spec) in enumerate(plan.methodology.steps()):
        index = offset + 1
        resolution = registry.resolve(stage, spec)
        if resolution.backend is None:
            gap = resolution.gap
            assert gap is not None
            records.append(
                StepRecord(
                    stage=stage,
                    candidate_id=plan.candidate_id,
                    method=spec.method,
                    backend="<unresolved>",
                    params=spec.params,
                    inputs_digest=digest_obj(context),
                    unsupported_steps=[gap],
                    random_seed=plan.random_seed,
                    duration_seconds=0.0,
                )
            )
            store.write_step_record(run_id, index, records[-1])
            return _failed(plan, records, gap.message, [gap])

        backend = resolution.backend
        inputs_digest = digest_obj(context)
        started = time.perf_counter()
        try:
            outputs, assumptions = backend.run(spec, context, plan.random_seed)
        except Exception as exc:  # a backend blowing up fails its candidate, not the run
            elapsed = time.perf_counter() - started
            records.append(
                StepRecord(
                    stage=stage,
                    candidate_id=plan.candidate_id,
                    method=spec.method,
                    backend=backend.name,
                    params=spec.params,
                    inputs_digest=inputs_digest,
                    random_seed=plan.random_seed,
                    duration_seconds=elapsed,
                )
            )
            store.write_step_record(run_id, index, records[-1])
            return _failed(
                plan, records, f"{stage}/{spec.method} raised {type(exc).__name__}: {exc}"
            )
        elapsed = time.perf_counter() - started

        stage_metrics = _coerce_metrics(outputs.pop(METRICS_KEY, None))
        metrics.update(stage_metrics)
        context.update(outputs)

        records.append(
            StepRecord(
                stage=stage,
                candidate_id=plan.candidate_id,
                method=spec.method,
                backend=backend.name,
                params=spec.params,
                rationale=spec.source,
                inputs_digest=inputs_digest,
                artifacts=_persist(store, stage, outputs),
                metrics=stage_metrics,
                assumptions=list(assumptions),
                random_seed=plan.random_seed,
                duration_seconds=elapsed,
            )
        )
        store.write_step_record(run_id, index, records[-1])

    return CandidateResult(plan=plan, step_records=records, metrics=metrics)


def execute_run(
    plans: list[CandidatePlan],
    registry: BackendRegistry,
    store: ArtifactStore,
    *,
    run_id: str,
    objective: Objective = "spatial_transfer",
    random_seed: int = 42,
    package_versions: dict[str, str] | None = None,
) -> RunResult:
    """Execute every candidate and write the run record. Selection is not done here."""
    manifest = RunManifest(
        run_id=run_id,
        objective=objective,
        data_digest=digest_obj([plan.data_source for plan in plans]),
        package_versions=package_versions or _default_versions(),
        random_seed=random_seed,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    store.write_manifest(manifest)

    results = [execute_candidate(plan, registry, store, run_id) for plan in plans]
    result = RunResult(manifest=manifest, candidates=results)
    store.write_result(result)
    return result


def _default_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for package in ("auto-sdm", "numpy", "scikit-learn", "pydantic"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:  # pragma: no cover - depends on the environment
            continue
    return versions


__all__ = ["execute_candidate", "execute_run"]
