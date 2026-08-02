"""The execution contract.

A ``CandidatePlan`` is deliberately split into *where the data comes from* and *what is
done to it*. The same ``MethodologySpec`` therefore runs unchanged against a user's real
dataset or against a virtual species with a known niche — which is what makes selector
calibration possible without a parallel code path.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "0.1"


class FrozenModel(BaseModel):
    """Immutable configuration prevents a plan changing during execution."""

    model_config = ConfigDict(frozen=True)


# --------------------------------------------------------------------------------------
# Visibility records
#
# Nothing is silently substituted. Both carry a stable ``code`` so identical events
# aggregate across runs into a ranked backlog rather than a pile of opaque strings.
# --------------------------------------------------------------------------------------


class Assumption(FrozenModel):
    """A fallback the system made, recorded rather than hidden."""

    code: str
    message: str
    field: str | None = None


class UnsupportedStep(FrozenModel):
    """A method no registered backend can execute.

    ``feature`` names the dimension (algorithm, cv_design, background, ...) so the
    capability backlog is not blind to anything but model choice.
    """

    code: str
    message: str
    feature: str
    requested_value: str | None = None


# --------------------------------------------------------------------------------------
# Data sources
# --------------------------------------------------------------------------------------

OccurrenceType = Literal["presence_only", "presence_absence"]


class RealDataSource(FrozenModel):
    """A user's dataset. The system does not know the truth."""

    kind: Literal["real"] = "real"
    occurrences_path: str
    predictor_paths: list[str] = Field(min_length=1)
    occurrence_type: OccurrenceType
    crs: str = "EPSG:4326"


class VirtualDataSource(FrozenModel):
    """A simulated species with a known niche, generated over *real* predictor rasters
    so calibration inherits genuine collinearity and spatial structure.

    ``true_coefficients`` is the ground truth the selector is scored against; it must
    never be visible to the proposer, executor, or selector.
    """

    kind: Literal["virtual"] = "virtual"
    predictor_paths: list[str] = Field(min_length=1)
    true_coefficients: list[float] = Field(min_length=1)
    true_intercept: float = -0.5
    occurrence_type: OccurrenceType = "presence_absence"
    n_presences: int = Field(default=200, ge=20)
    n_absences: int = Field(default=200, ge=20)
    sampling_bias: str | None = None
    crs: str = "EPSG:4326"

    @model_validator(mode="after")
    def coefficients_match_predictors(self) -> VirtualDataSource:
        if len(self.true_coefficients) != len(self.predictor_paths):
            raise ValueError("true_coefficients must match predictor_paths")
        return self


DataSource = Annotated[RealDataSource | VirtualDataSource, Field(discriminator="kind")]


# --------------------------------------------------------------------------------------
# Methodology
#
# ``method`` is an open string, not a Literal: the search space must be able to grow
# through synthesized backends. Validation happens when the registry resolves it, and an
# unresolvable method becomes an UnsupportedStep rather than a schema error.
# --------------------------------------------------------------------------------------


class MethodSpec(FrozenModel):
    """One methodological choice, resolved against the backend registry at execution."""

    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None  # provenance: DOI or run id this choice came from


class MethodologySpec(FrozenModel):
    """What is done to the data. Backend-agnostic by construction."""

    cleaning: MethodSpec
    accessible_area: MethodSpec
    predictors: MethodSpec
    background: MethodSpec | None = None  # presence-only regimes require one
    split: MethodSpec
    algorithm: MethodSpec

    def steps(self) -> list[tuple[str, MethodSpec]]:
        """Ordered (stage, spec) pairs. The pipeline order is the declaration order."""
        pairs = [
            ("cleaning", self.cleaning),
            ("accessible_area", self.accessible_area),
            ("predictors", self.predictors),
            ("background", self.background),
            ("split", self.split),
            ("algorithm", self.algorithm),
        ]
        return [(name, spec) for name, spec in pairs if spec is not None]


Objective = Literal["spatial_transfer", "interpolation", "temporal_projection"]


class CandidatePlan(FrozenModel):
    """One point in the search space: a data source, a methodology, and a seed."""

    schema_version: str = SCHEMA_VERSION
    candidate_id: str
    data_source: DataSource
    methodology: MethodologySpec
    objective: Objective = "spatial_transfer"
    random_seed: int = 42
    assumptions: list[Assumption] = Field(default_factory=list)
    unsupported_steps: list[UnsupportedStep] = Field(default_factory=list)

    @model_validator(mode="after")
    def presence_only_requires_background(self) -> CandidatePlan:
        if (
            self.data_source.occurrence_type == "presence_only"
            and self.methodology.background is None
        ):
            raise ValueError("presence-only data requires a background strategy")
        return self

    @property
    def is_executable(self) -> bool:
        """A plan carrying unsupported steps is recorded, never executed."""
        return not self.unsupported_steps


# --------------------------------------------------------------------------------------
# Run records
# --------------------------------------------------------------------------------------


class Artifact(FrozenModel):
    """Content-addressed output. ``digest`` makes a rerun's identity checkable."""

    name: str
    path: str
    media_type: str
    digest: str


class StepRecord(FrozenModel):
    """What happened at one stage of one candidate.

    ``inputs_digest`` covers every input the stage consumed. Rewind is only sound if
    this is complete, because it is what decides whether downstream stages invalidate.
    """

    stage: str
    candidate_id: str
    method: str
    backend: str
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None
    inputs_digest: str
    artifacts: list[Artifact] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    assumptions: list[Assumption] = Field(default_factory=list)
    unsupported_steps: list[UnsupportedStep] = Field(default_factory=list)
    random_seed: int
    duration_seconds: float = Field(ge=0)


class SelectionScore(FrozenModel):
    """Why a candidate placed where it did. ``components`` keeps the composite legible
    instead of collapsing it to a single opaque number."""

    candidate_id: str
    total: float
    components: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


class CandidateResult(FrozenModel):
    plan: CandidatePlan
    step_records: list[StepRecord] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    score: SelectionScore | None = None
    failed: bool = False
    failure_reason: str | None = None


class RunManifest(FrozenModel):
    """Run-level provenance. Without this, cross-run learning is learning from noise."""

    schema_version: str = SCHEMA_VERSION
    run_id: str
    objective: Objective
    data_digest: str
    package_versions: dict[str, str] = Field(default_factory=dict)
    random_seed: int = 42
    created_at: str

    @property
    def is_reproducible(self) -> bool:
        return bool(self.data_digest and self.package_versions)


class RunResult(FrozenModel):
    manifest: RunManifest
    candidates: list[CandidateResult] = Field(default_factory=list)
    selected: str | None = None

    @property
    def unsupported_steps(self) -> list[UnsupportedStep]:
        """Every capability gap this run hit — the input to the ranked backlog."""
        return [step for candidate in self.candidates for step in candidate.plan.unsupported_steps]
