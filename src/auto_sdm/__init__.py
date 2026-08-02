"""auto-sdm — autonomous methodology search for species distribution models."""

from .backends import Backend, BackendRegistry, Resolution
from .models import (
    SCHEMA_VERSION,
    Artifact,
    Assumption,
    CandidatePlan,
    CandidateResult,
    DataSource,
    MethodologySpec,
    MethodSpec,
    Objective,
    OccurrenceType,
    RealDataSource,
    RunManifest,
    RunResult,
    SelectionScore,
    StepRecord,
    UnsupportedStep,
    VirtualDataSource,
)

__all__ = [
    "SCHEMA_VERSION",
    "Artifact",
    "Assumption",
    "Backend",
    "BackendRegistry",
    "CandidatePlan",
    "CandidateResult",
    "DataSource",
    "MethodSpec",
    "MethodologySpec",
    "Objective",
    "OccurrenceType",
    "RealDataSource",
    "Resolution",
    "RunManifest",
    "RunResult",
    "SelectionScore",
    "StepRecord",
    "UnsupportedStep",
]
