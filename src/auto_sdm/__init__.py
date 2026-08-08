"""auto-sdm — autonomous methodology search for species distribution models."""

from .backends import Backend, BackendRegistry, Resolution
from .builtin import BUILTIN_BACKENDS, default_registry
from .executor import execute_candidate, execute_run
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
from .profile import (
    DataProfile,
    EnvironmentalProfile,
    SamplingProfile,
    SpatialProfile,
    TemporalProfile,
    VolumeProfile,
    build_profile,
    profile_context,
)
from .sources import UnsupportedDataSource, load_data_source
from .store import ArtifactStore

__all__ = [
    "BUILTIN_BACKENDS",
    "SCHEMA_VERSION",
    "Artifact",
    "ArtifactStore",
    "Assumption",
    "Backend",
    "BackendRegistry",
    "DataProfile",
    "EnvironmentalProfile",
    "SamplingProfile",
    "SpatialProfile",
    "TemporalProfile",
    "VolumeProfile",
    "build_profile",
    "profile_context",
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
    "UnsupportedDataSource",
    "UnsupportedStep",
    "VirtualDataSource",
    "default_registry",
    "execute_candidate",
    "execute_run",
    "load_data_source",
]
