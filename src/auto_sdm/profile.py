"""Data profiling.

A methodological prior — *"presence-only, fewer than 50 records, heterogeneous extent →
regularized MaxEnt with target-group background"* — is a rule keyed on characteristics of
the data. Without a structured profile there is nothing to match such a rule against, so
this module is the join key the proposer's priors and its run history both need.

Everything here is **measurement, not judgment**. Given the same data and seed the profile
is identical, because the run record claims reproducibility and calibration must be able
to attribute regret to the selector rather than to a profiler that drifted. Interpreting a
profile into scenario labels is a separate, non-deterministic layer that sits on top of
this one and is recorded as such.

Distances are in *cells*, not metres. Converting would require an equal-area projection
the system does not yet do, and a degree is not a metre anywhere except one line on the
globe; cell units are the honest unit until that exists.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .backends import BackendRegistry
from .models import Assumption, FrozenModel, MethodSpec

PROFILE_STAGE = "profile"


class VolumeProfile(FrozenModel):
    """How much data there is — the single strongest constraint on method choice."""

    n_records: int
    n_unique_cells: int
    n_presences: int
    n_absences: int
    prevalence: float | None = None  # None for presence-only: unknowable, not 1.0
    duplicate_fraction: float


class SpatialProfile(FrozenModel):
    """Where the records are, and how far the landscape is correlated with itself.

    ``autocorrelation_range_cells`` is the principled setting for spatial block size:
    blocks smaller than the correlation range leak between folds, which is precisely the
    inflation spatial CV exists to prevent.

    When the landscape carries a trend across the whole extent the semivariogram never
    reaches a sill inside the study area, and the estimate is a *lower bound* rather than
    a range. ``autocorrelation_range_bounded`` says which of the two it is, because the
    difference decides whether block size can be read off it at all.
    """

    extent_cells: int
    extent_rows: int
    extent_cols: int
    occupied_cells: int
    coverage_fraction: float
    autocorrelation_range_cells: float
    autocorrelation_range_bounded: bool
    nearest_neighbour_index: float  # <1 clustered, ~1 random, >1 dispersed
    mean_nearest_neighbour_cells: float


class SamplingProfile(FrozenModel):
    """How the sample sits relative to what was available.

    ``covariate_shift`` is the direct signal for whether importance weighting matters: it
    is the standardised gap between sampled and landscape predictor means.
    """

    environmental_coverage: float
    covariate_shift: float
    max_univariate_shift: float


class EnvironmentalProfile(FrozenModel):
    """Structure among the predictors themselves."""

    n_predictors: int
    max_abs_correlation: float
    max_vif: float
    effective_dimensionality: float
    niche_breadth: float | None = None  # None without presences to compare against


class TemporalProfile(FrozenModel):
    """When the records are from. Absent unless the source carried dates."""

    n_dated_records: int
    span_days: int
    first_date: str
    last_date: str
    median_records_per_year: float


class DataProfile(FrozenModel):
    """A deterministic characterisation of one dataset."""

    volume: VolumeProfile
    spatial: SpatialProfile
    sampling: SamplingProfile
    environmental: EnvironmentalProfile
    temporal: TemporalProfile | None = None

    @property
    def is_small_sample(self) -> bool:
        """The regime where regularisation and feature restriction start to dominate."""
        return self.volume.n_unique_cells < 50

    @property
    def is_spatially_clustered(self) -> bool:
        return self.spatial.nearest_neighbour_index < 0.8

    def suggested_block_size(self, min_blocks_per_side: int = 4) -> int:
        """Block size at or above the autocorrelation range, capped so folds remain possible.

        The cap is not a nicety. A landscape with a trend across the whole extent has no
        sill inside the study area, and the unclamped range there exceeds the grid — which
        would produce one block, no folds, and no cross-validation at all. Where the cap
        binds, blocks are smaller than the correlation range and spatial CV is only
        partially protective; ``autocorrelation_range_bounded`` is what says so.
        """
        shortest_side = min(self.spatial.extent_rows, self.spatial.extent_cols)
        ceiling = max(1, shortest_side // min_blocks_per_side)
        return max(1, min(int(np.ceil(self.spatial.autocorrelation_range_cells)), ceiling))


# --------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------


def _valid_mask(grids: np.ndarray) -> np.ndarray:
    return np.isfinite(grids).all(axis=0)


def _sample_cells(mask: np.ndarray, rng: np.random.Generator, count: int) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    if rows.size <= count:
        return np.stack([rows, cols], axis=1)
    picked = rng.choice(rows.size, size=count, replace=False)
    return np.stack([rows[picked], cols[picked]], axis=1)


def _variogram_range(
    grids: np.ndarray, rng: np.random.Generator, n_pairs: int = 20_000
) -> tuple[float, bool]:
    """Distance at which the empirical semivariogram reaches most of its sill.

    Computed on the predictors rather than on the labels, so it is defined for
    presence-only data too — and because it is the *landscape's* correlation structure
    that determines how far apart folds must be to be independent.

    Returns the range and whether it is *bounded*: a landscape with a trend across the
    whole extent has no sill inside the study area, and the number is then a lower bound
    on the range rather than an estimate of it.
    """
    mask = _valid_mask(grids)
    sample = _sample_cells(mask, rng, 3_000)
    if sample.shape[0] < 20:
        return 1.0, False

    left = rng.integers(0, sample.shape[0], size=n_pairs)
    right = rng.integers(0, sample.shape[0], size=n_pairs)
    keep = left != right
    left, right = left[keep], right[keep]

    distances = np.linalg.norm((sample[left] - sample[right]).astype(float), axis=1)
    values = grids[:, sample[:, 0], sample[:, 1]]
    standardised = (values - np.nanmean(values, axis=1, keepdims=True)) / np.where(
        np.nanstd(values, axis=1, keepdims=True) == 0,
        1.0,
        np.nanstd(values, axis=1, keepdims=True),
    )
    squared_difference = ((standardised[:, left] - standardised[:, right]) ** 2).mean(axis=0)

    max_distance = float(distances.max())
    if max_distance <= 0:
        return 1.0, False
    edges = np.linspace(0, max_distance, 26)
    which = np.clip(np.digitize(distances, edges) - 1, 0, len(edges) - 2)
    semivariance = np.array(
        [
            squared_difference[which == index].mean() / 2.0 if np.any(which == index) else np.nan
            for index in range(len(edges) - 1)
        ]
    )
    centres = (edges[:-1] + edges[1:]) / 2.0
    usable = np.isfinite(semivariance)
    if usable.sum() < 3:
        return 1.0, False

    sill = float(np.nanmean(semivariance[usable][-5:]))
    if sill <= 0:
        return 1.0, False
    reached = np.nonzero(usable & (semivariance >= 0.95 * sill))[0]
    if not reached.size:
        return max_distance, False
    estimate = float(centres[reached[0]])
    # Reaching the sill only in the last few bins means the curve was still climbing at
    # the edge of the study area: a trend, not a range.
    plateaued = bool(reached[0] < len(centres) - 5)
    return estimate, plateaued


def _nearest_neighbour(coords: np.ndarray, extent_cells: int) -> tuple[float, float]:
    """Mean nearest-neighbour distance and the Clark-Evans index against CSR."""
    n = coords.shape[0]
    if n < 2:
        return 0.0, 1.0
    points = coords.astype(float)
    # n is bounded by occurrence counts, so the dense distance matrix stays affordable.
    deltas = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    observed = float(nearest.mean())
    expected = 0.5 * np.sqrt(extent_cells / n) if extent_cells > 0 else 0.0
    return observed, float(observed / expected) if expected > 0 else 1.0


def _environmental_coverage(sampled: np.ndarray, landscape: np.ndarray, n_bins: int = 10) -> float:
    """Fraction of the landscape's per-predictor quantile bins the sample touches."""
    covered = []
    for index in range(landscape.shape[0]):
        edges = np.nanquantile(landscape[index], np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if edges.size < 3:
            continue
        occupied = np.unique(np.clip(np.digitize(sampled[index], edges) - 1, 0, edges.size - 2))
        covered.append(occupied.size / (edges.size - 1))
    return float(np.mean(covered)) if covered else 0.0


def _collinearity(landscape: np.ndarray) -> tuple[float, float, float]:
    """Max |r|, max VIF, and effective dimensionality from the correlation matrix."""
    k = landscape.shape[0]
    if k < 2:
        return 0.0, 1.0, 1.0
    correlation = np.corrcoef(landscape)
    correlation = np.nan_to_num(correlation, nan=0.0)
    off_diagonal = correlation[~np.eye(k, dtype=bool)]
    max_correlation = float(np.abs(off_diagonal).max())

    try:
        inverse = np.linalg.inv(correlation)
        max_vif = float(np.max(np.abs(np.diag(inverse))))
    except np.linalg.LinAlgError:  # perfectly collinear predictors
        max_vif = float("inf")

    eigenvalues = np.clip(np.linalg.eigvalsh(correlation), 0, None)
    total = eigenvalues.sum()
    effective = float(total**2 / (eigenvalues**2).sum()) if total > 0 else 1.0
    return max_correlation, max_vif, effective


def build_profile(
    context: dict[str, Any], random_seed: int = 42
) -> tuple[DataProfile, list[Assumption]]:
    """Measure one dataset. Deterministic given ``random_seed``."""
    grids = context["grids"]
    coords = context["coords"]
    labels = context["labels"]
    occurrence_type = context["occurrence_type"]
    rng = np.random.default_rng(random_seed)
    assumptions: list[Assumption] = []

    mask = _valid_mask(grids)
    extent_cells = int(mask.sum())
    unique_cells = np.unique(coords, axis=0)
    n_records = int(coords.shape[0])
    n_presences = int((labels == 1).sum())
    n_absences = int((labels == 0).sum())

    volume = VolumeProfile(
        n_records=n_records,
        n_unique_cells=int(unique_cells.shape[0]),
        n_presences=n_presences,
        n_absences=n_absences,
        prevalence=(
            float(n_presences / n_records)
            if occurrence_type == "presence_absence" and n_records
            else None
        ),
        duplicate_fraction=(float(1.0 - unique_cells.shape[0] / n_records) if n_records else 0.0),
    )

    mean_nn, nn_index = _nearest_neighbour(unique_cells, extent_cells)
    acf_range, acf_bounded = _variogram_range(grids, rng)
    if not acf_bounded:
        assumptions.append(
            Assumption(
                code="autocorrelation_range_unbounded",
                message=f"the semivariogram had not reached a sill by {acf_range:.0f} cells; the "
                "landscape carries a trend across the whole extent, so this is a lower bound "
                "and spatial blocks cannot fully decorrelate the folds",
                field="profile.spatial",
            )
        )
    spatial = SpatialProfile(
        extent_cells=extent_cells,
        extent_rows=int(grids.shape[1]),
        extent_cols=int(grids.shape[2]),
        occupied_cells=int(unique_cells.shape[0]),
        coverage_fraction=float(unique_cells.shape[0] / extent_cells) if extent_cells else 0.0,
        autocorrelation_range_cells=acf_range,
        autocorrelation_range_bounded=acf_bounded,
        nearest_neighbour_index=nn_index,
        mean_nearest_neighbour_cells=mean_nn,
    )

    landscape_cells = _sample_cells(mask, rng, 5_000)
    landscape = grids[:, landscape_cells[:, 0], landscape_cells[:, 1]]
    sampled = grids[:, coords[:, 0], coords[:, 1]]
    finite_sampled = sampled[:, np.isfinite(sampled).all(axis=0)]
    if finite_sampled.shape[1] < sampled.shape[1]:
        assumptions.append(
            Assumption(
                code="profile_excluded_nodata_records",
                message=f"{sampled.shape[1] - finite_sampled.shape[1]} records fell on cells "
                "with no predictor value and were excluded from the environmental profile",
                field="profile",
            )
        )

    landscape_mean = np.nanmean(landscape, axis=1)
    landscape_std = np.where(np.nanstd(landscape, axis=1) == 0, 1.0, np.nanstd(landscape, axis=1))
    shift = np.abs(np.nanmean(finite_sampled, axis=1) - landscape_mean) / landscape_std
    sampling = SamplingProfile(
        environmental_coverage=_environmental_coverage(finite_sampled, landscape),
        covariate_shift=float(np.mean(shift)),
        max_univariate_shift=float(np.max(shift)),
    )

    max_correlation, max_vif, effective = _collinearity(landscape)
    presence_values = finite_sampled[:, labels[np.isfinite(sampled).all(axis=0)] == 1]
    environmental = EnvironmentalProfile(
        n_predictors=int(grids.shape[0]),
        max_abs_correlation=max_correlation,
        max_vif=max_vif,
        effective_dimensionality=effective,
        niche_breadth=(
            float(np.mean(np.nanstd(presence_values, axis=1) / landscape_std))
            if presence_values.shape[1] > 1
            else None
        ),
    )

    temporal = _temporal_profile(context.get("dates"), assumptions)
    return (
        DataProfile(
            volume=volume,
            spatial=spatial,
            sampling=sampling,
            environmental=environmental,
            temporal=temporal,
        ),
        assumptions,
    )


def _temporal_profile(
    dates: np.ndarray | None, assumptions: list[Assumption]
) -> TemporalProfile | None:
    if dates is None:
        return None
    usable = dates[~np.isnat(dates)]
    if usable.size == 0:
        assumptions.append(
            Assumption(
                code="no_usable_dates",
                message="a date column was present but no value in it could be parsed",
                field="profile.temporal",
            )
        )
        return None
    span_days = int((usable.max() - usable.min()).astype("timedelta64[D]").astype(int))
    years = max(span_days / 365.25, 1e-9)
    return TemporalProfile(
        n_dated_records=int(usable.size),
        span_days=span_days,
        first_date=str(usable.min()),
        last_date=str(usable.max()),
        median_records_per_year=float(usable.size / years) if span_days else float(usable.size),
    )


# --------------------------------------------------------------------------------------
# Registry backend
#
# Profiling is a registry stage so profilers are swappable and growable capabilities like
# everything else, rather than one hardcoded implementation.
# --------------------------------------------------------------------------------------


class StandardProfiler:
    name = "python-profile-standard"
    stage = PROFILE_STAGE
    methods = ("standard",)
    runtime = "python"

    def run(
        self, spec: MethodSpec, inputs: dict[str, Any], random_seed: int
    ) -> tuple[dict[str, Any], list[Assumption]]:
        profile, assumptions = build_profile(inputs, random_seed)
        return {"profile": profile}, assumptions


def profile_context(
    context: dict[str, Any],
    registry: BackendRegistry,
    spec: MethodSpec | None = None,
    random_seed: int = 42,
) -> tuple[DataProfile | None, list[Assumption]]:
    """Resolve and run a profiler. Returns ``None`` if no backend implements the method."""
    resolution = registry.resolve(PROFILE_STAGE, spec or MethodSpec(method="standard"))
    if resolution.backend is None:
        return None, []
    outputs, assumptions = resolution.backend.run(
        spec or MethodSpec(method="standard"), context, random_seed
    )
    profile = outputs["profile"]
    assert isinstance(profile, DataProfile)
    return profile, assumptions
