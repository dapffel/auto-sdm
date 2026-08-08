"""The data profile.

Two properties matter more than any individual number. It must be *deterministic*, or
calibration cannot attribute regret to the selector rather than to a profiler that
drifted. And it must *discriminate* — a profile that reports the same thing for clustered
and dispersed sampling is not a join key for anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from auto_sdm import (
    DataProfile,
    MethodSpec,
    VirtualDataSource,
    build_profile,
    default_registry,
    load_data_source,
    profile_context,
)

GRID = 60


def context_from(source: VirtualDataSource, seed: int = 7) -> dict:
    context, _ = load_data_source(source, random_seed=seed)
    return context


def synthetic_context(
    coords: np.ndarray, labels: np.ndarray, grids: np.ndarray | None = None
) -> dict:
    """A context assembled directly, so a test can control the spatial arrangement."""
    if grids is None:
        rows, cols = np.mgrid[0:GRID, 0:GRID]
        grids = np.stack([rows / GRID, np.sin(cols / 8.0)])
    return {
        "grids": grids,
        "predictor_names": [f"p{i}" for i in range(grids.shape[0])],
        "coords": coords,
        "labels": labels,
        "dates": None,
        "transform": None,
        "crs": None,
        "occurrence_type": "presence_absence",
    }


# -- determinism ----------------------------------------------------------------------


def test_the_profile_is_deterministic_given_the_seed(virtual_source: VirtualDataSource) -> None:
    """The whole reason measurement is separated from interpretation."""
    context = context_from(virtual_source)
    first, _ = build_profile(context, random_seed=3)
    second, _ = build_profile(context, random_seed=3)
    assert first == second


def test_the_profile_is_serialisable(virtual_source: VirtualDataSource) -> None:
    profile, _ = build_profile(context_from(virtual_source), random_seed=3)
    assert DataProfile.model_validate_json(profile.model_dump_json()) == profile


# -- volume ---------------------------------------------------------------------------


def test_volume_counts_records_presences_and_duplicates() -> None:
    coords = np.array([[5, 5], [5, 5], [6, 6], [7, 7]])
    labels = np.array([1, 1, 1, 0])
    profile, _ = build_profile(synthetic_context(coords, labels))
    assert profile.volume.n_records == 4
    assert profile.volume.n_unique_cells == 3
    assert profile.volume.n_presences == 3
    assert profile.volume.n_absences == 1
    assert profile.volume.prevalence == pytest.approx(0.75)
    assert profile.volume.duplicate_fraction == pytest.approx(0.25)


def test_prevalence_is_none_for_presence_only_rather_than_one(
    predictor_paths: list[str],
) -> None:
    """Presence-only prevalence is unknowable; reporting 1.0 would be a fabricated fact."""
    source = VirtualDataSource(
        predictor_paths=predictor_paths,
        true_coefficients=[1.0, 0.0],
        occurrence_type="presence_only",
        n_presences=40,
    )
    profile, _ = build_profile(context_from(source))
    assert profile.volume.prevalence is None


def test_small_samples_are_flagged(predictor_paths: list[str]) -> None:
    source = VirtualDataSource(
        predictor_paths=predictor_paths,
        true_coefficients=[1.0, 0.0],
        occurrence_type="presence_only",
        n_presences=30,
    )
    profile, _ = build_profile(context_from(source))
    assert profile.is_small_sample


# -- spatial --------------------------------------------------------------------------


def test_clustered_and_dispersed_sampling_are_distinguished() -> None:
    """If this fails the profile cannot key a prior about sampling design."""
    rng = np.random.default_rng(0)
    clustered = np.stack([rng.integers(0, 6, size=80), rng.integers(0, 6, size=80)], axis=1)
    dispersed = np.stack([rng.integers(0, GRID, size=80), rng.integers(0, GRID, size=80)], axis=1)
    labels = np.ones(80, dtype=int)

    clustered_profile, _ = build_profile(synthetic_context(clustered, labels))
    dispersed_profile, _ = build_profile(synthetic_context(dispersed, labels))
    assert clustered_profile.spatial.nearest_neighbour_index < 0.8
    assert clustered_profile.is_spatially_clustered
    assert not dispersed_profile.is_spatially_clustered
    assert (
        clustered_profile.spatial.mean_nearest_neighbour_cells
        < dispersed_profile.spatial.mean_nearest_neighbour_cells
    )


def test_a_smoother_landscape_has_a_longer_autocorrelation_range() -> None:
    """The measurement that should set spatial block size."""
    rows, cols = np.mgrid[0:GRID, 0:GRID]
    rng = np.random.default_rng(0)
    smooth = np.stack([rows / GRID, cols / GRID])
    rough = np.stack([rng.standard_normal((GRID, GRID)), rng.standard_normal((GRID, GRID))])
    coords = np.stack([rng.integers(0, GRID, size=50), rng.integers(0, GRID, size=50)], axis=1)
    labels = np.ones(50, dtype=int)

    smooth_profile, _ = build_profile(synthetic_context(coords, labels, smooth))
    rough_profile, _ = build_profile(synthetic_context(coords, labels, rough))
    assert (
        smooth_profile.spatial.autocorrelation_range_cells
        > rough_profile.spatial.autocorrelation_range_cells
    )
    assert rough_profile.spatial.autocorrelation_range_bounded


def test_a_landscape_wide_trend_is_reported_as_a_lower_bound() -> None:
    """A monotonic gradient has no sill inside the extent; the range is not a range."""
    rows, cols = np.mgrid[0:GRID, 0:GRID]
    trend = np.stack([rows / GRID, cols / GRID])
    coords = np.stack([np.arange(0, GRID, 2), np.arange(0, GRID, 2)], axis=1)
    profile, assumptions = build_profile(
        synthetic_context(coords, np.ones(coords.shape[0], dtype=int), trend)
    )
    assert not profile.spatial.autocorrelation_range_bounded
    assert any(a.code == "autocorrelation_range_unbounded" for a in assumptions)


def test_block_size_never_exceeds_what_the_extent_can_hold() -> None:
    """An unclamped range on a trended landscape yields one block and no folds at all."""
    rows, cols = np.mgrid[0:GRID, 0:GRID]
    trend = np.stack([rows / GRID, cols / GRID])
    coords = np.stack([np.arange(0, GRID, 2), np.arange(0, GRID, 2)], axis=1)
    profile, _ = build_profile(
        synthetic_context(coords, np.ones(coords.shape[0], dtype=int), trend)
    )
    assert profile.spatial.autocorrelation_range_cells > GRID
    assert profile.suggested_block_size() == GRID // 4
    assert profile.suggested_block_size(min_blocks_per_side=6) == GRID // 6


def test_coverage_reflects_how_much_of_the_extent_is_occupied() -> None:
    rng = np.random.default_rng(1)
    coords = np.stack([rng.integers(0, GRID, size=100), rng.integers(0, GRID, size=100)], axis=1)
    profile, _ = build_profile(synthetic_context(coords, np.ones(100, dtype=int)))
    assert profile.spatial.extent_cells == GRID * GRID
    assert 0 < profile.spatial.coverage_fraction < 0.05


# -- sampling -------------------------------------------------------------------------


def test_biased_sampling_shows_covariate_shift() -> None:
    """The direct signal for whether importance weighting is worth its cost."""
    rng = np.random.default_rng(2)
    labels = np.ones(120, dtype=int)
    # Confined to the top of the grid, where the first predictor (row/GRID) is smallest.
    biased = np.stack([rng.integers(0, 8, size=120), rng.integers(0, GRID, size=120)], axis=1)
    representative = np.stack(
        [rng.integers(0, GRID, size=120), rng.integers(0, GRID, size=120)], axis=1
    )
    biased_profile, _ = build_profile(synthetic_context(biased, labels))
    even_profile, _ = build_profile(synthetic_context(representative, labels))
    assert biased_profile.sampling.covariate_shift > even_profile.sampling.covariate_shift
    assert biased_profile.sampling.environmental_coverage < (
        even_profile.sampling.environmental_coverage
    )


# -- environmental --------------------------------------------------------------------


def test_collinear_predictors_are_detected() -> None:
    rows, _ = np.mgrid[0:GRID, 0:GRID]
    rng = np.random.default_rng(3)
    correlated = np.stack([rows / GRID, rows / GRID + 0.001 * rng.standard_normal((GRID, GRID))])
    independent = np.stack([rng.standard_normal((GRID, GRID)), rng.standard_normal((GRID, GRID))])
    coords = np.stack([rng.integers(0, GRID, size=60), rng.integers(0, GRID, size=60)], axis=1)
    labels = np.ones(60, dtype=int)

    collinear_profile, _ = build_profile(synthetic_context(coords, labels, correlated))
    independent_profile, _ = build_profile(synthetic_context(coords, labels, independent))
    assert collinear_profile.environmental.max_abs_correlation > 0.99
    assert collinear_profile.environmental.max_vif > independent_profile.environmental.max_vif
    # Two near-identical predictors carry about one predictor's worth of information.
    assert collinear_profile.environmental.effective_dimensionality < 1.2
    assert independent_profile.environmental.effective_dimensionality > 1.8


def test_nodata_records_are_excluded_and_recorded() -> None:
    rows, cols = np.mgrid[0:GRID, 0:GRID]
    grids = np.stack([rows / GRID, np.sin(cols / 8.0)])
    grids[:, 0:5, :] = np.nan
    coords = np.array([[0, 0], [1, 1], [30, 30], [31, 31], [32, 32]])
    labels = np.ones(5, dtype=int)
    _, assumptions = build_profile(synthetic_context(coords, labels, grids))
    assert "profile_excluded_nodata_records" in [a.code for a in assumptions]


# -- temporal -------------------------------------------------------------------------


def test_no_dates_means_no_temporal_profile(virtual_source: VirtualDataSource) -> None:
    profile, _ = build_profile(context_from(virtual_source))
    assert profile.temporal is None


def test_the_temporal_span_is_measured() -> None:
    coords = np.stack([np.arange(4), np.arange(4)], axis=1)
    context = synthetic_context(coords, np.ones(4, dtype=int))
    context["dates"] = np.array(
        ["2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01"], dtype="datetime64[D]"
    )
    profile, _ = build_profile(context)
    assert profile.temporal is not None
    assert profile.temporal.span_days == 1096
    assert profile.temporal.first_date == "2018-01-01"
    assert profile.temporal.n_dated_records == 4


def test_a_short_temporal_window_is_visible() -> None:
    coords = np.stack([np.arange(3), np.arange(3)], axis=1)
    context = synthetic_context(coords, np.ones(3, dtype=int))
    context["dates"] = np.array(["2020-06-01", "2020-06-03", "2020-06-05"], dtype="datetime64[D]")
    profile, _ = build_profile(context)
    assert profile.temporal is not None
    assert profile.temporal.span_days == 4


def test_unparseable_dates_leave_no_temporal_profile_but_are_recorded() -> None:
    coords = np.stack([np.arange(2), np.arange(2)], axis=1)
    context = synthetic_context(coords, np.ones(2, dtype=int))
    context["dates"] = np.array(["NaT", "NaT"], dtype="datetime64[D]")
    profile, assumptions = build_profile(context)
    assert profile.temporal is None
    assert "no_usable_dates" in [a.code for a in assumptions]


# -- registry integration -------------------------------------------------------------


def test_the_profiler_resolves_through_the_registry(virtual_source: VirtualDataSource) -> None:
    """Profilers are swappable capabilities, not one hardcoded implementation."""
    profile, _ = profile_context(context_from(virtual_source), default_registry(), random_seed=3)
    assert isinstance(profile, DataProfile)


def test_an_unknown_profiler_is_a_gap_not_a_crash(virtual_source: VirtualDataSource) -> None:
    profile, _ = profile_context(
        context_from(virtual_source), default_registry(), MethodSpec(method="deep_profiler")
    )
    assert profile is None


def test_profiling_works_on_real_ingested_data(real_source) -> None:
    context, _ = load_data_source(real_source, random_seed=7)
    profile, _ = build_profile(context, random_seed=3)
    assert profile.volume.n_records > 0
    assert profile.environmental.n_predictors == 2
