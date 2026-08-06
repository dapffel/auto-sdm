from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from auto_sdm import RealDataSource, UnsupportedDataSource, VirtualDataSource, load_data_source
from auto_sdm.sources import CONTEXT_KEYS


def test_ground_truth_never_reaches_the_context(virtual_source: VirtualDataSource) -> None:
    """The load-bearing invariant: everything downstream reads this context."""
    context, _ = load_data_source(virtual_source, random_seed=1)
    flattened = repr(sorted(context)) + repr(
        [value for value in context.values() if not isinstance(value, np.ndarray)]
    )
    assert "coefficient" not in flattened
    assert "true_" not in flattened
    assert set(context) == set(CONTEXT_KEYS)


def test_the_virtual_species_reflects_its_niche(virtual_source: VirtualDataSource) -> None:
    """Positive coefficient on 'temp' should put presences at higher rows."""
    context, _ = load_data_source(virtual_source, random_seed=1)
    coords, labels = context["coords"], context["labels"]
    assert coords[labels == 1][:, 0].mean() > coords[labels == 0][:, 0].mean()


def test_generation_is_deterministic_given_the_seed(virtual_source: VirtualDataSource) -> None:
    first, _ = load_data_source(virtual_source, random_seed=3)
    second, _ = load_data_source(virtual_source, random_seed=3)
    third, _ = load_data_source(virtual_source, random_seed=4)
    assert np.array_equal(first["coords"], second["coords"])
    assert not np.array_equal(first["coords"], third["coords"])


def test_presence_only_generates_no_absences(predictor_paths: list[str]) -> None:
    source = VirtualDataSource(
        predictor_paths=predictor_paths,
        true_coefficients=[1.0, 0.0],
        occurrence_type="presence_only",
        n_presences=50,
    )
    context, _ = load_data_source(source, random_seed=1)
    assert set(np.unique(context["labels"])) == {1}


def test_real_and_virtual_contexts_have_the_same_shape(
    real_source: RealDataSource, virtual_source: VirtualDataSource
) -> None:
    """One methodology runs against either source only if the contexts agree."""
    real, _ = load_data_source(real_source, random_seed=7)
    virtual, _ = load_data_source(virtual_source, random_seed=7)
    assert set(real) == set(virtual)
    assert np.array_equal(real["coords"], virtual["coords"])
    assert np.array_equal(real["labels"], virtual["labels"])


def test_a_format_with_no_reader_is_a_typed_gap(tmp_path: Path) -> None:
    raster = tmp_path / "bio1.asc"
    raster.write_text("ncols 2\nnrows 2\n1 2\n3 4\n")
    source = RealDataSource(
        occurrences_path=str(tmp_path / "nope.csv"),
        predictor_paths=[str(raster)],
        occurrence_type="presence_absence",
    )
    with pytest.raises(UnsupportedDataSource) as caught:
        load_data_source(source, random_seed=1)
    assert caught.value.gap.code == "unsupported_predictor_format"
    assert caught.value.gap.requested_value == ".asc"


def test_a_corrupt_raster_is_a_typed_gap_not_a_crash(tmp_path: Path) -> None:
    """A truncated download is the same kind of event as a missing reader."""
    pytest.importorskip("rasterio")
    raster = tmp_path / "bio1.tif"
    raster.write_bytes(b"II*\0")
    source = RealDataSource(
        occurrences_path=str(tmp_path / "nope.csv"),
        predictor_paths=[str(raster)],
        occurrence_type="presence_absence",
    )
    with pytest.raises(UnsupportedDataSource) as caught:
        load_data_source(source, random_seed=1)
    assert caught.value.gap.code == "unreadable_raster"


def test_mixing_georeferenced_and_bare_predictors_is_a_typed_gap(
    tmp_path: Path, predictor_paths: list[str], geotiff_paths: list[str]
) -> None:
    source = VirtualDataSource(
        predictor_paths=[geotiff_paths[0], predictor_paths[1]], true_coefficients=[1.0, 1.0]
    )
    with pytest.raises(UnsupportedDataSource) as caught:
        load_data_source(source, random_seed=1)
    assert caught.value.gap.code == "mixed_predictor_georeferencing"


def test_misaligned_predictor_grids_are_a_typed_gap(tmp_path: Path) -> None:
    first, second = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(first, np.zeros((10, 10)))
    np.save(second, np.zeros((10, 12)))
    source = VirtualDataSource(
        predictor_paths=[str(first), str(second)], true_coefficients=[1.0, 1.0]
    )
    with pytest.raises(UnsupportedDataSource) as caught:
        load_data_source(source, random_seed=1)
    assert caught.value.gap.code == "misaligned_predictors"


def test_an_unbuilt_sampling_bias_is_a_gap_not_an_ignored_field(
    predictor_paths: list[str],
) -> None:
    """Silently ignoring the field would make a biased benchmark look unbiased."""
    source = VirtualDataSource(
        predictor_paths=predictor_paths,
        true_coefficients=[1.0, 0.0],
        sampling_bias="road_proximity",
    )
    with pytest.raises(UnsupportedDataSource) as caught:
        load_data_source(source, random_seed=1)
    assert caught.value.gap.code == "unsupported_sampling_bias"
    assert caught.value.gap.requested_value == "road_proximity"
