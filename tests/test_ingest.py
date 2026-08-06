"""Real user data: GeoTIFF predictors, lon/lat occurrences, CRS, and dates.

The recurring theme is that records must never vanish quietly. A user whose CRS is wrong
should see that stated in the run record, not a clean run over whatever 3% of their data
happened to land on the grid.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from auto_sdm import ArtifactStore, CandidatePlan, RealDataSource, load_data_source
from auto_sdm.rasters import read_predictors

pytest.importorskip("rasterio")

# geotiff_paths writes one degree per cell from (10E, 50N) running south and east.
ORIGIN_LON, ORIGIN_LAT = 10.0, 50.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def cell_centre(row: int, col: int) -> tuple[float, float]:
    return ORIGIN_LON + col + 0.5, ORIGIN_LAT - row - 0.5


def source_for(occurrences: str, geotiff_paths: list[str], **kwargs: object) -> RealDataSource:
    return RealDataSource(
        occurrences_path=occurrences,
        predictor_paths=geotiff_paths,
        occurrence_type="presence_absence",
        **kwargs,  # type: ignore[arg-type]
    )


def test_geotiffs_carry_their_transform_and_crs(geotiff_paths: list[str]) -> None:
    grids = read_predictors(geotiff_paths)
    assert grids.is_georeferenced
    assert grids.crs == "EPSG:4326"
    assert grids.transform is not None and grids.transform[2] == ORIGIN_LON


def test_bare_npy_predictors_are_not_georeferenced(predictor_paths: list[str]) -> None:
    """An identity transform would silently mean 'cell indices are degrees'."""
    grids = read_predictors(predictor_paths)
    assert not grids.is_georeferenced
    assert grids.transform is None


def test_lon_lat_occurrences_land_on_the_right_cells(
    tmp_path: Path, geotiff_paths: list[str]
) -> None:
    expected = [(3, 7), (20, 41), (0, 0), (59, 59)]
    lon_lat = [cell_centre(row, col) for row, col in expected]
    path = write_csv(
        tmp_path / "occ.csv",
        [
            {"decimalLongitude": lon, "decimalLatitude": lat, "occurrenceStatus": "PRESENT"}
            for lon, lat in lon_lat
        ],
    )
    context, assumptions = load_data_source(source_for(path, geotiff_paths), random_seed=1)
    assert [tuple(pair) for pair in context["coords"]] == expected
    assert assumptions == []


@pytest.mark.parametrize(
    "columns", [("decimalLongitude", "decimalLatitude"), ("longitude", "latitude"), ("lon", "lat")]
)
def test_common_column_spellings_are_recognised(
    tmp_path: Path, geotiff_paths: list[str], columns: tuple[str, str]
) -> None:
    lon, lat = cell_centre(5, 5)
    path = write_csv(tmp_path / "occ.csv", [{columns[0]: lon, columns[1]: lat}])
    context, _ = load_data_source(source_for(path, geotiff_paths), random_seed=1)
    assert [tuple(pair) for pair in context["coords"]] == [(5, 5)]


def test_occurrence_status_strings_become_labels(tmp_path: Path, geotiff_paths: list[str]) -> None:
    rows = []
    for index, status in enumerate(["PRESENT", "ABSENT", "present", "absent"]):
        lon, lat = cell_centre(index, index)
        rows.append({"lon": lon, "lat": lat, "occurrenceStatus": status})
    path = write_csv(tmp_path / "occ.csv", rows)
    context, _ = load_data_source(source_for(path, geotiff_paths), random_seed=1)
    assert list(context["labels"]) == [1, 0, 1, 0]


def test_records_outside_the_extent_are_recorded_not_dropped_silently(
    tmp_path: Path, geotiff_paths: list[str]
) -> None:
    """The defect this ingest work exists to fix."""
    inside = [cell_centre(row, row) for row in range(3)]
    outside = [(-40.0, 12.0), (170.0, -80.0)]
    path = write_csv(
        tmp_path / "occ.csv",
        [{"lon": lon, "lat": lat} for lon, lat in inside + outside],
    )
    context, assumptions = load_data_source(source_for(path, geotiff_paths), random_seed=1)
    assert context["coords"].shape[0] == 3
    dropped = [a for a in assumptions if a.code == "dropped_records_outside_extent"]
    assert len(dropped) == 1
    assert "2 of 5" in dropped[0].message


def test_a_wrong_crs_is_visible_rather_than_quietly_halving_the_data(
    tmp_path: Path, geotiff_paths: list[str]
) -> None:
    """Web-mercator metres read as degrees land nowhere near the extent."""
    path = write_csv(
        tmp_path / "occ.csv",
        [{"lon": 1_113_194.9 + index, "lat": 6_446_275.8} for index in range(10)],
    )
    _, assumptions = load_data_source(
        source_for(path, geotiff_paths, crs="EPSG:4326"), random_seed=1
    )
    assert any(a.code == "dropped_records_outside_extent" for a in assumptions)


def test_occurrences_are_reprojected_when_the_crs_differs(
    tmp_path: Path, geotiff_paths: list[str]
) -> None:
    pytest.importorskip("pyproj")
    from pyproj import Transformer

    target_lon, target_lat = cell_centre(10, 10)
    to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = to_mercator.transform(target_lon, target_lat)
    path = write_csv(tmp_path / "occ.csv", [{"lon": x, "lat": y}])

    context, assumptions = load_data_source(
        source_for(path, geotiff_paths, crs="EPSG:3857"), random_seed=1
    )
    assert [tuple(pair) for pair in context["coords"]] == [(10, 10)]
    assert any(a.code == "reprojected_occurrences" for a in assumptions)


def test_dates_are_parsed_into_the_context(tmp_path: Path, geotiff_paths: list[str]) -> None:
    rows = [
        {"lon": cell_centre(i, i)[0], "lat": cell_centre(i, i)[1], "eventDate": value}
        for i, value in enumerate(["2019-04-01", "2021-07-15T09:30:00Z", "1998"])
    ]
    path = write_csv(tmp_path / "occ.csv", rows)
    context, _ = load_data_source(source_for(path, geotiff_paths), random_seed=1)
    assert list(context["dates"].astype(str)) == ["2019-04-01", "2021-07-15", "1998-01-01"]


def test_unparseable_dates_are_recorded_rather_than_guessed(
    tmp_path: Path, geotiff_paths: list[str]
) -> None:
    rows = [
        {"lon": cell_centre(i, i)[0], "lat": cell_centre(i, i)[1], "eventDate": value}
        for i, value in enumerate(["2019-04-01", "sometime in spring", ""])
    ]
    path = write_csv(tmp_path / "occ.csv", rows)
    context, assumptions = load_data_source(source_for(path, geotiff_paths), random_seed=1)
    assert any(a.code == "dropped_unparseable_dates" for a in assumptions)
    assert np.isnat(context["dates"][1])


def test_unparseable_coordinates_are_recorded(tmp_path: Path, geotiff_paths: list[str]) -> None:
    lon, lat = cell_centre(4, 4)
    path = write_csv(
        tmp_path / "occ.csv",
        [{"lon": lon, "lat": lat}, {"lon": "", "lat": ""}, {"lon": "n/a", "lat": "n/a"}],
    )
    context, assumptions = load_data_source(source_for(path, geotiff_paths), random_seed=1)
    assert context["coords"].shape[0] == 1
    assert any(a.code == "dropped_unparseable_coordinates" for a in assumptions)


def test_lon_lat_against_ungeoreferenced_predictors_is_a_typed_gap(
    tmp_path: Path, predictor_paths: list[str]
) -> None:
    from auto_sdm import UnsupportedDataSource

    path = write_csv(tmp_path / "occ.csv", [{"lon": 10.5, "lat": 49.5}])
    with pytest.raises(UnsupportedDataSource) as caught:
        load_data_source(source_for(path, predictor_paths), random_seed=1)
    assert caught.value.gap.code == "ungeoreferenced_predictors"


def test_unrecognised_columns_are_a_typed_gap(tmp_path: Path, geotiff_paths: list[str]) -> None:
    from auto_sdm import UnsupportedDataSource

    path = write_csv(tmp_path / "occ.csv", [{"easting": 1.0, "northing": 2.0}])
    with pytest.raises(UnsupportedDataSource) as caught:
        load_data_source(source_for(path, geotiff_paths), random_seed=1)
    assert caught.value.gap.code == "unrecognised_occurrence_columns"


def test_nodata_cells_survive_as_nan(tmp_path: Path, geotiff_paths: list[str]) -> None:
    """Nodata must not be read as a real predictor value of zero."""
    import rasterio

    with rasterio.open(geotiff_paths[0], "r+") as handle:
        band = handle.read(1)
        band[0:5, 0:5] = np.nan
        handle.write(band, 1)
    grids = read_predictors(geotiff_paths)
    assert np.isnan(grids.values[0, 0:5, 0:5]).all()
    assert np.isfinite(grids.values[0, 30:, 30:]).all()


def test_a_full_candidate_runs_on_geotiff_and_lon_lat_data(
    tmp_path: Path, geotiff_paths: list[str], predictor_paths: list[str], virtual_source
) -> None:
    """End to end on data shaped the way a user's actually is."""
    from conftest import methodology

    from auto_sdm import default_registry, execute_candidate

    drawn, _ = load_data_source(virtual_source, random_seed=7)
    rows = [
        {
            "decimalLongitude": cell_centre(int(row), int(col))[0],
            "decimalLatitude": cell_centre(int(row), int(col))[1],
            "occurrenceStatus": "PRESENT" if label else "ABSENT",
            "eventDate": "2020-05-01",
        }
        for (row, col), label in zip(drawn["coords"], drawn["labels"])
    ]
    path = write_csv(tmp_path / "occ.csv", rows)
    plan = CandidatePlan(
        candidate_id="geo",
        data_source=source_for(path, geotiff_paths),
        methodology=methodology(),
        random_seed=7,
    )
    result = execute_candidate(plan, default_registry(), ArtifactStore(tmp_path / "store"), "run1")
    assert not result.failed, result.failure_reason
    assert 0.5 < result.metrics["auc_mean"] <= 1.0
    assert result.step_records[0].stage == "data_source"
