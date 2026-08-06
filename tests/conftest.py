from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from auto_sdm import (
    ArtifactStore,
    CandidatePlan,
    MethodologySpec,
    MethodSpec,
    RealDataSource,
    VirtualDataSource,
    load_data_source,
)

GRID_SIZE = 60


@pytest.fixture
def predictor_paths(tmp_path: Path) -> list[str]:
    """Two spatially structured predictor grids.

    Structure matters: a random grid would make spatial block CV and random CV agree,
    which is exactly the difference the design exists to expose.
    """
    rows, cols = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE]
    rng = np.random.default_rng(0)
    layers = {
        "temp": rows / GRID_SIZE + 0.05 * rng.standard_normal((GRID_SIZE, GRID_SIZE)),
        "precip": np.sin(cols / 8.0) + 0.05 * rng.standard_normal((GRID_SIZE, GRID_SIZE)),
    }
    paths = []
    for name, grid in layers.items():
        path = tmp_path / f"{name}.npy"
        np.save(path, grid)
        paths.append(str(path))
    return paths


@pytest.fixture
def virtual_source(predictor_paths: list[str]) -> VirtualDataSource:
    return VirtualDataSource(
        predictor_paths=predictor_paths,
        true_coefficients=[1.6, -0.9],
        n_presences=150,
        n_absences=150,
    )


@pytest.fixture
def geotiff_paths(tmp_path: Path, predictor_paths: list[str]) -> list[str]:
    """The same two layers written as georeferenced GeoTIFFs.

    A real CRS and affine transform, so the coordinate path is exercised rather than
    stubbed: one degree per cell starting at 10°E, 50°N and running south.
    """
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    transform = from_origin(10.0, 50.0, 1.0, 1.0)
    paths = []
    for source in predictor_paths:
        grid = np.load(source)
        path = tmp_path / (Path(source).stem + ".tif")
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=grid.shape[0],
            width=grid.shape[1],
            count=1,
            dtype="float64",
            crs="EPSG:4326",
            transform=transform,
            nodata=np.nan,
        ) as handle:
            handle.write(grid, 1)
        paths.append(str(path))
    return paths


@pytest.fixture
def real_source(tmp_path: Path, predictor_paths: list[str], virtual_source) -> RealDataSource:
    """The same records as ``virtual_source``, arriving through the real-data path.

    Built by drawing from the virtual species and writing the result to CSV, so a test can
    assert that one methodology produces the same answer through either source.
    """
    context, _ = load_data_source(virtual_source, random_seed=7)
    path = tmp_path / "occurrences.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "col", "presence"])
        for (row, col), label in zip(context["coords"], context["labels"]):
            writer.writerow([int(row), int(col), int(label)])
    return RealDataSource(
        occurrences_path=str(path),
        predictor_paths=predictor_paths,
        occurrence_type="presence_absence",
    )


def methodology(
    algorithm: str = "logistic",
    split: str = "spatial_block",
    background: str | None = None,
) -> MethodologySpec:
    return MethodologySpec(
        cleaning=MethodSpec(method="drop_duplicate_cells"),
        accessible_area=MethodSpec(method="full_extent"),
        predictors=MethodSpec(method="standardize"),
        background=MethodSpec(method=background) if background else None,
        split=MethodSpec(method=split, params={"n_folds": 4, "block_size": 12}),
        algorithm=MethodSpec(method=algorithm),
    )


@pytest.fixture
def plan(virtual_source: VirtualDataSource) -> CandidatePlan:
    return CandidatePlan(
        candidate_id="c1",
        data_source=virtual_source,
        methodology=methodology(),
        random_seed=7,
    )


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "store")
