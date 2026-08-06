"""Reading predictor grids.

Two formats, deliberately unequal. GeoTIFF is what users have and carries the affine
transform and CRS that make occurrence coordinates meaningful; ``.npy`` is the bare
internal representation with no georeferencing, kept because it makes the test suite fast
and dependency-free.

``rasterio`` is an optional dependency. Its absence is reported as a typed capability gap
through the same path an unimplemented method takes, rather than as an import error at
module load — a missing runtime is a capability the system lacks, not a crash.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import UnsupportedStep

try:  # pragma: no cover - exercised by whichever environment runs the tests
    import rasterio

    HAVE_RASTERIO = True
except ImportError:  # pragma: no cover
    HAVE_RASTERIO = False

NPY_SUFFIXES = (".npy",)
GEOTIFF_SUFFIXES = (".tif", ".tiff")


class UnreadableRasters(Exception):
    """Predictors that cannot be read. Carries the typed gap for the backlog."""

    def __init__(self, gap: UnsupportedStep) -> None:
        super().__init__(gap.message)
        self.gap = gap


@dataclass(frozen=True)
class PredictorGrids:
    """A stack of aligned predictor layers.

    ``transform`` is the affine mapping cell indices to world coordinates, as the
    six-tuple ``(a, b, c, d, e, f)``. It is ``None`` for ``.npy`` input, and code that
    needs to place real-world coordinates must treat that as a gap rather than assuming
    an identity transform — an identity transform silently means "cell indices are
    degrees", which is wrong everywhere on Earth.
    """

    values: np.ndarray
    names: list[str]
    transform: tuple[float, float, float, float, float, float] | None
    crs: str | None

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.values.shape[1]), int(self.values.shape[2])

    @property
    def is_georeferenced(self) -> bool:
        return self.transform is not None


def _gap(code: str, message: str, requested: str | None = None) -> UnreadableRasters:
    return UnreadableRasters(
        UnsupportedStep(
            code=code, message=message, feature="predictor_format", requested_value=requested
        )
    )


def _read_npy(path: Path) -> np.ndarray:
    grid = np.load(path)
    if grid.ndim != 2:
        raise _gap(
            "unsupported_predictor_shape",
            f"predictor {path.name} is {grid.ndim}-dimensional; expected a 2-D grid",
            str(grid.ndim),
        )
    return grid.astype(float)


def _read_geotiff(path: Path) -> tuple[np.ndarray, tuple[float, ...], str | None]:
    if not HAVE_RASTERIO:
        raise _gap(
            "missing_geospatial_runtime",
            f"reading {path.suffix} needs rasterio; install auto-sdm[geo]",
            path.suffix.lower(),
        )
    try:
        handle = rasterio.open(path)
    except Exception as exc:
        # A corrupt or truncated raster is a file the system cannot read, which is the
        # same kind of event as a format it has no reader for — not a crash.
        raise _gap(
            "unreadable_raster",
            f"{path.name} could not be opened as a raster: {exc}",
            path.suffix.lower(),
        ) from exc
    with handle:
        if handle.count != 1:
            raise _gap(
                "unsupported_multiband_raster",
                f"{path.name} has {handle.count} bands; one predictor per file is expected",
                str(handle.count),
            )
        # masked=True honours the nodata value; filling with NaN lets the rest of the
        # system treat "no data here" uniformly, and predictors drops those records with
        # a recorded assumption rather than imputing them.
        band = handle.read(1, masked=True).filled(np.nan).astype(float)
        crs = str(handle.crs) if handle.crs else None
        return band, tuple(handle.transform)[:6], crs


def read_predictors(predictor_paths: list[str]) -> PredictorGrids:
    """Read and align a predictor stack. Raises ``UnreadableRasters`` rather than guessing."""
    layers: list[np.ndarray] = []
    names: list[str] = []
    transforms: set[tuple[float, ...]] = set()
    crs_values: set[str | None] = set()
    georeferenced: list[bool] = []

    for raw in predictor_paths:
        path = Path(raw)
        suffix = path.suffix.lower()
        georeferenced.append(suffix in GEOTIFF_SUFFIXES)
        if suffix in NPY_SUFFIXES:
            layers.append(_read_npy(path))
        elif suffix in GEOTIFF_SUFFIXES:
            band, transform, crs = _read_geotiff(path)
            layers.append(band)
            transforms.add(transform)
            crs_values.add(crs)
        else:
            raise _gap(
                "unsupported_predictor_format",
                f"no reader for predictor format {suffix!r}; supported: "
                f"{', '.join(NPY_SUFFIXES + GEOTIFF_SUFFIXES)}",
                suffix,
            )
        names.append(path.stem)

    if any(georeferenced) and not all(georeferenced):
        raise _gap(
            "mixed_predictor_georeferencing",
            "cannot mix georeferenced and ungeoreferenced predictors in one stack",
        )

    shapes = {layer.shape for layer in layers}
    if len(shapes) > 1:
        raise _gap("misaligned_predictors", f"predictor grids differ in shape: {sorted(shapes)}")
    if len(transforms) > 1:
        raise _gap(
            "misaligned_predictors",
            "predictor grids have different affine transforms; resampling to a common grid "
            "is not built yet",
        )
    if len(crs_values) > 1:
        raise _gap(
            "misaligned_predictors",
            f"predictor grids are in different CRSs: {sorted(str(c) for c in crs_values)}; "
            "reprojecting predictors is not built yet",
        )
    common = next(iter(transforms)) if transforms else None
    return PredictorGrids(
        values=np.stack(layers),
        names=names,
        transform=(
            (common[0], common[1], common[2], common[3], common[4], common[5])
            if common is not None
            else None
        ),
        crs=next(iter(crs_values)) if crs_values else None,
    )


def world_to_cell(
    transform: tuple[float, float, float, float, float, float],
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the affine transform: world coordinates to (row, col) indices.

    Implemented directly rather than via rasterio so that the coordinate path stays
    usable wherever a transform came from.
    """
    a, b, c, d, e, f = transform
    determinant = a * e - b * d
    if determinant == 0:
        raise _gap("degenerate_transform", "the raster's affine transform is not invertible")
    dx = xs - c
    dy = ys - f
    cols = (e * dx - b * dy) / determinant
    rows = (-d * dx + a * dy) / determinant
    return np.floor(rows).astype(int), np.floor(cols).astype(int)
