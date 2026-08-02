"""Materialising a data source into the context the stages operate on.

The two source kinds converge on exactly one shape — gridded predictors, occurrence
coordinates, labels — so that from the first stage onward nothing downstream can tell
whether it is looking at a user's dataset or a simulated species. That is what lets the
selector be calibrated on the real execution path rather than a parallel one.

``VirtualDataSource.true_coefficients`` is used *here* to generate the species and then
discarded. It is never placed in the returned context, because everything downstream
reads that context.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .models import DataSource, RealDataSource, UnsupportedStep, VirtualDataSource

# Predictors are read as ``.npy`` grids for now. GeoTIFF is a real capability the system
# does not yet have, and it is reported as such rather than approximated.
SUPPORTED_PREDICTOR_SUFFIXES = (".npy",)


class UnsupportedDataSource(Exception):
    """A data source the loader cannot read. Carries the typed gap for the backlog."""

    def __init__(self, gap: UnsupportedStep) -> None:
        super().__init__(gap.message)
        self.gap = gap


def _load_grids(predictor_paths: list[str]) -> tuple[np.ndarray, list[str]]:
    grids = []
    names = []
    for raw in predictor_paths:
        path = Path(raw)
        if path.suffix.lower() not in SUPPORTED_PREDICTOR_SUFFIXES:
            raise UnsupportedDataSource(
                UnsupportedStep(
                    code="unsupported_predictor_format",
                    message=(
                        f"no reader for predictor format {path.suffix!r}; "
                        f"supported: {', '.join(SUPPORTED_PREDICTOR_SUFFIXES)}"
                    ),
                    feature="predictor_format",
                    requested_value=path.suffix.lower(),
                )
            )
        grid = np.load(path)
        if grid.ndim != 2:
            raise UnsupportedDataSource(
                UnsupportedStep(
                    code="unsupported_predictor_shape",
                    message=f"predictor {path.name} is {grid.ndim}-dimensional; expected a 2-D grid",
                    feature="predictor_format",
                    requested_value=str(grid.ndim),
                )
            )
        grids.append(grid.astype(float))
        names.append(path.stem)

    shapes = {grid.shape for grid in grids}
    if len(shapes) > 1:
        raise UnsupportedDataSource(
            UnsupportedStep(
                code="misaligned_predictors",
                message=f"predictor grids differ in shape: {sorted(shapes)}",
                feature="predictor_format",
            )
        )
    return np.stack(grids), names


def _standardise(grids: np.ndarray) -> np.ndarray:
    flat = grids.reshape(grids.shape[0], -1)
    mean = np.nanmean(flat, axis=1, keepdims=True)
    std = np.nanstd(flat, axis=1, keepdims=True)
    std[std == 0] = 1.0
    return ((flat - mean) / std).reshape(grids.shape)


def _load_real(source: RealDataSource) -> dict[str, Any]:
    grids, names = _load_grids(source.predictor_paths)
    _, height, width = grids.shape

    rows: list[tuple[int, int]] = []
    labels: list[int] = []
    with open(source.occurrences_path, newline="") as handle:
        for record in csv.DictReader(handle):
            keys = {key.lower(): key for key in record}
            if "row" not in keys or "col" not in keys:
                raise UnsupportedDataSource(
                    UnsupportedStep(
                        code="unsupported_occurrence_columns",
                        message="occurrence files need 'row' and 'col' columns; "
                        "projected coordinate handling is not built yet",
                        feature="occurrence_format",
                        requested_value=",".join(sorted(record)),
                    )
                )
            rows.append((int(record[keys["row"]]), int(record[keys["col"]])))
            labels.append(int(record[keys["presence"]]) if "presence" in keys else 1)

    coords = np.array(rows, dtype=int).reshape(-1, 2)
    inside = (
        (coords[:, 0] >= 0) & (coords[:, 0] < height) & (coords[:, 1] >= 0) & (coords[:, 1] < width)
    )
    return {
        "grids": grids,
        "predictor_names": names,
        "coords": coords[inside],
        "labels": np.array(labels, dtype=int)[inside],
        "occurrence_type": source.occurrence_type,
    }


def _load_virtual(source: VirtualDataSource, random_seed: int) -> dict[str, Any]:
    if source.sampling_bias is not None:
        raise UnsupportedDataSource(
            UnsupportedStep(
                code="unsupported_sampling_bias",
                message=f"no generator for sampling bias {source.sampling_bias!r}",
                feature="sampling_bias",
                requested_value=source.sampling_bias,
            )
        )

    grids, names = _load_grids(source.predictor_paths)
    standardised = _standardise(grids)
    coefficients = np.array(source.true_coefficients, dtype=float).reshape(-1, 1)
    logits = source.true_intercept + (standardised.reshape(len(names), -1) * coefficients).sum(0)
    suitability = 1.0 / (1.0 + np.exp(-logits))

    rng = np.random.default_rng(random_seed)
    _, height, width = grids.shape
    cells = np.arange(suitability.size)

    def draw(weights: np.ndarray, count: int) -> np.ndarray:
        total = weights.sum()
        if total <= 0:
            raise UnsupportedDataSource(
                UnsupportedStep(
                    code="degenerate_virtual_species",
                    message="the specified niche gives every cell zero weight",
                    feature="virtual_species",
                )
            )
        replace = count > cells.size
        return rng.choice(cells, size=count, replace=replace, p=weights / total)

    presence_cells = draw(suitability, source.n_presences)
    if source.occurrence_type == "presence_absence":
        absence_cells = draw(1.0 - suitability, source.n_absences)
        chosen = np.concatenate([presence_cells, absence_cells])
        labels = np.concatenate(
            [np.ones(presence_cells.size, dtype=int), np.zeros(absence_cells.size, dtype=int)]
        )
    else:
        chosen = presence_cells
        labels = np.ones(presence_cells.size, dtype=int)

    coords = np.stack([chosen // width, chosen % width], axis=1)
    assert coords[:, 0].max() < height
    return {
        "grids": grids,
        "predictor_names": names,
        "coords": coords,
        "labels": labels,
        "occurrence_type": source.occurrence_type,
    }


def load_data_source(source: DataSource, random_seed: int) -> dict[str, Any]:
    """Build the starting context. Raises ``UnsupportedDataSource`` rather than guessing."""
    if isinstance(source, VirtualDataSource):
        return _load_virtual(source, random_seed)
    return _load_real(source)
