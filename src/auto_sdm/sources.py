"""Materialising a data source into the context the stages operate on.

The two source kinds converge on exactly one shape, so that from the first stage onward
nothing downstream can tell whether it is looking at a user's dataset or a simulated
species. That is what lets the selector be calibrated on the real execution path rather
than a parallel one.

``VirtualDataSource.true_coefficients`` is used *here* to generate the species and then
discarded. It is never placed in the returned context, because everything downstream
reads that context.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import Assumption, DataSource, RealDataSource, UnsupportedStep, VirtualDataSource
from .occurrences import UnreadableOccurrences, read_occurrences
from .rasters import PredictorGrids, UnreadableRasters, read_predictors

# The keys every stage may rely on. Kept explicit because "real and virtual are
# indistinguishable downstream" is only true if both produce exactly this set.
CONTEXT_KEYS = (
    "grids",
    "predictor_names",
    "coords",
    "labels",
    "dates",
    "transform",
    "crs",
    "occurrence_type",
)


class UnsupportedDataSource(Exception):
    """A data source the loader cannot read. Carries the typed gap for the backlog."""

    def __init__(self, gap: UnsupportedStep) -> None:
        super().__init__(gap.message)
        self.gap = gap


def _context(
    grids: PredictorGrids,
    coords: np.ndarray,
    labels: np.ndarray,
    occurrence_type: str,
    dates: np.ndarray | None = None,
) -> dict[str, Any]:
    return {
        "grids": grids.values,
        "predictor_names": grids.names,
        "coords": coords,
        "labels": labels,
        "dates": dates,
        "transform": list(grids.transform) if grids.transform is not None else None,
        "crs": grids.crs,
        "occurrence_type": occurrence_type,
    }


def _read_grids(predictor_paths: list[str]) -> PredictorGrids:
    try:
        return read_predictors(predictor_paths)
    except UnreadableRasters as exc:
        raise UnsupportedDataSource(exc.gap) from exc


def _load_real(source: RealDataSource) -> tuple[dict[str, Any], list[Assumption]]:
    grids = _read_grids(source.predictor_paths)
    try:
        table = read_occurrences(
            source.occurrences_path,
            grids,
            occurrence_crs=source.crs,
            date_field=source.date_field,
        )
    except UnreadableOccurrences as exc:
        raise UnsupportedDataSource(exc.gap) from exc

    return (
        _context(grids, table.coords, table.labels, source.occurrence_type, table.dates),
        table.assumptions,
    )


def _standardise(grids: np.ndarray) -> np.ndarray:
    flat = grids.reshape(grids.shape[0], -1)
    mean = np.nanmean(flat, axis=1, keepdims=True)
    std = np.nanstd(flat, axis=1, keepdims=True)
    std[std == 0] = 1.0
    return ((flat - mean) / std).reshape(grids.shape)


def _load_virtual(
    source: VirtualDataSource, random_seed: int
) -> tuple[dict[str, Any], list[Assumption]]:
    if source.sampling_bias is not None:
        raise UnsupportedDataSource(
            UnsupportedStep(
                code="unsupported_sampling_bias",
                message=f"no generator for sampling bias {source.sampling_bias!r}",
                feature="sampling_bias",
                requested_value=source.sampling_bias,
            )
        )

    grids = _read_grids(source.predictor_paths)
    standardised = _standardise(grids.values)
    coefficients = np.array(source.true_coefficients, dtype=float).reshape(-1, 1)
    logits = source.true_intercept + (
        standardised.reshape(len(grids.names), -1) * coefficients
    ).sum(0)
    suitability = 1.0 / (1.0 + np.exp(-logits))

    assumptions: list[Assumption] = []
    usable = np.isfinite(suitability)
    if not usable.all():
        # Real rasters have nodata; the niche is undefined there rather than zero.
        suitability = np.where(usable, suitability, 0.0)
        assumptions.append(
            Assumption(
                code="virtual_species_excluded_nodata_cells",
                message=f"{int((~usable).sum())} cells have no predictor value and cannot hold "
                "simulated occurrences",
                field="predictors",
            )
        )

    rng = np.random.default_rng(random_seed)
    _, width = grids.shape
    cells = np.arange(suitability.size)

    def draw(weights: np.ndarray, count: int) -> np.ndarray:
        weights = np.where(usable, weights, 0.0)
        total = weights.sum()
        if total <= 0:
            raise UnsupportedDataSource(
                UnsupportedStep(
                    code="degenerate_virtual_species",
                    message="the specified niche gives every cell zero weight",
                    feature="virtual_species",
                )
            )
        return rng.choice(cells, size=count, replace=count > cells.size, p=weights / total)

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
    return _context(grids, coords, labels, source.occurrence_type), assumptions


def load_data_source(
    source: DataSource, random_seed: int
) -> tuple[dict[str, Any], list[Assumption]]:
    """Build the starting context and whatever was assumed to build it.

    Raises ``UnsupportedDataSource`` rather than guessing at anything it cannot read.
    """
    if isinstance(source, VirtualDataSource):
        return _load_virtual(source, random_seed)
    return _load_real(source)
