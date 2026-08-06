"""Reading occurrence tables.

Real occurrence data arrives as a CSV with whatever column names its source used, in
whatever CRS its source used, with rows that are unparseable or fall outside the study
extent. Every one of those is a place where records can vanish.

Nothing here drops a record silently. Unparseable coordinates, unparseable dates, and
records outside the extent each produce a typed ``Assumption`` carrying the count — a
user with a CRS mismatch must not see a clean run over the 3% of their data that happened
to land on the grid.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np

from .models import Assumption, UnsupportedStep
from .rasters import PredictorGrids, world_to_cell

try:  # pragma: no cover - exercised by whichever environment runs the tests
    from pyproj import Transformer

    HAVE_PYPROJ = True
except ImportError:  # pragma: no cover
    HAVE_PYPROJ = False

# Column aliases, lowercased. Darwin Core first, since GBIF exports are the common case.
LONGITUDE_ALIASES = ("decimallongitude", "longitude", "lon", "long", "x")
LATITUDE_ALIASES = ("decimallatitude", "latitude", "lat", "y")
ROW_ALIASES = ("row",)
COL_ALIASES = ("col", "column")
PRESENCE_ALIASES = ("presence", "occurrencestatus", "occurrence")
DATE_ALIASES = ("eventdate", "date", "observed_on", "year")


class UnreadableOccurrences(Exception):
    """An occurrence table that cannot be read. Carries the typed gap for the backlog."""

    def __init__(self, gap: UnsupportedStep) -> None:
        super().__init__(gap.message)
        self.gap = gap


@dataclass(frozen=True)
class OccurrenceTable:
    coords: np.ndarray
    labels: np.ndarray
    dates: np.ndarray | None
    assumptions: list[Assumption] = field(default_factory=list)


def _gap(code: str, message: str, requested: str | None = None) -> UnreadableOccurrences:
    return UnreadableOccurrences(
        UnsupportedStep(
            code=code, message=message, feature="occurrence_format", requested_value=requested
        )
    )


def _find(header: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in header:
            return header[alias]
    return None


def _parse_presence(raw: str) -> int:
    text = raw.strip().lower()
    if text in ("1", "true", "yes", "present", "presence"):
        return 1
    if text in ("0", "false", "no", "absent", "absence"):
        return 0
    raise ValueError(f"unrecognised presence value {raw!r}")


def _parse_date(raw: str) -> date:
    text = raw.strip()
    if not text:
        raise ValueError("empty date")
    if len(text) == 4 and text.isdigit():  # a bare year, common in older records
        return date(int(text), 1, 1)
    # GBIF eventDate is often a range or carries a time component; the leading date is
    # what a temporal profile needs, and truncating it is recorded by the caller.
    return datetime.fromisoformat(text.split("/")[0].replace("Z", "+00:00")).date()


def read_occurrences(
    occurrences_path: str,
    grids: PredictorGrids,
    occurrence_crs: str,
    date_field: str | None = None,
) -> OccurrenceTable:
    """Read a CSV into cell indices, labels, and dates against a predictor stack."""
    path = Path(occurrences_path)
    if not path.exists():
        raise _gap("missing_occurrence_file", f"no occurrence file at {occurrences_path}")

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        raise _gap("empty_occurrence_file", f"{path.name} contains no records")

    header = {name.lower(): name for name in fieldnames}
    lon_key = _find(header, LONGITUDE_ALIASES)
    lat_key = _find(header, LATITUDE_ALIASES)
    row_key = _find(header, ROW_ALIASES)
    col_key = _find(header, COL_ALIASES)
    presence_key = _find(header, PRESENCE_ALIASES)
    chosen_date_field = date_field or _find(header, DATE_ALIASES)

    using_world_coordinates = lon_key is not None and lat_key is not None
    if not using_world_coordinates and (row_key is None or col_key is None):
        raise _gap(
            "unrecognised_occurrence_columns",
            "occurrence files need longitude/latitude columns (any of "
            f"{LONGITUDE_ALIASES}/{LATITUDE_ALIASES}) or explicit row/col indices; found "
            f"{sorted(header)}",
            ",".join(sorted(header)),
        )
    if using_world_coordinates and not grids.is_georeferenced:
        raise _gap(
            "ungeoreferenced_predictors",
            "occurrences carry world coordinates but the predictors have no affine "
            "transform; supply GeoTIFF predictors or pre-indexed row/col occurrences",
        )

    assumptions: list[Assumption] = []
    xs: list[float] = []
    ys: list[float] = []
    labels: list[int] = []
    dates: list[date | None] = []
    unparseable_coords = 0
    unparseable_labels = 0
    unparseable_dates = 0

    for record in rows:
        try:
            if using_world_coordinates:
                assert lon_key is not None and lat_key is not None
                xs.append(float(record[lon_key]))
                ys.append(float(record[lat_key]))
            else:
                assert row_key is not None and col_key is not None
                ys.append(float(record[row_key]))
                xs.append(float(record[col_key]))
        except (TypeError, ValueError):
            unparseable_coords += 1
            continue

        if presence_key is None:
            labels.append(1)
        else:
            try:
                labels.append(_parse_presence(record[presence_key] or ""))
            except ValueError:
                labels.append(1)
                unparseable_labels += 1

        if chosen_date_field is None:
            dates.append(None)
        else:
            try:
                dates.append(_parse_date(record.get(chosen_date_field) or ""))
            except (TypeError, ValueError):
                dates.append(None)
                unparseable_dates += 1

    if unparseable_coords:
        assumptions.append(
            Assumption(
                code="dropped_unparseable_coordinates",
                message=f"{unparseable_coords} of {len(rows)} records had unreadable coordinates",
                field="occurrences",
            )
        )
    if unparseable_labels:
        assumptions.append(
            Assumption(
                code="unreadable_presence_values_treated_as_presence",
                message=f"{unparseable_labels} records had an unrecognised presence value",
                field="occurrences.presence",
            )
        )
    if unparseable_dates:
        assumptions.append(
            Assumption(
                code="dropped_unparseable_dates",
                message=f"{unparseable_dates} records had an unreadable date; their temporal "
                "information is unavailable",
                field="occurrences.date",
            )
        )
    if not xs:
        raise _gap("no_usable_occurrences", f"no record in {path.name} had usable coordinates")

    x_array = np.asarray(xs, dtype=float)
    y_array = np.asarray(ys, dtype=float)

    if using_world_coordinates:
        x_array, y_array, reprojection = _reproject(x_array, y_array, occurrence_crs, grids.crs)
        assumptions.extend(reprojection)
        assert grids.transform is not None
        row_indices, col_indices = world_to_cell(grids.transform, x_array, y_array)
    else:
        row_indices = y_array.astype(int)
        col_indices = x_array.astype(int)

    coords = np.stack([row_indices, col_indices], axis=1)
    height, width = grids.shape
    inside = (
        (coords[:, 0] >= 0) & (coords[:, 0] < height) & (coords[:, 1] >= 0) & (coords[:, 1] < width)
    )
    if not inside.all():
        outside = int((~inside).sum())
        assumptions.append(
            Assumption(
                code="dropped_records_outside_extent",
                message=f"{outside} of {inside.size} records fell outside the predictor extent; "
                "a large share usually means the occurrence CRS is wrong",
                field="occurrences",
            )
        )

    date_array: np.ndarray | None = None
    if chosen_date_field is not None:
        date_array = np.array(
            [np.datetime64(value) if value else np.datetime64("NaT") for value in dates],
            dtype="datetime64[D]",
        )[inside]

    return OccurrenceTable(
        coords=coords[inside],
        labels=np.asarray(labels, dtype=int)[inside],
        dates=date_array,
        assumptions=assumptions,
    )


def _reproject(
    xs: np.ndarray, ys: np.ndarray, source_crs: str, target_crs: str | None
) -> tuple[np.ndarray, np.ndarray, list[Assumption]]:
    """Move occurrence coordinates into the predictor CRS.

    A mismatch that goes unnoticed is the single most effective way to silently lose most
    of a dataset, so an unknown target CRS is recorded rather than assumed compatible.
    """
    if target_crs is None or _same_crs(source_crs, target_crs):
        return xs, ys, []
    if not HAVE_PYPROJ:
        raise _gap(
            "missing_reprojection_runtime",
            f"occurrences are in {source_crs} but predictors are in {target_crs}; "
            "reprojection needs pyproj (install auto-sdm[geo])",
            source_crs,
        )
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    # tolist() keeps pyproj on its sequence path; a one-element array takes the scalar
    # path instead and warns.
    moved_x, moved_y = transformer.transform(xs.tolist(), ys.tolist())
    return (
        np.asarray(moved_x, dtype=float),
        np.asarray(moved_y, dtype=float),
        [
            Assumption(
                code="reprojected_occurrences",
                message=f"occurrence coordinates reprojected from {source_crs} to {target_crs}",
                field="occurrences.crs",
            )
        ],
    )


def _same_crs(first: str, second: str) -> bool:
    if first.strip().upper() == second.strip().upper():
        return True
    if not HAVE_PYPROJ:
        return False
    from pyproj import CRS

    try:
        return bool(CRS.from_user_input(first) == CRS.from_user_input(second))
    except Exception:  # pragma: no cover - malformed CRS strings are reported downstream
        return False
