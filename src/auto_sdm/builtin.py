"""The built-in Python backends.

Enough of the search space to run end to end: two algorithms, two cross-validation
designs, one background strategy. Deliberately small — the point of Phase 1 is that the
machinery works, and the space grows by registering backends rather than by editing the
executor.

Each backend is deterministic given ``random_seed``; the run record claims reproducibility
on their behalf.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .backends import BackendRegistry, Runtime
from .models import Assumption, MethodSpec
from .profile import StandardProfiler

Outputs = tuple[dict[str, Any], list[Assumption]]


class _Backend:
    """Shared plumbing. Subclasses declare their stage and methods and implement run."""

    name: str = ""
    stage: str = ""
    methods: tuple[str, ...] = ()
    runtime: Runtime = "python"

    def run(self, spec: MethodSpec, inputs: dict[str, Any], random_seed: int) -> Outputs:
        raise NotImplementedError


# --------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------


class DropDuplicateCells(_Backend):
    """One record per raster cell. Duplicates are sampling effort, not signal."""

    name = "python-clean-duplicates"
    stage = "cleaning"
    methods = ("drop_duplicate_cells",)

    def run(self, spec: MethodSpec, inputs: dict[str, Any], random_seed: int) -> Outputs:
        coords = inputs["coords"]
        labels = inputs["labels"]
        keys = coords[:, 0].astype(np.int64) * (coords[:, 1].max() + 1) + coords[:, 1]
        # Keyed on cell *and* label, so a presence and an absence in one cell both survive
        # to be caught by whatever the methodology says to do about contradictions.
        _, keep = np.unique(keys * 2 + labels, return_index=True)
        keep = np.sort(keep)
        return (
            {
                "coords": coords[keep],
                "labels": labels[keep],
                "metrics": {
                    "n_records": float(keep.size),
                    "n_dropped": float(coords.shape[0] - keep.size),
                },
            },
            [],
        )


# --------------------------------------------------------------------------------------
# Accessible area
# --------------------------------------------------------------------------------------


class FullExtent(_Backend):
    """Treat the whole predictor extent as accessible.

    Almost never the ecologically right answer — but it is an explicit, named choice a
    candidate can be held to, rather than an unstated default.
    """

    name = "python-area-full-extent"
    stage = "accessible_area"
    methods = ("full_extent",)

    def run(self, spec: MethodSpec, inputs: dict[str, Any], random_seed: int) -> Outputs:
        grids = inputs["grids"]
        mask = np.ones(grids.shape[1:], dtype=bool)
        return {"accessible_mask": mask}, []


# --------------------------------------------------------------------------------------
# Predictors
# --------------------------------------------------------------------------------------


class StandardisePredictors(_Backend):
    """Extract predictor values at each record and z-score them.

    Standardisation uses whole-landscape statistics rather than per-fold statistics: the
    folds have not been drawn yet, and landscape statistics are not a function of the
    response, so this does not leak.
    """

    name = "python-predictors-standardise"
    stage = "predictors"
    methods = ("standardize", "standardise")

    def run(self, spec: MethodSpec, inputs: dict[str, Any], random_seed: int) -> Outputs:
        grids = inputs["grids"]
        flat = grids.reshape(grids.shape[0], -1)
        mean = np.nanmean(flat, axis=1)
        std = np.nanstd(flat, axis=1)
        std[std == 0] = 1.0

        coords = inputs["coords"]
        labels = inputs["labels"]
        raw = grids[:, coords[:, 0], coords[:, 1]].T
        design = (raw - mean) / std

        assumptions: list[Assumption] = []
        finite = np.isfinite(design).all(axis=1)
        if not finite.all():
            assumptions.append(
                Assumption(
                    code="dropped_records_with_missing_predictors",
                    message=f"{int((~finite).sum())} records fell on cells with no predictor value",
                    field="predictors",
                )
            )
        return (
            {
                "X": design[finite],
                "coords": coords[finite],
                "labels": labels[finite],
                "predictor_mean": mean,
                "predictor_std": std,
                "metrics": {"n_predictors": float(design.shape[1])},
            },
            assumptions,
        )


# --------------------------------------------------------------------------------------
# Background
# --------------------------------------------------------------------------------------


class RandomBackground(_Backend):
    """Uniform pseudo-absences across the accessible area.

    The naive baseline. It is registered precisely so that target-group background has
    something to beat on the calibration benchmark.
    """

    name = "python-background-random"
    stage = "background"
    methods = ("random",)

    def run(self, spec: MethodSpec, inputs: dict[str, Any], random_seed: int) -> Outputs:
        grids = inputs["grids"]
        mask = inputs.get("accessible_mask", np.ones(grids.shape[1:], dtype=bool))
        requested = int(spec.params.get("n_background", 5000))

        rows, cols = np.nonzero(mask)
        rng = np.random.default_rng(random_seed)

        assumptions: list[Assumption] = []
        count = requested
        if count > rows.size:
            assumptions.append(
                Assumption(
                    code="background_capped_by_extent",
                    message=f"requested {requested} background points but only {rows.size} cells "
                    "are accessible",
                    field="background.n_background",
                )
            )
            count = rows.size

        picked = rng.choice(rows.size, size=count, replace=False)
        background = np.stack([rows[picked], cols[picked]], axis=1)

        mean = inputs["predictor_mean"]
        std = inputs["predictor_std"]
        values = (grids[:, background[:, 0], background[:, 1]].T - mean) / std
        finite = np.isfinite(values).all(axis=1)

        return (
            {
                "X": np.vstack([inputs["X"], values[finite]]),
                "coords": np.vstack([inputs["coords"], background[finite]]),
                "labels": np.concatenate(
                    [inputs["labels"], np.zeros(int(finite.sum()), dtype=int)]
                ),
                "metrics": {"n_background": float(finite.sum())},
            },
            assumptions,
        )


# --------------------------------------------------------------------------------------
# Cross-validation designs
# --------------------------------------------------------------------------------------


class RandomKFold(_Backend):
    """Random folds.

    Registered as a *comparator*, not a recommendation: spatial autocorrelation inflates
    random-CV scores unevenly across candidates, so having it in the space is what makes
    that inflation measurable on the benchmark instead of merely asserted.
    """

    name = "python-split-random-kfold"
    stage = "split"
    methods = ("random_kfold",)

    def run(self, spec: MethodSpec, inputs: dict[str, Any], random_seed: int) -> Outputs:
        n_folds = int(spec.params.get("n_folds", 5))
        labels = inputs["labels"]
        rng = np.random.default_rng(random_seed)
        folds = rng.permutation(labels.size) % n_folds
        return {"folds": folds, "metrics": {"n_folds": float(n_folds)}}, []


class SpatialBlockSplit(_Backend):
    """Checkerboard spatial blocks assigned to folds.

    Blocks are assigned by their own index rather than at random per record, so a fold
    holds out contiguous geography — which is the entire point of blocking.
    """

    name = "python-split-spatial-block"
    stage = "split"
    methods = ("spatial_block",)

    def run(self, spec: MethodSpec, inputs: dict[str, Any], random_seed: int) -> Outputs:
        n_folds = int(spec.params.get("n_folds", 5))
        block_size = int(spec.params.get("block_size", 10))
        coords = inputs["coords"]

        block_rows = coords[:, 0] // block_size
        block_cols = coords[:, 1] // block_size
        block_ids = block_rows * (block_cols.max() + 1) + block_cols
        unique_blocks = np.unique(block_ids)

        rng = np.random.default_rng(random_seed)
        assignment = dict(zip(unique_blocks, rng.permutation(unique_blocks.size) % n_folds))
        folds = np.array([assignment[block] for block in block_ids])

        assumptions: list[Assumption] = []
        if unique_blocks.size < n_folds:
            assumptions.append(
                Assumption(
                    code="fewer_blocks_than_folds",
                    message=f"{unique_blocks.size} spatial blocks at block_size={block_size} "
                    f"cannot fill {n_folds} folds",
                    field="split.n_folds",
                )
            )
        return (
            {
                "folds": folds,
                "metrics": {
                    "n_folds": float(n_folds),
                    "n_blocks": float(unique_blocks.size),
                },
            },
            assumptions,
        )


# --------------------------------------------------------------------------------------
# Algorithms
# --------------------------------------------------------------------------------------


class _CrossValidatedAlgorithm(_Backend):
    """Fit once per fold and report per-fold AUC.

    ``auc_sd`` is reported alongside ``auc_mean`` because fold stability is a selection
    criterion in its own right — a candidate that wins on the mean while swinging across
    blocks is worse than its mean implies.
    """

    stage = "algorithm"

    def _fit(self, X: np.ndarray, y: np.ndarray, spec: MethodSpec, random_seed: int) -> Any:
        raise NotImplementedError

    def run(self, spec: MethodSpec, inputs: dict[str, Any], random_seed: int) -> Outputs:
        X = inputs["X"]
        y = inputs["labels"]
        folds = inputs["folds"]

        scores = []
        skipped = 0
        predictions = np.full(y.size, np.nan)
        for fold in np.unique(folds):
            test = folds == fold
            train = ~test
            if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
                # A fold with one class present cannot produce an AUC. Recording it as an
                # assumption keeps a candidate scored on two usable folds from looking
                # identical to one scored on five.
                skipped += 1
                continue
            model = self._fit(X[train], y[train], spec, random_seed)
            fold_predictions = model.predict_proba(X[test])[:, 1]
            predictions[test] = fold_predictions
            scores.append(float(roc_auc_score(y[test], fold_predictions)))

        assumptions: list[Assumption] = []
        if skipped:
            assumptions.append(
                Assumption(
                    code="folds_skipped_single_class",
                    message=f"{skipped} of {np.unique(folds).size} folds held out only one class",
                    field="split",
                )
            )
        if not scores:
            raise ValueError("no fold contained both classes; the candidate cannot be scored")

        final = self._fit(X, y, spec, random_seed)
        return (
            {
                "model": final,
                "cv_predictions": predictions,
                "metrics": {
                    "auc_mean": float(np.mean(scores)),
                    "auc_sd": float(np.std(scores)),
                    "n_scored_folds": float(len(scores)),
                },
            },
            assumptions,
        )


class LogisticAlgorithm(_CrossValidatedAlgorithm):
    name = "python-logistic"
    methods = ("logistic", "glm")

    def _fit(self, X: np.ndarray, y: np.ndarray, spec: MethodSpec, random_seed: int) -> Any:
        model = LogisticRegression(
            C=float(spec.params.get("C", 1.0)),
            max_iter=int(spec.params.get("max_iter", 1000)),
        )
        model.fit(X, y)
        return model


class GradientBoostingAlgorithm(_CrossValidatedAlgorithm):
    name = "python-gbm"
    methods = ("gbm", "brt")

    def _fit(self, X: np.ndarray, y: np.ndarray, spec: MethodSpec, random_seed: int) -> Any:
        model = GradientBoostingClassifier(
            n_estimators=int(spec.params.get("n_estimators", 100)),
            learning_rate=float(spec.params.get("learning_rate", 0.1)),
            max_depth=int(spec.params.get("max_depth", 3)),
            random_state=random_seed,
        )
        model.fit(X, y)
        return model


BUILTIN_BACKENDS: tuple[Any, ...] = (
    StandardProfiler(),
    DropDuplicateCells(),
    FullExtent(),
    StandardisePredictors(),
    RandomBackground(),
    RandomKFold(),
    SpatialBlockSplit(),
    LogisticAlgorithm(),
    GradientBoostingAlgorithm(),
)


def default_registry() -> BackendRegistry:
    """The executable search space as shipped."""
    registry = BackendRegistry()
    for backend in BUILTIN_BACKENDS:
        registry.register(backend)
    return registry
