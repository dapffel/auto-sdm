# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Install

```bash
pip install -e ".[dev]"      # editable install with dev tools (pytest, black, isort, mypy)
```

## Development Commands

```bash
pytest                                  # run all tests
pytest tests/test_models.py::test_name  # run a single test
black src tests                         # format
isort src tests                         # sort imports
mypy src/                               # type check
```

Keep the `black`/`isort`/`mypy`/`pytest` baseline clean; every commit should pass all four.

## Architecture

auto-sdm searches over SDM *methodologies* — it does not run one pipeline. Candidates are
proposed, executed, and one is selected. See `README.md` for the full rationale.

The load-bearing problem is **selection without ground truth**: held-out AUC on presence
data largely measures agreement with the sampling process, so a search that optimizes it
returns the wrong model convincingly. Selection is therefore composite (spatial block CV,
CBI, importance-weighted CV, extrapolation penalty, fold stability) and weighted by the
user's stated `objective`.

### The contract (`models.py`)

`CandidatePlan` splits into **`data_source`** (where data comes from) and
**`methodology`** (what is done to it). This split is load-bearing: the same
`MethodologySpec` runs unchanged against a `RealDataSource` or a `VirtualDataSource`,
which is what lets selector calibration reuse the real path rather than duplicate it.

`VirtualDataSource.true_coefficients` is ground truth. **It must never reach the proposer,
executor, or selector** — only the calibration harness scoring their output.

`MethodSpec.method` is an open string, not a `Literal`. The search space must be able to
grow through synthesized backends, so an unknown method is a *capability gap*, not a
schema error.

### Nothing is silently substituted

Every fallback is a typed `Assumption`; every unimplementable method is a typed
`UnsupportedStep`. Both carry a stable `code` so they aggregate across runs into a backlog
ranked by how much real methodology each gap would unlock. This is the mechanism that
decides what the system builds next — treat it as load-bearing, not as logging.

### The registry (`backends.py`)

Backends are capability providers keyed on `(stage, method)`, not a language choice. A
Python, R, or synthesized backend is indistinguishable to the executor. `resolve()` returns
a `Resolution` that is *exactly one of* a backend or an `UnsupportedStep` — it never raises
for an unknown method and never substitutes a near-match.

`Backend.run` implementations must be deterministic given `random_seed`; the run record
claims reproducibility, and a backend that ignores the seed silently breaks that claim.

### Modules (`src/auto_sdm/`)

- **`models.py`** — the whole execution contract. Pydantic v2, frozen. `CandidatePlan`,
  `DataSource` (discriminated on `kind`), `MethodologySpec`/`MethodSpec`, the typed
  `Assumption`/`UnsupportedStep`, and the run records (`StepRecord`, `Artifact`,
  `SelectionScore`, `CandidateResult`, `RunManifest`, `RunResult`).
- **`backends.py`** — `Backend` protocol, `BackendRegistry`, `Resolution`.
- **`digest.py`** — content addressing. Canonicalises values (including arrays) to a stable
  hash. Undigestable types **raise**: a `repr` fallback would embed a memory address and
  make every rerun look different, which is worse than failing loudly.
- **`store.py`** — `ArtifactStore`. Content-addressed blobs plus per-run JSON records.
- **`sources.py`** — materialises a `DataSource` into the stage context. Both kinds converge
  on `CONTEXT_KEYS` exactly, so nothing downstream can tell them apart. `true_coefficients`
  is used here and discarded — it must never appear in the returned context. Returns
  `(context, assumptions)`.
- **`rasters.py`** — predictor stacks. GeoTIFF via `rasterio` (optional, `auto-sdm[geo]`) and
  bare `.npy`. `PredictorGrids.transform` is `None` for `.npy`: code needing world
  coordinates must treat that as a gap, because an identity transform silently means "cell
  indices are degrees".
- **`occurrences.py`** — occurrence CSVs. Darwin Core and common column aliases, CRS
  reprojection via `pyproj`, date parsing.
- **`profile.py`** — `DataProfile`: the deterministic characterisation the proposer's priors
  and run history both key on. Registry stage `"profile"`.
- **`priors.py`** — `Prior`/`Condition`: methodological priors as *data*, keyed on dotted
  paths into the profile and the request. `DEFAULT_PRIORS` is hand-authored in the structure
  `lit-review` will emit into.
- **`proposer.py`** — `propose()` enumerates the executable space and ranks it; `explain()`
  reports per-method weights and the priors behind them.
- **`executor.py`** — walks `MethodologySpec.steps()` in order, resolving each stage.
- **`builtin.py`** — the shipped Python backends and `default_registry()`.

### The stage contract (`executor.py`)

Stages communicate through one `dict[str, Any]` context: a backend's outputs are merged into
it and become the next stage's inputs. Two conventions:

- The reserved output key **`metrics`** is lifted into the `StepRecord` rather than threaded
  onward, so metrics never masquerade as data.
- Every array a stage emits is persisted content-addressed. A stage that leaves the data
  alone re-emits identical bytes, which makes "this stage changed nothing" a checkable fact
  about the run record rather than a claim in a step card.

A candidate can fail three ways, and they are deliberately distinct: an **unresolved method**
(typed `UnsupportedStep`, attached to the returned plan so it reaches the backlog), an
**unreadable data source** (same, via `UnsupportedDataSource`), or a **backend raising**
(a crash, *not* a capability gap — it must not pollute the backlog). None of them fail the run.

### Ingest never loses records quietly

Every place a record can vanish emits an `Assumption` carrying the count: unparseable
coordinates, unparseable dates, unrecognised presence values, reprojection, and records
outside the predictor extent. This is not politeness — a user whose occurrence CRS is wrong
would otherwise get a clean run with a plausible AUC over whatever fraction of their data
happened to land on the grid. Loading gets its own `data_source` step record so those
assumptions are answerable from the run record.

Optional geospatial dependencies are guarded at the import site: without `rasterio`/`pyproj`,
GeoTIFF reading and reprojection resolve to typed gaps (`missing_geospatial_runtime`,
`missing_reprojection_runtime`) rather than raising `ImportError`. Apply the same pattern to
`xferweight` and `sdm_pipepy` when they are wired in.

### The profile is measurement, not judgment

`build_profile` is deterministic given its seed. This is load-bearing: if a profile drifts
between runs, calibration cannot attribute regret to the selector rather than to the
profiler, and cross-run learning becomes learning from noise. Interpreting a profile into
scenario labels is a separate, non-deterministic layer that must be recorded as such.

The profile should **condition** the proposal distribution, never filter it. Narrowing to
"the three methods for this scenario" replaces the search with a heuristic and destroys the
ability to measure whether the heuristic was right — which is the premise of the project. An
always-nonzero tail is what lets calibration find the regions of data space where a prior is
wrong.

Fields state what they cannot know rather than guessing: `prevalence` is `None` for
presence-only data (not `1.0`), and `autocorrelation_range_bounded` is `False` when the
landscape carries a trend across the whole extent, because the semivariogram never reaches a
sill inside the study area and the number is a lower bound. `suggested_block_size()` clamps
to the extent — an unclamped range on a trended landscape exceeds the grid, giving one block
and no folds at all.

### Priors rank; they never filter

The registry decides what is **possible**; priors only decide what is **promising**. Weights
are additive in log space and every candidate keeps a nonzero probability, so a budgeted
search *samples* rather than truncates. Truncation would convert the search into a
heuristic and remove the system's ability to discover that a prior was wrong — which is the
one thing calibration exists to measure. A prior naming a method with no backend is inert,
never a way to conjure a capability.

Enumerate with `registry.backends_for(stage)`, not `methods_for`: `gbm` and `brt` are one
backend, and treating aliases as distinct candidates spends half the compute twice.

### Not yet built

Proposer, selector, calibration harness, and web app. Phases 2-5 in `README.md`. Predictors
must already share a grid and CRS — resampling and reprojecting *predictors* is a typed gap
(`misaligned_predictors`), as is any raster format other than GeoTIFF/`.npy`.

## Ecosystem

Composes existing tested packages rather than reimplementing them: `lit-review`
(knowledge generation), `xferweight` (IWCV and covariate-shift weighting), `sdm_pipepy`
(real rasters, MESS, virtual species over real climate).
