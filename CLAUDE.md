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

### Not yet built

Proposer, executor, selector, calibration harness, artifact store, and web app. Phases 1-5
in `README.md`.

## Ecosystem

Composes existing tested packages rather than reimplementing them: `lit-review`
(knowledge generation), `xferweight` (IWCV and covariate-shift weighting), `sdm_pipepy`
(real rasters, MESS, virtual species over real climate).
