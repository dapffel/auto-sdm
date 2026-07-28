# auto-sdm

An autonomous multi-agent system for building species distribution models (SDMs).

You supply occurrence data and predictors. The system profiles it, cleans it, delineates the
accessible area, selects predictors, splits spatially, trains, evaluates, projects, and returns a
reproducible report — deciding for itself at each step, and explaining why.

> **Status: pre-alpha.** Design phase. Nothing here runs yet.

---

## Why this is hard (and what the design is built around)

**In SDM you never observe the ground truth.** There are no true absences, and the real range is
unknown — you only see where people looked. A model can score 0.92 AUC by learning the distribution
of roads and national parks. Every design decision below follows from taking that seriously:

- Agents **select and parameterize deterministic recipes** rather than authoring raw geospatial
  code. Numerics that are silently wrong are worse than numerics that fail loudly.
- Every step is judged by a **critic agent** against an explicit rubric, not by whether it ran.
- Correctness is measured against a **benchmark of virtual species with known true niches** — the
  only place ground truth actually exists.

## Pipeline

Each stage is a node in a LangGraph state machine. The graph branches on the detected data regime
(presence-only vs. presence-absence), which changes background sampling, algorithm choice, and
calibration metrics.

| # | Stage | The judgment call | Literature anchor |
|---|-------|-------------------|-------------------|
| 1 | Ingest & Profile | CRS, taxonomy resolution, data regime detection | GBIF / DarwinCore |
| 2 | Clean | thinning distance, outlier rules, coordinate validity | CoordinateCleaner, spThin |
| 3 | Accessible area (M) | extent delineation — high-impact, chronically under-reported | Barve et al. 2011 (BAM) |
| 4 | Predictors | collinearity, variable selection, background sampling | VIF/PCA, target-group background |
| 5 | Split | spatial block CV; block size from autocorrelation range | blockCV, Roberts et al. 2017 |
| 6 | Train | algorithm set, tuning grid, ensemble weighting | ENMeval, flexsdm, biomod2 |
| 7 | Evaluate | metric set, threshold rule | AUC/TSS **+ Boyce/CBI**, omission rate |
| 8 | Project & Report | transfer extent, extrapolation flagging | MESS/ExDet, ODMAP |

Reporting follows **ODMAP** (Zurell et al. 2020) — it defines what each step must record, and
doubles as the critics' rubric.

## Architecture

### Worker ↔ Critic

Every stage is a pair, never a lone agent. The worker proposes a decision plus rationale; the critic
scores it against a stage-specific rubric; a bounded revise loop follows; then pass or escalate.
Critic verdicts are the highest-value signal the learning loop consumes.

### Skills are executable, not prompts

A method extracted from the literature is registered as:

```
skills/<name>/
  SKILL.md        # when to use, provenance (DOI), assumptions, failure modes
  recipe.R        # deterministic, typed JSON in → JSON out
  schema.json     # the parameter space the agent may choose within
  applicability   # data regimes, data-size bounds, required predictors
  fixture/        # golden input + expected output; must pass to register
```

The agent's judgment goes into *which* recipe and *which* parameters — the part that genuinely
requires having read the literature. The numerics stay deterministic, versioned, and testable.

### The run record

Each stage emits three things:

- **`artifacts/`** — rasters, fitted models, plots; content-addressed
- **`step_record.json`** — inputs, recipe + version, parameters, rationale, critic score, metrics,
  seeds, timings. This is the learning corpus.
- **`step_card.html`** — the human-readable "what happened here", rendered from the record

Plus a run-level `provenance.json`: data hashes, container digest, package versions, every seed.
Runs are byte-reproducible, or cross-run learning is learning from noise.

### Autonomy with rewind

Runs execute end-to-end unattended. Because every stage is checkpointed and its inputs fully
captured, you can inspect any step card, override a decision, and resume from that node —
downstream stages invalidate and re-run.

### Learning loop

Two feeds, both gated by the benchmark:

1. **Run experience** — retrieval over past `step_record.json`. At decision time the worker sees
   what worked on comparable datasets. Retrieval-augmented priors, not weight updates.
2. **Paper ingestion** — an agent reads a new SDM paper and proposes a *diff*: a new recipe, a
   changed parameter range, a rubric update.

Proposals land as pull requests and must not degrade the benchmark. Self-modifying tooling without
a measured external referent degrades quietly; the gate is not optional.

### Benchmark

- **Virtual species** — simulated from known response curves and sampled with known bias. The
  question becomes "did it recover the true niche", which no real dataset can answer.
- **Published datasets with independent evaluation data** — real-world sanity.
- **Regression suite** — once recipes are mutable, fixed runs must not silently degrade.

## Stack

- **Orchestration** — LangGraph, SQLite checkpointer
- **Model gateway** — single `LLMClient` interface; Claude Opus 4.8 by default, provider-agnostic
  seam from day one
- **Modeling kernel** — R in `rocker/geospatial` (pinned digest), driven as a typed script runner:
  `flexsdm`, `ENMeval`, `blockCV`, `terra`
- **Web app** — FastAPI + React: upload → live run stream → step cards → ODMAP report
- **Target environment** — single machine, local Docker

## Roadmap

| Phase | Scope |
|-------|-------|
| **0** | Contracts: `StepRecord` / `RunManifest` schemas, R bridge, artifact + provenance store |
| **1** | Thin vertical slice — one species, presence-only, MaxEnt, all 8 nodes with critics and cards, CLI only, minimal virtual-species benchmark |
| **2** | Depth — presence-absence regime, ensembles, tuning, spatial CV strategies, ODMAP report generator, full benchmark |
| **3** | Web application |
| **4** | Learning loop and paper ingestion, behind the benchmark gate |

## License

TBD
