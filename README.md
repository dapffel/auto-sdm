# auto-sdm

An autonomous agentic research component for species distribution modeling.

You supply occurrence data, predictors, and how you intend to use the model. The system searches
over modeling methodologies — algorithms, cross-validation designs, background strategies,
predictor treatments — executes the candidates, and returns the one best suited to your data and
your stated use, with the evidence for why.

It is not a pipeline that runs a model. It is a **search that selects one**.

> **Status: design phase.** Nothing in this repository runs yet. Several components it depends on
> already exist and are tested — see [Ecosystem](#ecosystem).

---

## The hard problem: selection without ground truth

Ordinary AutoML picks the candidate with the best held-out score. In SDM that is actively
misleading. There are no true absences and the real range is unknown — only where people looked.
Held-out AUC on presence data largely measures agreement with the *sampling process*, so the
winner is frequently the model that best learned where surveys happened. A search that optimizes
it will confidently return the wrong model, and the report will look excellent.

Selection is therefore the load-bearing component, and it is composite:

- **Spatial block cross-validation**, never random. Autocorrelation does not inflate candidates
  uniformly, so random CV biases the *ranking*, not merely the scores.
- **Boyce index / CBI** for presence-only data, where AUC means least.
- **Importance-weighted CV** — estimates performance under the *target* distribution, which is the
  real question whenever the model will be applied somewhere.
- **Extrapolation penalty** — MESS/ExDet. A candidate whose ranking depends on novel environmental
  space is untrustworthy regardless of its score.
- **Fold stability** — a candidate that wins on the mean but swings across blocks is worse than
  its mean implies.

### Best for what?

Different intended uses select genuinely different models, so intended use is a required input,
never a default:

| Objective | What dominates selection |
|---|---|
| **Spatial transfer** *(default)* | importance-weighted CV, extrapolation penalty |
| Interpolation within the sampled region | CBI, fold stability |
| Temporal projection | extrapolation penalty, response-curve plausibility |

## Calibrating the selector

The claim "this is the best model" is only worth as much as the rule that chose it. Virtual
species make that rule measurable.

Because a simulated species has a known true niche, the whole search can be run over simulated
data and the selector's pick compared against the truth-optimal candidate:

```
selector_regret = score_true(selector_pick) − score_true(best_candidate)
```

Aggregated across simulated conditions — sampling bias, niche breadth, sample size, predictor
misspecification — this yields a calibrated answer to a question that is unanswerable on real
data: *do these selection criteria actually recover ecological truth?*

Selection rules become a tunable component with a measurable objective. This is what the learning
loop learns.

## Architecture

```
                  user data + intended use
                            │
                            ▼
                       PROPOSER ─────── literature corpus (what papers do)
                   candidate plans              run history (what worked before)
                            │
                            ▼
                       EXECUTOR ─────── Python backend │ R backend
                            │           per-candidate artifacts + step cards
                            ▼
                       SELECTOR ─────── spatial CV · CBI · IWCV
                            │            extrapolation · stability
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
     best model + ODMAP report     ledger → capability gaps
     + honest uncertainty                  → selector calibration
                                                    ▲
                                        virtual-species benchmark
                                            (measures regret)
```

### Backends are capability providers

Not a language choice. A candidate plan names a method; a registry resolves it to whichever
backend implements it:

```
"maxent"      → Python  (maxnet-style penalized logistic over expanded features)
"brt" / "gbm" → Python  (xgboost)
"GAM"         → R       (mgcv)
"MARS"        → R       (earth)
```

Methods with no registered backend are recorded as `UnsupportedStep` — **never silently
substituted** — and aggregate into a ranked capability backlog. R backends are added per method
when the backlog justifies it, so search covers R-only algorithms without imposing an R runtime
on day one.

### Nothing is silently substituted

Every fallback is a typed `Assumption`; every unimplementable method is a typed `UnsupportedStep`.
Both carry stable codes, so they aggregate across runs into a data-driven backlog: which
capability, if built, would unlock the most real methodology. This is the same mechanism applied
to the system's own development.

### The run record

Each candidate and each stage emits:

- **`artifacts/`** — rasters, fitted models, plots; content-addressed
- **`step_record.json`** — inputs, resolved backend, parameters, rationale, metrics, seeds
- **`step_card.html`** — the human-readable "what happened here"

Plus a run-level `provenance.json`: data hashes, package versions, every seed. Runs are
reproducible, or cross-run learning is learning from noise.

### Autonomy with rewind

Searches run end-to-end unattended. Because every stage is checkpointed with its inputs fully
captured, you can inspect any step card, override a decision, and resume from that point —
downstream stages invalidate and re-run.

## Ecosystem

auto-sdm composes existing, independently tested packages rather than reimplementing them:

| Package | Role |
|---|---|
| [`lit-review`](https://github.com/dapffel/lit-review) | Extracts structured SDM methodology from papers. LangGraph pipeline, provider-agnostic via LiteLLM, run-history RAG. Feeds the proposer. |
| [`virtual-sdm-runner`](https://github.com/dapffel/virtual-sdm-runner) | Compiles methodology into reproducible virtual-species experiments with known truth. The benchmark and the execution contract. |
| [`xferweight`](https://github.com/dapffel/xferweight) | Importance weighting for covariate shift — Shimodaira weights, IWCV, effective-sample-size diagnostics. The transferability layer. |
| [`sdm_pipepy`](https://github.com/dapffel/sdm_pipepy) | Real-raster machinery: WorldClim loading, spatial partitioning, MESS novelty. |

## Stack

- **Orchestration** — LangGraph with checkpointing
- **Model gateway** — LiteLLM; provider-agnostic from day one
- **Modeling** — Python-native core (scikit-learn, xgboost, numpy); R backends per method as the
  capability backlog justifies
- **Web app** — FastAPI + React: upload → live search → candidate comparison → ODMAP report
- **Target environment** — single machine

## Roadmap

| Phase | Scope |
|---|---|
| **0** | Generalize the execution contract to admit real data; backend registry; per-step artifacts and cards |
| **1** | Search over a small space (3 algorithms × 2 CV designs × 2 background strategies) on real data with a fixed selector. End to end, CLI. |
| **2** | **Selector calibration** — run the search over virtual species, measure regret per rule, adopt the defensible composite |
| **3** | Real-data depth — cleaning, accessible area (M), predictor selection, ODMAP report generation |
| **4** | R backends for the top capability gaps; web application |
| **5** | Learning loop closes — proposer conditions on run history, selector weights tuned by measured regret |

Phase 2 precedes the depth work deliberately: a broader search under an uncalibrated selector only
picks wrong more elaborately.

## License

TBD
