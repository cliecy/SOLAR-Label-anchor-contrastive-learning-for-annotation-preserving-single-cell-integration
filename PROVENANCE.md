# Provenance

This package is a minimal, curated subset of the internal working
repository for the SOLAR manuscript. Nothing in it was modified from the
originating repository except path edits needed to make the packaged
scripts self-contained (documented inline where they occur), and no result
in `data/` was recomputed for this package -- every CSV here is copied
unchanged from the internal result set.

## Source commits

- **Original source commit**: `f62c4b155c977e59307d53f5d5ec1bd718c6674f`
  (internal repository `scib_reproduce`).
- **Seed-fix commit**: `3fdacca` (internal repository `scib_reproduce`).
  This commit moved the deterministic-seeding call
  (`_seed_everything(seed)`) to immediately before `SOLARModel(...)` is
  constructed in `SOLAR/benchmark/adapter.py`'s `run_inductive()`. Before
  this fix, the model's expression encoder was weight-initialized before
  any seed was set, so `seed` did not fully control initialization.
  **This fix is present in `src/SOLAR/benchmark/adapter.py` in this
  package** — see lines around the `_seed_everything(seed)` call
  immediately preceding the `SOLARModel(...)` construction.

## What this package represents

- **Core implementation** (`src/SOLAR/`): the label-anchor contrastive
  learning model (encoder, anchor bank, loss), the reference-only training
  and query-mapping adapter (`run_inductive`), and the CLI entry point.
  Self-contained — its only third-party dependencies are `anndata`, `h5py`,
  `numpy`, `scipy`, `scikit-learn`, and `torch` (see `pyproject.toml`).
- **Configuration** (`configs/benchmark_rebuttal_v1.yaml`): the benchmark
  configuration actually used to produce the primary held-out-batch results
  reported for the manuscript's main comparison (`solar_orthogonal`
  variant, `hvg2000_pca40` preprocessing profile, split seeds 40-44,
  `anchor_seed=0`).
- **Results data** (`data/*.csv`): final, already-scored machine-readable
  tables — official scib-metric values per job
  (`raw_scib_metrics_all_runs.csv`) and the query-only classifier summary
  (`classifier_unified_summary.csv`) — copied unchanged from the internal
  canonical result set. `aggregation_sensitivity_3_real_datasets_with_kbet_REFERENCE.csv`
  is the expected output of `scripts/build_main_table.py`, included so a
  reader can diff their own regenerated table against it without having to
  trust a fresh run blindly.
- **Scripts** (`scripts/`): `build_main_table.py` regenerates the
  aggregation-sensitivity results table from `data/raw_scib_metrics_all_runs.csv`;
  `build_figure.py` regenerates the query-label-recovery figure from
  `data/classifier_unified_summary.csv`. Both are self-contained given only
  the files in `data/`.
- **Example** (`examples/smoke_test.py`): a toy synthetic-data run of the
  core `run_inductive` API, to confirm the package installs and runs
  end-to-end. Not a scientific result.

## What is deliberately excluded

Per the minimal-disclosure principle for this package: raw per-cell
embeddings, model checkpoints, the full ~500-job Slurm orchestration
pipeline and its logs, the internal rebuttal/peer-review correspondence and
audit documents, superseded pre-seed-fix results, the historical literal
one-hot anchor diagnostic, and duplicate/nested archives. None of these are
required to install the core implementation, run the smoke example, or
regenerate the included results table and figure from the included data.

## Programming languages

**Python only.** No R script or R environment is included in this package:
FastMNN (the one baseline in the internal repository that uses R via
Bioconductor's `batchelor::fastMNN`) does not appear in the main-cohort
methods reproduced by `scripts/build_main_table.py` or
`scripts/build_figure.py` (see `data/raw_scib_metrics_all_runs.csv`'s
`method` column), so no R dependency is needed to install, run, or verify
anything in this package.
