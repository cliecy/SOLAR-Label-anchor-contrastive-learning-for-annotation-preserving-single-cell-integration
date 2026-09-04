# SOLAR: label-anchor contrastive learning for annotation-preserving single-cell integration

Reproducibility package accompanying the SOLAR manuscript. This is a minimal
package: the core model implementation, exact benchmark configurations,
final machine-readable results, integrity audits, and scripts that regenerate
selected tables and figures without access to the raw single-cell datasets.

## v1.1 audited external-baseline extension

v1.1 adds a three-real-dataset, whole-batch-holdout comparison without
changing any manuscript file:

- **SCLSC (main table):** nine held-out batches, model seeds 40--44
  (45 independently trained embeddings), reference-only internal
  train/validation splitting, common distance-weighted 15-NN annotation, the
  method-native unweighted 10-NN annotation readout, and official scib 0.2.0
  scoring.
- **Symphony (supplementary):** nine held-out batches at seed 40, official
  Harmony reference construction and Symphony query mapping, common 15-NN
  annotation, and official scib 0.2.0 scoring. This single-seed result does
  not support a variability claim.
- **scmap-cell (supplementary):** nine held-out batches at seed 40 using the
  native projection and abstention semantics. It is annotation-only and is
  not assigned embedding metrics.

Adding a method changes dataset-wise min-max scaling. The v1.1 builder
therefore recomputes the complete five-method main cohort and the separate
six-method supplementary embedding cohort; it never appends a newly scaled
row to a v1.0 table. See `configs/baseline_extension_v1_1.yaml`,
`data/baseline_extension_artifact_audit.json`,
`data/baseline_extension_scib_audit.json`, and
`data/baseline_extension_table_audit.json`.

Officially undefined metrics are retained as missing values, never imputed. If
one row lacks a metric, the table builder excludes that metric for every method
in the matching normalization group for aggregation A, B, or C. This preserves
equal metric coverage and identical composite denominators within each
comparison.

One SCLSC kBET job uses the contract-approved `scib-metrics 0.5.10`
Python fallback after the legacy implementation exhausted its diffusion
neighbor search. The audit identifies that single non-equivalent fallback,
and the package includes without-kBET sensitivity tables; SCLSC remains third
under every A/B/C ordering with or without kBET.

## v1.0.1 dimension-sweep correction

**Use v1.0.1 or later for every latent-dimension result.** In v1.0.0, the
`solar_orthogonal` variant fixed `embedding_dim=128`; because variant keyword
arguments took precedence over `TrainConfig`, jobs labelled 30D and 64D
actually emitted 128-dimensional embeddings. The primary 128D results and
non-dimension analyses were unaffected.

v1.0.1 removes that variant-level override, adds a non-default-dimension smoke
assertion, and replaces the labelled 30D/64D rows with independently trained
30D and 64D runs. All 50 corrected embeddings were dimension-checked before
official scib 0.2.0 scoring. The corrected sweep has 75 unique
(dataset, seed, dimension) rows: five datasets, seeds 40--44, and dimensions
30/64/128. The historical v1.0.0 release remains available for provenance but
must not be used to support a dimension-sensitivity claim.

See `PROVENANCE.md`, `data/training_audit.json`,
`data/scoring_audit.json`, and
`data/dimension_sensitivity_corrected_audit.json` for the correction trail.

## Install

Requires Python >= 3.10.

```bash
pip install -e .
# Include pandas, matplotlib, and pytest for scripts and checks:
pip install -e ".[scripts]"
```

Or use the included lock file:

```bash
uv sync --extra scripts
```

## Run the smoke example

The smoke example generates synthetic reference/query data, trains the
`solar_orthogonal` path with `embedding_dim=8`, and asserts that both returned
embeddings are actually 8-dimensional. No download is required; it runs on
CPU.

```bash
python examples/smoke_test.py
```

Expected final output: `SMOKE TEST PASSED`.

## Use SOLAR directly

```python
from SOLAR.benchmark import PreprocessConfig, TrainConfig, run_inductive

result = run_inductive(
    reference=adata_reference,   # AnnData; labels_key must be in .obs
    query=adata_query,           # held out; query labels are never read
    variant="solar_orthogonal",
    labels_key="cell_type",
    batch_key="batch",
    seed=40,
    anchor_seed=0,
    preprocess=PreprocessConfig(n_components=40, source="X"),
    train=TrainConfig(
        embedding_dim=128,       # set 30 or 64 for the corrected sweep settings
        max_epochs=50,
        batch_size=512,
    ),
)
result.save("outputs/solar_orthogonal_seed40")
```

The files in `configs/` are immutable records of the actual orchestration
settings:

- `benchmark_rebuttal_v1.yaml`: primary 128D run,
  `solar_scib_rebuttal_v1`.
- `benchmark_rebuttal_dim30.yaml`: corrected 30D run,
  `solar_scib_rebuttal_dim30_v2`.
- `benchmark_rebuttal_dim64.yaml`: corrected 64D run,
  `solar_scib_rebuttal_dim64_v2`.
- `baseline_extension_v1_1.yaml`: external-baseline source identities,
  preprocessing deviations, held-out batches, seed sets, readouts, and
  separate main/supplementary aggregation cohorts.

They are reference records, not direct entry points for this curated package.
A specific job can be recreated through `run_inductive` or the
`solar-benchmark` CLI with the same parameters.

## Regenerate reported outputs

All commands below read only files under `data/`; they do not retrain models or
rescore embeddings.

### Main held-out comparison table

```bash
python scripts/build_main_table.py
```

This recreates
`data/aggregation_sensitivity_3_real_datasets_with_kbet.csv` and companion
tables. Compare the result with
`data/aggregation_sensitivity_3_real_datasets_with_kbet_REFERENCE.csv` to
floating-point tolerance.

### v1.1 external-baseline tables

```bash
python scripts/build_baseline_extension_tables.py
```

Writes `data/main_table_v1_1.csv`,
`data/supplementary_table_v1_1.csv`, the with/without-kBET cohort
aggregates, and `data/baseline_extension_table_audit.json`. The command
rejects incomplete method/batch/seed coverage, changed source hashes,
non-unique SCLSC seed embeddings, failed artifact or metric audits, and an
official-scIB table that differs from its strict audit.

### Query-label-recovery figure

```bash
python scripts/build_figure.py
```

Writes `data/Figure_query_label_recovery.pdf` and `.png` for all nine real
held-out batches.

### Corrected dimension statistics and figure

```bash
python scripts/summarize_dimension_sensitivity.py
python scripts/build_dimension_figure.py
```

The first command recreates the corrected summary, paired-difference, and audit
files. The second writes `data/Figure_dimension_sensitivity.pdf` and `.png`,
using the same 75 raw observations as manuscript Figure 4A. Both scripts reject
missing, duplicate, or run-label/dimension-mismatched inputs.

## Data source and interpretation boundary

`data/raw_scib_metrics_all_runs.csv` contains final per-job official scib 0.2.0
metrics. Its corrected dimension rows expose manifest-derived `dimension`,
`expected_dimension`, and `actual_dimension` fields. The 30D and 64D rows come
from `solar_scib_rebuttal_dim30_v2` and `solar_scib_rebuttal_dim64_v2`; the
128D rows come from the valid primary run. `data/classifier_unified_summary.csv`
contains reference-trained, query-scored kNN results.
`data/baseline_extension_scib_metrics.csv` contains the 54 strict-audited
SCLSC/Symphony official-scIB jobs. `data/baseline_extension_query_metrics.csv`
contains the 108 external-baseline annotation rows, including separate common
and method-native SCLSC readouts. The v1.1 classifier file restores the
canonical local-final-evidence provenance columns; its 470 pre-existing metric
rows are value-identical to v1.0.1.

The corrected sweep does **not** support dimension invariance. Label ASW, graph
connectivity, and PCR were similar across dimensions, whereas iLISI increased
at lower dimension (mean 1.853 at 30D, 1.436 at 64D, and 1.071 at 128D across
25 dataset-seed observations). Treat latent dimension as a tuning parameter
that can alter the biological-preservation/batch-mixing profile. The
historical held-out comparison uses 128D SOLAR and 30D frozen-reference
baselines; v1.1 additionally uses the published SCLSC 16D architecture and a
40D Symphony reference. These are method comparisons, not
latent-dimension-matched experiments.

No raw single-cell expression matrices, per-cell embeddings, checkpoints, or
per-cell predictions are included. The three real normalized datasets are
from the public scIB Figshare record
[`10.6084/m9.figshare.12420968`](https://doi.org/10.6084/m9.figshare.12420968);
exact file hashes are retained in the artifact audit.

## Integrity

```bash
sha256sum -c MANIFEST.sha256
```

`MANIFEST.sha256` covers every package file except itself. Programming language:
Python only; see `PROVENANCE.md` for scope and exclusions.

## License

MIT; see `LICENSE`.
