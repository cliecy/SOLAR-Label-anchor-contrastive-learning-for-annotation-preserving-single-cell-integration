# SOLAR: label-anchor contrastive learning for annotation-preserving single-cell integration

Reproducibility package accompanying the SOLAR manuscript. This is a
minimal package: the core model implementation, the configuration and
final machine-readable results behind the manuscript's main held-out-batch
comparison, and the scripts to regenerate one results table and one figure
from that data. See `PROVENANCE.md` for exact source commits and what was
deliberately left out.

## Install

Requires Python >= 3.10.

```bash
pip install -e .
# or, to also install the pandas/matplotlib/pytest needed for scripts/ and tests:
pip install -e ".[scripts]"
```

Or with `uv` (uses the included `uv.lock` for an exact, reproducible
environment):

```bash
uv sync --extra scripts
```

## Run the smoke example

Confirms the package installs and the core training/inference path works,
on tiny synthetic data (no download required, runs in well under a minute
on CPU):

```bash
python examples/smoke_test.py
```

Expected final output: `SMOKE TEST PASSED`.

## Use SOLAR directly

```python
from SOLAR.benchmark import run_inductive, PreprocessConfig, TrainConfig

result = run_inductive(
    reference=adata_reference,   # AnnData, must contain `labels_key` in .obs
    query=adata_query,           # AnnData, held out -- its labels are never read
    variant="solar_orthogonal",  # the manuscript's primary configuration
    labels_key="cell_type",
    batch_key="batch",
    seed=40,                     # one of the manuscript's split seeds, 40-44
    anchor_seed=0,
    preprocess=PreprocessConfig(n_components=40, source="X"),
    train=TrainConfig(max_epochs=50, batch_size=512),
)
result.save("outputs/solar_orthogonal_seed40")
```

`configs/benchmark_rebuttal_v1.yaml` records the exact configuration
(preprocessing profile `hvg2000_pca40`, split seeds, `anchor_seed=0`) used
to produce the primary reported results. It is a reference record of the
internal job-orchestration configuration, not an entry point this package
runs directly — reproduce a specific job with `run_inductive`/`solar-benchmark`
above using the same parameters.

## Regenerate a results table and a figure

Both scripts below only read from `data/` (already-scored, final
machine-readable tables — no raw single-cell data, no GPU, seconds to run):

```bash
pip install -e ".[scripts]"   # if not already installed
python scripts/build_main_table.py
```

This reproduces `data/aggregation_sensitivity_3_real_datasets_with_kbet.csv`
(and two companion tables); compare it to the included
`data/aggregation_sensitivity_3_real_datasets_with_kbet_REFERENCE.csv` —
they should match to floating-point tolerance.

```bash
python scripts/build_figure.py
```

Writes `data/Figure_query_label_recovery.pdf` and `.png`: query-label
recovery (macro-F1, balanced accuracy) across all 9 real held-out batches,
for matched-PCA / SOLAR / frozen-reference scVI / frozen-reference scANVI.

## Data source

`data/raw_scib_metrics_all_runs.csv` and `data/classifier_unified_summary.csv`
are final, per-job scored outputs (official scib 0.2.0 metrics; a
reference-trained, query-scored kNN classifier) from the manuscript's
held-out-batch evaluation protocol. No raw single-cell expression data is
included in this package. The public single-cell datasets used to produce
these results are cited in the manuscript.

## Integrity

```bash
sha256sum -c MANIFEST.sha256
```

`MANIFEST.sha256` covers every file in this package except itself.
Programming language: **Python only** — see `PROVENANCE.md` for why no R
environment is included.

## License

MIT — see `LICENSE`.
