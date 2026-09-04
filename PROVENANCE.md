# Provenance

This package is a minimal, curated subset of the internal SOLAR benchmark
repository. Result CSV files are copied unchanged from the canonical evidence
set; packaged analysis scripts only change paths so that they read `data/`
locally. No package-building step retrains a model or rescales a metric.

## v1.1 external-baseline record

The v1.1 extension uses the same nine whole-batch holdouts from the three real
scIB datasets as the existing held-out comparison. Source matrices, fixed
reference/query splits, and reference-selected 2,000-HVG lists are unchanged.
The exact contract is `configs/baseline_extension_v1_1.yaml`.

SCLSC uses upstream commit
`2e827dfebb793dffb548fd297e5c4e70fa40692f`, its official MLP,
instance-type contrastive loss, Adam optimizer, 16-dimensional output, and
seeded 90:10 stratified reference-internal train/validation design. Seeds
40--44 were independently trained for every held-out batch: 45 embeddings
total. Query expression was materialized only after fitting and model
selection; query labels were not visible to the model. Reliable raw counts are
not available across all three frozen inputs, so the shared normalized-X,
reference-only 2,000-HVG contract replaces the paper's count filtering,
Pearson-residual HVG selection, normalization, and log transform. The pinned
repository's patience-10 default is used; the paper reports patience 5, and the
discrepancy is explicit in every run record.

The artifact gate passed 63/63 runs: 45 SCLSC, nine Symphony, and nine
scmap-cell. All nine SCLSC batch groups contain exactly seeds 40--44 and five
distinct embedding SHA-256 values. Across the 45 SCLSC runs, the common
distance-weighted 15-NN readout has mean macro-F1 `0.7807353661` and mean
balanced accuracy `0.8525904974`; the method-native unweighted 10-NN
training-subset readout has `0.7697227259` and `0.8518399826`, respectively.

Symphony uses commit `6c8c4f2e9fa6bb6379627a28880da2857f405056`,
package 0.1.2, Harmony 1.2.4, and the official reference-building/query-mapping
path. scmap-cell uses Bioconductor source commit
`8fee1fcd119b066e3d9322f633b5d686b90cca47`, package 1.32.0, and native
abstention. Both are single-seed supplementary results (seed 40) and do not
support variability claims; scmap-cell is annotation-only.

Because scIB composite values are dataset-wise and cohort-scaled, the package
recomputes every method in the five-method main cohort and the separate
six-method supplementary embedding cohort under orderings A/B/C, both with
and without kBET. It does not splice a new row into the v1.0.1 scaled table.

Officially undefined metrics remain `not_applicable`; they are not assigned a
surrogate score. The table builder excludes such a metric for every method in
the matching A/B/C normalization group so that method-specific missingness
cannot change a composite denominator. In SCLSC's immune-cell `10X`, seed-43
embedding, scib 0.2.0 trajectory conservation has no valid root: all 445 cells
in the canonical HSPC start cluster lie outside the largest 1,969-cell
connected component among the 2,629 pseudotime-annotated cells. This
domain failure was reproduced twice and is retained as an explicit,
machine-audited `not_applicable` record rather than imputed.

One SCLSC immune-cell `Oetjen_A`, seed-43 job exhausted the legacy kBET
diffusion-neighbor search. The predeclared contract permits a
`scib-metrics 0.5.10` Python fallback only after that legacy failure; its kBET
value is `0.3001132616`, and the strict audit records the backend as
non-equivalent. Both with-kBET and without-kBET tables are included. SCLSC
ranks third under all A/B/C orderings in both tables, so its main-table
placement does not depend on this fallback.

## v1.0.1 correction record

The v1.0.0 package exposed a latent-dimension configuration defect. The
`solar_orthogonal` variant supplied `embedding_dim=128`, and
`run_inductive()` resolved variant keyword arguments before the fallback to
`TrainConfig.embedding_dim`. Consequently, jobs labelled 30D and 64D emitted
128-dimensional embeddings. This was detected by inspecting saved
`status.json` files and embedding shapes. The primary 128D run was configured
for 128D and is unaffected.

The correction is a clean replacement, not relabelling:

1. Removed the variant-level `embedding_dim=128` override so the configured
   training dimension is authoritative.
2. Added a regression check that requests 8D and asserts 8D model kwargs and
   output embeddings.
3. Repacked the exact corrected source as
   `SCMBench_SOLAR_label_anchor_benchmark_kit_20260902_dimfix.tar.gz`, SHA-256
   `d000915f9d39ed95a8b5c5422ebe2a60b7054d16ec79951d77d487c6ab0a704c`.
4. Trained 25 independent 30D jobs and 25 independent 64D jobs (five datasets
   x seeds 40--44) in new output directories. Every saved embedding had the
   requested second dimension; all 25 paired 30D/64D embedding files had
   distinct SHA-256 hashes.
5. Recomputed every embedding-dependent official scib 0.2.0 metric, generated
   fresh run audits and provenance snapshots, and rebuilt the 550-row canonical
   evidence table.

The old run directories and v1.0.0 release are retained as historical evidence
of the correction. They are superseded and must not be used for dimension
claims.

## Source commits

Internal repository: `https://github.com/cliecy/scib-reproduce`.

- Frozen v2 experimental/evidence base:
  `f62c4b155c977e59307d53f5d5ec1bd718c6674f`.
- Deterministic model-initialization seed fix: `3fdacca`.
- Dimension override removal and regression test:
  `a2866ace36e4656346a427792ca0cf4ae98c82cd`.
- Corrected source archive and 30D/64D configs:
  `ed7f12f1e664a989d511b3d4f0d344a2205ffd6a`.
- Fifty-job embedding-shape/hash audit:
  `04f42b32b1799136122f5c1f46020897e3a3c02f`.
- Config-aware source-provenance capture:
  `8df95fa3eb5892b3c9b0b3bd651f3d3e8933524f`.
- Corrected raw table, dimension summaries, and 550-job audit:
  `c365a37828421073ce083552d80f1002580a68f1`.
- External-baseline adapters, locked environments, and Slurm contracts:
  `67b23e44e722e17c1d052fbf1a8be4bfc83465f1`.
- Explicit base/extension source identity:
  `d42fdc5`.
- Order-correct reconstruction of the persisted SCLSC internal split audit:
  `ad2d212`.
- Explicit trajectory-root domain guards for both scoring paths:
  `60fb0f6` and `f312205`.
- Final strict applicability and approved-fallback contract:
  `1ddc864`.
- Audited external-baseline summary outputs:
  `3bf5043`.

The fixed `src/SOLAR/benchmark/variants.py` and the non-default-dimension
assertion in `examples/smoke_test.py` are included here.

## Corrected run identity

| Requested dimension | Canonical run ID | Jobs | Official scib state | Cohort SHA-256 |
|---:|---|---:|---|---|
| 30 | `solar_scib_rebuttal_dim30_v2` | 25 | 25 completed; audit passed | `d5bb01920ac3aa2d8103544fe4cda28b63f03bc063f506682bb379cd649beeab` |
| 64 | `solar_scib_rebuttal_dim64_v2` | 25 | 25 completed; audit passed | `18e6ec20bfdbebdac32602900c4656cd66dbba7eb988f203a8e0caa07fe135e2` |
| 128 | `solar_scib_rebuttal_v1` | 25 core SOLAR jobs | previously verified | primary cohort |

Official metrics used scib commit
`e2a37e0ed63dc34b60aa535cc656400552af757a`, kBET commit
`afc5f431bcbefd73267acc066a0f2e4eaa10a355`, and paper-environment lock
SHA-256 `9fbdc1dd3a9c39607163dc2dfd81f5fb7055d8adde5c485d7a092b85fa504e00`.
See `data/training_audit.json` and `data/scoring_audit.json` for per-run proof.

## Package contents

- `src/SOLAR/`: core model, anchor bank, loss, reference-only training, and
  query-mapping adapter.
- `configs/`: primary 128D, corrected 30D/64D, and v1.1
  external-baseline orchestration records.
- `data/raw_scib_metrics_all_runs.csv`: canonical 550-row per-job table with
  corrected dimension rows and explicit expected/actual dimensions.
- `data/classifier_unified_summary.csv`: query-only classifier results.
- `data/baseline_extension_*`: raw external-baseline query/scIB metrics,
  per-class results, strict artifact/scoring audits, regenerated main and
  supplementary tables, aggregation sensitivities, and table-build audit.
- `data/dimension_sensitivity_corrected_*.{csv,json}`: 75-row sweep summaries,
  paired differences, and integrity assertions.
- `scripts/`: self-contained table and figure regeneration utilities.
- `examples/smoke_test.py`: synthetic end-to-end run that specifically guards
  configurable output dimension. It is not a scientific result.

## Deliberate exclusions

The package excludes raw expression matrices, per-cell embeddings, checkpoints,
full Slurm orchestration outputs/logs, internal peer-review correspondence,
historical superseded dimension outputs, and nested archives. These are not
needed to install SOLAR or reproduce the included summaries and figures from
machine-readable results. Per-embedding identities and split/seed assertions
are retained in `data/training_audit.json` and
`data/baseline_extension_artifact_audit.json`.

## Programming languages

Packaged executable source is Python only. The Symphony and scmap-cell
supplementary data were generated with R 4.5.3, Symphony 0.1.2/Harmony 1.2.4,
and scmap 1.32.0; those heavy execution environments are not bundled. Official
scib/kBET software identities remain recorded above and in the scoring audits.
