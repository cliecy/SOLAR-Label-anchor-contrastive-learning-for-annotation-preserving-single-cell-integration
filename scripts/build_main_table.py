#!/usr/bin/env python3
"""Aggregation sensitivity analysis (v2), rewritten per an explicit
methodology review. This is a hotfix to the analysis code only -- no model
is retrained, no hyperparameter changes, no cohort changes based on results.

MAIN COHORT (the primary, matched comparison): solar_orthogonal,
unintegrated_pca_matched, frozen_reference_scvi, frozen_reference_scanvi --
held-out-batch track, the 3 REAL datasets only (pancreas, immune_cell_human,
lung_atlas). frozen_reference_scvi/scanvi do not have matched 5-dataset,
5-seed coverage (only reseeded on the 3 real datasets), so "all 5 datasets"
is never used for this comparison anywhere in this script or its output --
only reported separately, explicitly labeled a sanity check.

solar_none and solar_orthogonal_uniq are SOLAR-family ablations, reported
separately and NEVER added to the main cohort's min-max scaling pool (doing
so would change every other method's scaled score too, since min-max scaling
is cohort-dependent).

PAIRING UNIT: dataset x heldout_batch x split_seed. On the held-out-batch
track, held_out_batch_split() takes no seed argument -- the same cells are
reference/query for every model_seed at a given (dataset, heldout_batch), so
split_seed is NA (constant) for every row here; the real per-trial pairing
key is (dataset, heldout_batch, model_seed) for the 3 stochastic methods.
unintegrated_pca_matched has no model_seed (deterministic PCA reusing the
matching split's already-fitted, reference-only components/mean) and is
broadcast into every model_seed's comparison group rather than being left
in an arbitrary, unrelated "seed 0" bucket.

THREE GENUINELY DIFFERENT ORDERINGS (A and C do not collapse to the same
formula at different pooling depths, per the fix requested):
  A. Per-(dataset, heldout_batch, model_seed) scaling: min-max scale across
     the method cohort within EACH replicate (batch x seed) group, compute
     the composite there, then average composites across model_seeds and
     batches.
  B. Seed-mean-then-scale: average raw metrics across model_seeds FIRST
     within (dataset, heldout_batch) for the 3 stochastic methods (one row
     per batch x method), combine with PCA's existing one row per batch,
     THEN min-max scale across methods within (dataset, heldout_batch).
  C. Dataset-level: average raw metrics across BOTH heldout_batch AND
     model_seed simultaneously within (dataset, method) -- one row per
     (dataset, method) -- THEN min-max scale across methods within the
     whole DATASET (not within batch). This scales at a coarser grain than
     A/B by construction, not merely by a different final pooling step.

METRIC DIRECTION: every metric in BATCH_METRICS/BIOLOGY_METRICS
(scripts/scib_contract.py) is "higher is better" and min-max
scaled with no inversion -- this includes kBET. Verified directly against
scripts/paper_scib_worker.py's kBET computation (see
KBET_DEFINITION below), not merely assumed from the aggregation code.
`higher_is_better` is written as an explicit metadata column into every
output table.

MISSING METRICS: skipna=True throughout; hvg_overlap is disabled for
embedding outputs (not_applicable, not a failure) and contributes nothing
to batch_correction, never silently treated as 0.

Also computes a without-kBET sensitivity variant of the composite (same
three orderings, kBET dropped from BATCH_METRICS) to check whether the
ranking depends specifically on that one metric.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scib_contract import BATCH_METRICS, BIOLOGY_METRICS  # noqa: E402

KBET_DEFINITION = (
    "scripts/paper_scib_worker.py (task == 'kbet'): per_label = scIB.me.kBET(...); "
    "value = 1.0 - np.nanmean(per_label['kBET']). This is the official scib 0.2.0 "
    "kBET score as defined in the scib paper (Luecken et al. 2022): 1 minus the "
    "mean per-label rejection-rate-derived statistic returned by the R kBET "
    "package, i.e. higher = better mixing (well-mixed neighborhoods = low "
    "rejection rate = high final value). This matches, and is the ground truth "
    "for, the 'higher is better, no inversion' convention already used in "
    "scripts/scib_contract.py's BATCH_METRICS grouping -- the two "
    "were cross-checked and are consistent, not merely assumed to agree."
)

REAL_DATASETS = ["pancreas", "immune_cell_human", "lung_atlas"]
SIM_DATASETS = ["simulation_1", "simulation_2"]

MAIN_COHORT = ["solar_orthogonal", "unintegrated_pca_matched", "frozen_reference_scvi", "frozen_reference_scanvi"]
STOCHASTIC_MAIN = ["solar_orthogonal", "frozen_reference_scvi", "frozen_reference_scanvi"]
ABLATIONS = ["solar_none", "solar_orthogonal_uniq"]

METRIC_METADATA = pd.DataFrame(
    [{"metric": m, "group": "batch_correction", "higher_is_better": True} for m in BATCH_METRICS]
    + [{"metric": m, "group": "bio_conservation", "higher_is_better": True} for m in BIOLOGY_METRICS]
)


def rescale(values: pd.Series) -> pd.Series:
    finite = values.dropna()
    if finite.empty:
        return pd.Series(np.nan, index=values.index)
    lo, hi = finite.min(), finite.max()
    if np.isclose(lo, hi):
        return values.notna().astype(float).where(values.notna(), np.nan) * 0 + 0.5
    return (values - lo) / (hi - lo)


def composite(frame: pd.DataFrame, batch_metrics: list[str], bio_metrics: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    present_batch = [m for m in batch_metrics if m in frame.columns]
    present_bio = [m for m in bio_metrics if m in frame.columns]
    frame["batch_correction"] = frame[present_batch].mean(axis=1, skipna=True)
    frame["bio_conservation"] = frame[present_bio].mean(axis=1, skipna=True)
    frame["overall"] = 0.4 * frame["batch_correction"] + 0.6 * frame["bio_conservation"]
    return frame


def load_heldout_data() -> pd.DataFrame:
    df = pd.read_csv("data/raw_scib_metrics_all_runs.csv")
    metric_cols = {c: c.replace("metric__", "") for c in df.columns if c.startswith("metric__")}
    df = df.rename(columns=metric_cols)

    solar_rows = df[
        (df["method"].isin(MAIN_COHORT + ABLATIONS))
        & (df["run"] == "primary_128d")
        & (df["track"] == "heldout")
    ]
    pca_rows = df[(df["method"] == "unintegrated_pca_matched") & df["heldout_batch"].notna()]
    frozen_rows = df[df["method"].isin(["frozen_reference_scvi", "frozen_reference_scanvi"])]
    out = pd.concat([solar_rows, pca_rows, frozen_rows], ignore_index=True)

    keep = ["dataset_id", "method", "heldout_batch", "split_seed", "model_seed", "job_id"] + list(BATCH_METRICS) + list(BIOLOGY_METRICS)
    keep = [c for c in keep if c in out.columns]
    return out[keep]


def build_replicates(df: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    """One row per (dataset, heldout_batch, model_seed, method), with the
    deterministic unintegrated_pca_matched row broadcast into every
    model_seed slot for its (dataset, heldout_batch) -- it has no
    model_seed of its own, so it cannot be paired any other way without
    either dropping it or inventing a fake seed."""
    d = df[df["dataset_id"].isin(datasets)]
    stochastic = d[d["method"].isin(STOCHASTIC_MAIN + ABLATIONS)].copy()
    pca = d[d["method"] == "unintegrated_pca_matched"].copy()

    seeds = sorted(stochastic["model_seed"].dropna().unique().tolist())
    broadcast_rows = []
    for _, pca_row in pca.iterrows():
        for seed in seeds:
            row = pca_row.copy()
            row["model_seed"] = seed
            broadcast_rows.append(row)
    pca_broadcast = pd.DataFrame(broadcast_rows)
    return pd.concat([stochastic, pca_broadcast], ignore_index=True)


def ordering_a(df: pd.DataFrame, datasets: list[str], cohort: list[str], batch_metrics, bio_metrics) -> pd.DataFrame:
    replicates = build_replicates(df, datasets)
    replicates = replicates[replicates["method"].isin(cohort)]
    rows = []
    for (dataset_id, batch, seed), group in replicates.groupby(["dataset_id", "heldout_batch", "model_seed"]):
        scaled = group.copy()
        for m in batch_metrics + bio_metrics:
            if m in scaled.columns:
                scaled[m] = rescale(scaled[m])
        scaled = composite(scaled, batch_metrics, bio_metrics)
        for _, r in scaled.iterrows():
            rows.append({"dataset_id": dataset_id, "heldout_batch": batch, "model_seed": seed,
                         "method": r["method"], "overall": r["overall"]})
    per_replicate = pd.DataFrame(rows)
    per_dataset_method = per_replicate.groupby(["dataset_id", "method"])["overall"].mean().reset_index()
    return per_dataset_method


def ordering_b(df: pd.DataFrame, datasets: list[str], cohort: list[str], batch_metrics, bio_metrics) -> pd.DataFrame:
    d = df[df["dataset_id"].isin(datasets) & df["method"].isin(cohort)]
    metric_cols = [c for c in batch_metrics + bio_metrics if c in d.columns]
    # average across model_seed first (within dataset x heldout_batch x method);
    # PCA has no model_seed so this is a no-op for it (single row already).
    seed_mean = d.groupby(["dataset_id", "heldout_batch", "method"])[metric_cols].mean().reset_index()
    rows = []
    for (dataset_id, batch), group in seed_mean.groupby(["dataset_id", "heldout_batch"]):
        scaled = group.copy()
        for m in metric_cols:
            scaled[m] = rescale(scaled[m])
        scaled = composite(scaled, batch_metrics, bio_metrics)
        for _, r in scaled.iterrows():
            rows.append({"dataset_id": dataset_id, "heldout_batch": batch, "method": r["method"], "overall": r["overall"]})
    per_batch = pd.DataFrame(rows)
    return per_batch.groupby(["dataset_id", "method"])["overall"].mean().reset_index()


def ordering_c(df: pd.DataFrame, datasets: list[str], cohort: list[str], batch_metrics, bio_metrics) -> pd.DataFrame:
    d = df[df["dataset_id"].isin(datasets) & df["method"].isin(cohort)]
    metric_cols = [c for c in batch_metrics + bio_metrics if c in d.columns]
    # collapse BOTH heldout_batch and model_seed simultaneously -- one row
    # per (dataset, method) BEFORE any scaling happens (genuinely coarser
    # than A/B, not just a different pooling of an already-scaled result).
    dataset_mean = d.groupby(["dataset_id", "method"])[metric_cols].mean().reset_index()
    rows = []
    for dataset_id, group in dataset_mean.groupby("dataset_id"):
        scaled = group.copy()
        for m in metric_cols:
            scaled[m] = rescale(scaled[m])
        scaled = composite(scaled, batch_metrics, bio_metrics)
        rows.append(scaled[["dataset_id", "method", "overall"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["dataset_id", "method", "overall"])


def run_all_orderings(df: pd.DataFrame, datasets: list[str], cohort: list[str], batch_metrics, bio_metrics) -> pd.DataFrame:
    a = ordering_a(df, datasets, cohort, batch_metrics, bio_metrics).groupby("method")["overall"].mean().rename("A_per_replicate_scale")
    b = ordering_b(df, datasets, cohort, batch_metrics, bio_metrics).groupby("method")["overall"].mean().rename("B_seedmean_then_scale")
    c = ordering_c(df, datasets, cohort, batch_metrics, bio_metrics).groupby("method")["overall"].mean().rename("C_dataset_level_scale")
    combined = pd.concat([a, b, c], axis=1).sort_values("A_per_replicate_scale", ascending=False)
    for col in combined.columns:
        combined[f"rank_{col.split('_')[0]}"] = combined[col].rank(ascending=False)
    return combined


def report(df: pd.DataFrame, label: str, with_kbet: bool):
    batch_metrics = list(BATCH_METRICS) if with_kbet else [m for m in BATCH_METRICS if m != "kBET"]
    bio_metrics = list(BIOLOGY_METRICS)
    print(f"\n===== {label} (kBET {'included' if with_kbet else 'EXCLUDED'}) =====")
    combined = run_all_orderings(df, REAL_DATASETS, MAIN_COHORT, batch_metrics, bio_metrics)
    print(combined.round(3).to_string())

    if "solar_orthogonal" in combined.index and "frozen_reference_scanvi" in combined.index:
        for col in ("A_per_replicate_scale", "B_seedmean_then_scale", "C_dataset_level_scale"):
            delta = combined.loc["solar_orthogonal", col] - combined.loc["frozen_reference_scanvi", col]
            rank_col = f"rank_{col.split('_')[0]}"
            solar_rank = combined.loc["solar_orthogonal", rank_col]
            scanvi_rank = combined.loc["frozen_reference_scanvi", rank_col]
            print(f"[{col}] SOLAR - frozen_reference_scanvi = {delta:+.3f}; "
                  f"SOLAR rank={solar_rank:.0f}, scanvi rank={scanvi_rank:.0f}; "
                  f"reversed={solar_rank > scanvi_rank}")

    suffix = "_with_kbet" if with_kbet else "_without_kbet"
    combined["higher_is_better"] = True
    combined.to_csv(f"data/aggregation_sensitivity_3_real_datasets{suffix}.csv")
    return combined


def report_ablations(df: pd.DataFrame):
    """solar_none / solar_orthogonal_uniq reported against the SAME scaling
    pool used for the main cohort composite -- i.e. scored using the min/max
    already established by MAIN_COHORT, not re-scaled with the ablations
    folded in (which would change everyone else's score)."""
    print("\n===== SOLAR-family ablations (scored against the main-cohort scaling pool) =====")
    d = df[df["dataset_id"].isin(REAL_DATASETS) & df["method"].isin(MAIN_COHORT + ABLATIONS)]
    metric_cols = [c for c in list(BATCH_METRICS) + list(BIOLOGY_METRICS) if c in d.columns]
    dataset_mean = d.groupby(["dataset_id", "method"])[metric_cols].mean().reset_index()
    rows = []
    for dataset_id, group in dataset_mean.groupby("dataset_id"):
        main_pool = group[group["method"].isin(MAIN_COHORT)]
        scaled = group.copy()
        for m in metric_cols:
            lo, hi = main_pool[m].min(), main_pool[m].max()
            if pd.isna(lo) or np.isclose(lo, hi):
                scaled[m] = 0.5
            else:
                scaled[m] = ((group[m] - lo) / (hi - lo)).clip(0, 1)
        scaled = composite(scaled, list(BATCH_METRICS), list(BIOLOGY_METRICS))
        rows.append(scaled[["dataset_id", "method", "overall"]])
    out = pd.concat(rows, ignore_index=True)
    summary = out.groupby("method")["overall"].mean().sort_values(ascending=False)
    print(summary.round(3).to_string())
    out.to_csv("data/aggregation_ablations_3_real_datasets.csv", index=False)


def report_simulation_sanity_check(df: pd.DataFrame):
    print("\n===== Simulation datasets: SANITY CHECK ONLY, not part of the ranking =====")
    print("frozen_reference_scvi/scanvi on simulations = original single-seed (seed 0) legacy run,")
    print("named scvi_reference_mapping/scanvi_reference_mapping -- not reseeded (out of scope).")
    raw = pd.read_csv("data/raw_scib_metrics_all_runs.csv")
    metric_cols = {c: c.replace("metric__", "") for c in raw.columns if c.startswith("metric__")}
    raw = raw.rename(columns=metric_cols)
    cols = list(BATCH_METRICS) + list(BIOLOGY_METRICS)
    sims = raw[raw["dataset_id"].isin(SIM_DATASETS)]
    for method in ["solar_orthogonal", "scvi_reference_mapping", "scanvi_reference_mapping"]:
        sub = sims[sims["method"] == method]
        n = len(sub)
        print(f"-- {method} (n={n}) --")
        print(sub[cols].mean(numeric_only=True).round(3).to_string())


def main() -> int:
    METRIC_METADATA.to_csv("data/metric_direction_metadata.csv", index=False)
    df = load_heldout_data()
    print(f"loaded {len(df)} heldout-track rows for methods: {sorted(df['method'].unique())}")
    print(f"\nKBET DEFINITION (verified against source, not assumed):\n{KBET_DEFINITION}")

    report(df, "MAIN COMPARISON: 3 real datasets, held-out-batch track", with_kbet=True)
    report(df, "MAIN COMPARISON: 3 real datasets, held-out-batch track", with_kbet=False)
    report_ablations(df)
    report_simulation_sanity_check(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
