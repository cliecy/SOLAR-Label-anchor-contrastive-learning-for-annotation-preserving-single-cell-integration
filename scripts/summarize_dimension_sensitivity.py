#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_scib_metrics_all_runs.csv"
OUT = ROOT / "data"
DIMENSION_BY_RUN = {"dim30": 30, "dim64": 64, "primary_128d": 128}
METRICS = {
    "metric__ASW_label": "Label ASW",
    "metric__NMI_cluster/label": "NMI",
    "metric__ARI_cluster/label": "ARI",
    "metric__graph_conn": "Graph connectivity",
    "metric__iLISI": "iLISI",
    "metric__PCR_batch": "PCR",
    "metric__trajectory": "Trajectory",
}


def main() -> int:
    raw = pd.read_csv(RAW)
    selected = raw.loc[
        raw["run"].isin(DIMENSION_BY_RUN)
        & (raw["method"] == "solar_orthogonal")
        & (raw["track"] == "core")
        & (raw["preprocessing_profile"] == "hvg2000_pca40")
    ].copy()
    selected["expected_dimension_from_run"] = selected["run"].map(DIMENSION_BY_RUN)
    if selected["dimension"].isna().any():
        raise ValueError("Dimension sweep rows lack manifest-derived actual dimensions")
    if not (
        selected["dimension"].astype(int) == selected["expected_dimension_from_run"]
    ).all():
        bad = selected.loc[
            selected["dimension"].astype(int)
            != selected["expected_dimension_from_run"],
            ["run", "job_id", "dimension", "expected_dimension_from_run"],
        ]
        raise ValueError(f"Dimension/run mismatch: {bad.to_dict('records')}")
    expected_keys = {
        (dataset, seed, dimension)
        for dataset in sorted(selected["dataset_id"].unique())
        for seed in (40, 41, 42, 43, 44)
        for dimension in (30, 64, 128)
    }
    observed_keys = set(
        selected[["dataset_id", "split_seed", "dimension"]]
        .astype({"split_seed": int, "dimension": int})
        .itertuples(index=False, name=None)
    )
    if observed_keys != expected_keys:
        raise ValueError(
            f"Incomplete dimension sweep: missing={sorted(expected_keys - observed_keys)}, "
            f"extra={sorted(observed_keys - expected_keys)}"
        )

    long = selected.melt(
        id_vars=["run", "dataset_id", "split_seed", "dimension"],
        value_vars=list(METRICS),
        var_name="metric_key",
        value_name="value",
    )
    long["metric"] = long["metric_key"].map(METRICS)
    long["dimension"] = long["dimension"].astype(int)
    summary = (
        long.groupby(["metric", "dimension"], sort=True)["value"]
        .agg(n="count", mean="mean", sd="std", median="median", min="min", max="max")
        .reset_index()
    )
    summary.to_csv(OUT / "dimension_sensitivity_corrected_summary.csv", index=False)

    paired_rows = []
    contrasts = ((30, 128), (64, 128), (30, 64))
    for metric_key, metric in METRICS.items():
        wide = selected.pivot(
            index=["dataset_id", "split_seed"],
            columns="dimension",
            values=metric_key,
        )
        for left, right in contrasts:
            differences = (wide[left] - wide[right]).dropna()
            paired_rows.append(
                {
                    "metric": metric,
                    "contrast": f"{left}D - {right}D",
                    "paired_n": int(len(differences)),
                    "mean_difference": float(differences.mean()),
                    "sd_difference": float(differences.std()),
                    "median_difference": float(differences.median()),
                    "mean_absolute_difference": float(differences.abs().mean()),
                    "max_absolute_difference": float(differences.abs().max()),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT / "dimension_sensitivity_corrected_paired.csv", index=False)

    exact_equal = {}
    for metric_key, metric in METRICS.items():
        wide = selected.pivot(
            index=["dataset_id", "split_seed"],
            columns="dimension",
            values=metric_key,
        ).dropna()
        exact_equal[metric] = {
            "complete_triplets": int(len(wide)),
            "exactly_equal_across_all_dimensions": int(
                np.logical_and(wide[30] == wide[64], wide[64] == wide[128]).sum()
            ),
        }

    audit = {
        "schema_version": 1,
        "source": "data/raw_scib_metrics_all_runs.csv",
        "source_runs": {
            "30": "solar_scib_rebuttal_dim30_v2",
            "64": "solar_scib_rebuttal_dim64_v2",
            "128": "solar_scib_rebuttal_v1",
        },
        "rows": int(len(selected)),
        "datasets": sorted(selected["dataset_id"].unique()),
        "seeds": sorted(int(value) for value in selected["split_seed"].unique()),
        "dimensions": sorted(int(value) for value in selected["dimension"].unique()),
        "unique_dataset_seed_dimension_keys": int(len(observed_keys)),
        "dimension_values_match_run_labels": True,
        "exact_equality_check": exact_equal,
        "summary_csv": "dimension_sensitivity_corrected_summary.csv",
        "paired_csv": "dimension_sensitivity_corrected_paired.csv",
    }
    (OUT / "dimension_sensitivity_corrected_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print()
    print(paired.to_string(index=False))
    print()
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
