#!/usr/bin/env python3
"""Build the audited v1.1 main and supplementary external-baseline tables."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from scib_contract import BATCH_METRICS, BIOLOGY_METRICS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CONFIG_PATH = ROOT / "configs/baseline_extension_v1_1.yaml"
BASE_SCIB_PATH = DATA / "raw_scib_metrics_all_runs.csv"
BASE_QUERY_PATH = DATA / "classifier_unified_summary.csv"
EXTENSION_SCIB_PATH = DATA / "baseline_extension_scib_metrics.csv"
EXTENSION_QUERY_PATH = DATA / "baseline_extension_query_metrics.csv"
ARTIFACT_AUDIT_PATH = DATA / "baseline_extension_artifact_audit.json"
SCIB_AUDIT_PATH = DATA / "baseline_extension_scib_audit.json"

REAL_DATASETS = ("pancreas", "immune_cell_human", "lung_atlas")
HELDOUT_BATCHES = {
    "pancreas": ("fluidigmc1", "inDrop2", "inDrop3"),
    "immune_cell_human": ("10X", "Oetjen_A", "Villani"),
    "lung_atlas": ("3", "4", "B1"),
}
SEEDS = (40, 41, 42, 43, 44)
MAIN_COHORT = (
    "solar_orthogonal",
    "unintegrated_pca_matched",
    "frozen_reference_scvi",
    "frozen_reference_scanvi",
    "sclsc_refonly",
)
MAIN_STOCHASTIC = (
    "solar_orthogonal",
    "frozen_reference_scvi",
    "frozen_reference_scanvi",
    "sclsc_refonly",
)
MAIN_BROADCAST = ("unintegrated_pca_matched",)
SUPPLEMENTARY_EMBEDDING_COHORT = MAIN_COHORT + ("symphony_matched_pca40",)
SUPPLEMENTARY_STOCHASTIC = MAIN_STOCHASTIC
SUPPLEMENTARY_BROADCAST = MAIN_BROADCAST + ("symphony_matched_pca40",)
EXPECTED_SCIB_COUNTS = {
    "solar_orthogonal": 45,
    "unintegrated_pca_matched": 9,
    "frozen_reference_scvi": 45,
    "frozen_reference_scanvi": 45,
    "sclsc_refonly": 45,
    "symphony_matched_pca40": 9,
}
EXPECTED_QUERY_COUNTS = {
    ("solar_orthogonal", "common_distance_weighted_15nn"): 45,
    ("unintegrated_pca_matched", "common_distance_weighted_15nn"): 9,
    ("frozen_reference_scvi", "common_distance_weighted_15nn"): 45,
    ("frozen_reference_scanvi", "common_distance_weighted_15nn"): 45,
    ("sclsc_refonly", "common_distance_weighted_15nn"): 45,
    ("sclsc_refonly", "official_sclsc_unweighted_10nn_training_subset"): 45,
    ("symphony_matched_pca40", "common_distance_weighted_15nn"): 9,
    ("scmap_cell", "official_scmap_cell"): 9,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def expected_pairs() -> set[tuple[str, str]]:
    return {
        (dataset, batch)
        for dataset, batches in HELDOUT_BATCHES.items()
        for batch in batches
    }


def audit_sclsc_multiseed(artifact_audit: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in artifact_audit.get("runs", [])
        if row.get("method") == "sclsc_refonly" and row.get("passed") is True
    ]
    if len(rows) != 45:
        raise ValueError(f"SCLSC artifact audit has {len(rows)} passed runs, expected 45")
    groups: dict[str, Any] = {}
    for dataset, batches in HELDOUT_BATCHES.items():
        for batch in batches:
            selected = [
                row
                for row in rows
                if str(row.get("dataset_id")) == dataset
                and str(row.get("heldout_batch")) == batch
            ]
            seeds = sorted(int(row["model_seed"]) for row in selected)
            hashes = [str(row["artifact_sha256"]) for row in selected]
            if seeds != list(SEEDS):
                raise ValueError(
                    f"SCLSC seed coverage mismatch for {dataset}/{batch}: {seeds}"
                )
            if len(set(hashes)) != len(SEEDS):
                raise ValueError(
                    f"SCLSC embeddings are not unique across seeds for {dataset}/{batch}"
                )
            groups[f"{dataset}::{batch}"] = {
                "seeds": seeds,
                "unique_embedding_sha256": len(set(hashes)),
            }
    return groups



def normalize_batch(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").astype(str)


def metric_columns(frame: pd.DataFrame) -> list[str]:
    return [name for name in BATCH_METRICS + BIOLOGY_METRICS if name in frame]


def load_scib(config: dict[str, Any]) -> pd.DataFrame:
    base = pd.read_csv(BASE_SCIB_PATH, dtype={"heldout_batch": str})
    expected_base_hash = config["source_evidence"]["local_files_sha256"][
        "raw_scib_metrics_all_runs_complete.csv"
    ]
    if sha256_file(BASE_SCIB_PATH) != expected_base_hash:
        raise ValueError("base scIB table differs from the local-final-paper evidence hash")
    base = base.rename(
        columns={name: name.removeprefix("metric__") for name in base if name.startswith("metric__")}
    )
    solar = base[
        (base["run"] == "primary_128d")
        & (base["track"] == "heldout")
        & (base["method"] == "solar_orthogonal")
    ].copy()
    pca = base[
        (base["method"] == "unintegrated_pca_matched")
        & base["heldout_batch"].notna()
    ].copy()
    frozen = base[
        base["method"].isin(("frozen_reference_scvi", "frozen_reference_scanvi"))
    ].copy()

    extension = pd.read_csv(EXTENSION_SCIB_PATH, dtype={"heldout_batch": str})
    extension = extension.rename(
        columns={name: name.removeprefix("metric__") for name in extension if name.startswith("metric__")}
    )
    extension = extension[
        extension["method"].isin(("sclsc_refonly", "symphony_matched_pca40"))
    ].copy()
    combined = pd.concat([solar, pca, frozen, extension], ignore_index=True, sort=False)
    combined = combined[combined["dataset_id"].isin(REAL_DATASETS)].copy()
    combined["heldout_batch"] = normalize_batch(combined["heldout_batch"])
    combined["model_seed"] = pd.to_numeric(combined["model_seed"], errors="coerce")

    allowed = expected_pairs()
    combined = combined[
        combined.apply(
            lambda row: (str(row["dataset_id"]), str(row["heldout_batch"])) in allowed,
            axis=1,
        )
    ].copy()
    counts = combined.groupby("method").size().astype(int).to_dict()
    if counts != EXPECTED_SCIB_COUNTS:
        raise ValueError(f"scIB method coverage mismatch: {counts}")
    duplicate_key = ["dataset_id", "heldout_batch", "method", "model_seed"]
    if combined.duplicated(duplicate_key).any():
        duplicates = combined.loc[combined.duplicated(duplicate_key, keep=False), duplicate_key]
        raise ValueError(f"duplicate scIB replicate keys:\n{duplicates.to_string(index=False)}")
    for column in metric_columns(combined):
        values = pd.to_numeric(combined[column], errors="coerce")
        finite_or_missing = values.isna() | np.isfinite(values)
        if not finite_or_missing.all():
            raise ValueError(f"scIB metric {column} contains non-finite values")
        combined[column] = values
    return combined


def load_query(config: dict[str, Any]) -> pd.DataFrame:
    expected_classifier_hash = config["source_evidence"]["local_files_sha256"][
        "classifier_unified_summary.csv"
    ]
    if sha256_file(BASE_QUERY_PATH) != expected_classifier_hash:
        raise ValueError("base classifier table differs from the local-final-paper evidence hash")
    base = pd.read_csv(BASE_QUERY_PATH, dtype={"heldout_batch": str})
    base = base[
        (base["track"] == "heldout")
        & base["dataset_id"].isin(REAL_DATASETS)
        & base["method"].isin(MAIN_COHORT[:-1])
    ].copy()
    base["heldout_batch"] = normalize_batch(base["heldout_batch"])
    base = base[
        base.apply(
            lambda row: (str(row["dataset_id"]), str(row["heldout_batch"])) in expected_pairs(),
            axis=1,
        )
    ].copy()
    base["readout"] = "common_distance_weighted_15nn"
    base["model_seed"] = pd.to_numeric(base["model_seed"], errors="coerce")
    base["coverage"] = 1.0
    base["assigned_only_macro_f1"] = np.nan
    base["assigned_only_accuracy"] = np.nan

    extension = pd.read_csv(EXTENSION_QUERY_PATH, dtype={"heldout_batch": str})
    extension["heldout_batch"] = normalize_batch(extension["heldout_batch"])
    combined = pd.concat([base, extension], ignore_index=True, sort=False)
    keys = combined.groupby(["method", "readout"], dropna=False).size().astype(int).to_dict()
    if keys != EXPECTED_QUERY_COUNTS:
        raise ValueError(f"query metric coverage mismatch: {keys}")
    duplicate_key = ["dataset_id", "heldout_batch", "method", "readout", "model_seed"]
    if combined.duplicated(duplicate_key).any():
        duplicates = combined.loc[combined.duplicated(duplicate_key, keep=False), duplicate_key]
        raise ValueError(f"duplicate query replicate keys:\n{duplicates.to_string(index=False)}")
    for column in ("macro_f1", "balanced_accuracy", "coverage"):
        values = pd.to_numeric(combined[column], errors="coerce")
        if not np.isfinite(values).all():
            raise ValueError(f"query metric {column} contains non-finite values")
        combined[column] = values
    return combined


def rescale(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.dropna()
    if finite.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)
    lower = float(finite.min())
    upper = float(finite.max())
    if np.isclose(lower, upper, rtol=0.0, atol=0.0):
        result = pd.Series(np.nan, index=values.index, dtype=float)
        result.loc[numeric.notna()] = 0.5
        return result
    return (numeric - lower) / (upper - lower)


def rescale_complete_cohort(values: pd.Series, source_values: pd.Series) -> pd.Series:
    """Scale only metrics observed for every row in the comparison group."""
    source = pd.to_numeric(source_values, errors="coerce")
    if source.isna().any():
        return pd.Series(np.nan, index=values.index, dtype=float)
    return rescale(values)


def add_composite(
    frame: pd.DataFrame,
    batch_metrics: tuple[str, ...],
    biology_metrics: tuple[str, ...],
) -> pd.DataFrame:
    result = frame.copy()
    present_batch = [name for name in batch_metrics if name in result]
    present_biology = [name for name in biology_metrics if name in result]
    result["batch_correction"] = result[present_batch].mean(axis=1, skipna=True)
    result["bio_conservation"] = result[present_biology].mean(axis=1, skipna=True)
    result["overall"] = 0.4 * result["batch_correction"] + 0.6 * result["bio_conservation"]
    return result


def replicate_grid(
    raw: pd.DataFrame,
    cohort: tuple[str, ...],
    stochastic: tuple[str, ...],
    broadcast: tuple[str, ...],
) -> pd.DataFrame:
    stochastic_rows = raw[raw["method"].isin(stochastic)].copy()
    fixed_rows = raw[raw["method"].isin(broadcast)].copy()
    broadcast_rows: list[pd.Series] = []
    for _, source in fixed_rows.iterrows():
        for seed in SEEDS:
            row = source.copy()
            row["model_seed"] = seed
            broadcast_rows.append(row)
    grid = pd.concat([stochastic_rows, pd.DataFrame(broadcast_rows)], ignore_index=True, sort=False)
    grid = grid[grid["method"].isin(cohort)].copy()
    expected_rows = 9 * len(SEEDS) * len(cohort)
    if len(grid) != expected_rows:
        raise ValueError(f"replicate grid has {len(grid)} rows, expected {expected_rows}")
    group_counts = grid.groupby(["dataset_id", "heldout_batch", "model_seed"])["method"].nunique()
    if not (group_counts == len(cohort)).all():
        raise ValueError("replicate grid lacks a method in one or more batch-by-seed groups")
    return grid


def ordering_a(
    raw: pd.DataFrame,
    cohort: tuple[str, ...],
    stochastic: tuple[str, ...],
    broadcast: tuple[str, ...],
    batch_metrics: tuple[str, ...],
) -> pd.Series:
    grid = replicate_grid(raw, cohort, stochastic, broadcast)
    metrics = [name for name in batch_metrics + BIOLOGY_METRICS if name in grid]
    rows: list[pd.DataFrame] = []
    for _, group in grid.groupby(["dataset_id", "heldout_batch", "model_seed"], sort=False):
        scaled = group.copy()
        for metric in metrics:
            scaled[metric] = rescale_complete_cohort(scaled[metric], group[metric])
        rows.append(add_composite(scaled, batch_metrics, BIOLOGY_METRICS))
    scores = pd.concat(rows, ignore_index=True)
    return scores.groupby("method")["overall"].mean().rename("A_per_replicate_scale")


def ordering_b(
    raw: pd.DataFrame,
    cohort: tuple[str, ...],
    batch_metrics: tuple[str, ...],
) -> pd.Series:
    selected = raw[raw["method"].isin(cohort)].copy()
    metrics = [name for name in batch_metrics + BIOLOGY_METRICS if name in selected]
    seed_mean = (
        selected.groupby(["dataset_id", "heldout_batch", "method"], as_index=False)[metrics]
        .mean()
    )
    rows: list[pd.DataFrame] = []
    for (dataset, batch), group in seed_mean.groupby(
        ["dataset_id", "heldout_batch"], sort=False
    ):
        if set(group["method"]) != set(cohort):
            raise ValueError("seed-mean cohort is incomplete for a held-out batch")
        source_group = selected[
            (selected["dataset_id"] == dataset) & (selected["heldout_batch"] == batch)
        ]
        scaled = group.copy()
        for metric in metrics:
            scaled[metric] = rescale_complete_cohort(
                scaled[metric], source_group[metric]
            )
        rows.append(add_composite(scaled, batch_metrics, BIOLOGY_METRICS))
    scores = pd.concat(rows, ignore_index=True)
    return scores.groupby("method")["overall"].mean().rename("B_seedmean_then_scale")


def ordering_c(
    raw: pd.DataFrame,
    cohort: tuple[str, ...],
    batch_metrics: tuple[str, ...],
) -> pd.Series:
    selected = raw[raw["method"].isin(cohort)].copy()
    metrics = [name for name in batch_metrics + BIOLOGY_METRICS if name in selected]
    dataset_mean = selected.groupby(["dataset_id", "method"], as_index=False)[metrics].mean()
    rows: list[pd.DataFrame] = []
    for dataset, group in dataset_mean.groupby("dataset_id", sort=False):
        if set(group["method"]) != set(cohort):
            raise ValueError("dataset-level cohort is incomplete")
        source_group = selected[selected["dataset_id"] == dataset]
        scaled = group.copy()
        for metric in metrics:
            scaled[metric] = rescale_complete_cohort(
                scaled[metric], source_group[metric]
            )
        rows.append(add_composite(scaled, batch_metrics, BIOLOGY_METRICS))
    scores = pd.concat(rows, ignore_index=True)
    return scores.groupby("method")["overall"].mean().rename("C_dataset_level_scale")


def aggregate_cohort(
    raw: pd.DataFrame,
    cohort: tuple[str, ...],
    stochastic: tuple[str, ...],
    broadcast: tuple[str, ...],
    *,
    include_kbet: bool,
) -> pd.DataFrame:
    batch_metrics = tuple(
        name for name in BATCH_METRICS if include_kbet or name != "kBET"
    )
    result = pd.concat(
        [
            ordering_a(raw, cohort, stochastic, broadcast, batch_metrics),
            ordering_b(raw, cohort, batch_metrics),
            ordering_c(raw, cohort, batch_metrics),
        ],
        axis=1,
    )
    result = result.reindex(cohort)
    for name in result.columns:
        result[f"rank_{name.split('_')[0]}"] = result[name].rank(ascending=False)
    result["higher_is_better"] = True
    result["kbet_included"] = include_kbet
    return result.reset_index(names="method")


def query_summary(query: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "readout"]
    row_summary = (
        query.groupby(keys, as_index=False)
        .agg(
            query_runs=("macro_f1", "size"),
            model_seeds=("model_seed", lambda values: max(int(values.nunique()), 1)),
            mean_macro_f1=("macro_f1", "mean"),
            standard_deviation_across_batch_seed_rows_macro_f1=("macro_f1", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            standard_deviation_across_batch_seed_rows_balanced_accuracy=(
                "balanced_accuracy",
                "std",
            ),
            mean_coverage=("coverage", "mean"),
            mean_assigned_only_macro_f1=("assigned_only_macro_f1", "mean"),
            mean_assigned_only_accuracy=("assigned_only_accuracy", "mean"),
        )
    )
    seed_means = (
        query.groupby(keys + ["model_seed"], as_index=False, dropna=False)
        .agg(
            seed_mean_macro_f1=("macro_f1", "mean"),
            seed_mean_balanced_accuracy=("balanced_accuracy", "mean"),
        )
    )
    seed_summary = (
        seed_means.groupby(keys, as_index=False)
        .agg(
            standard_deviation_across_seed_means_macro_f1=("seed_mean_macro_f1", "std"),
            standard_deviation_across_seed_means_balanced_accuracy=(
                "seed_mean_balanced_accuracy",
                "std",
            ),
        )
    )
    return row_summary.merge(seed_summary, on=keys, validate="one_to_one")


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    artifact_audit = read_json(ARTIFACT_AUDIT_PATH)
    scib_audit = read_json(SCIB_AUDIT_PATH)
    if artifact_audit.get("state") != "passed" or artifact_audit.get("passed_runs") != 63:
        raise ValueError("external-baseline artifact audit is not a complete 63/63 pass")
    if scib_audit.get("state") != "passed" or scib_audit.get("passed_jobs") != 54:
        raise ValueError("external-baseline official-scIB audit is not a complete 54/54 pass")
    config_hash = sha256_file(CONFIG_PATH)
    for name, audit in (("artifact", artifact_audit), ("scIB", scib_audit)):
        if audit.get("config_sha256") != config_hash:
            raise ValueError(f"{name} audit was generated under a different contract")
    if scib_audit.get("release_table_sha256") != sha256_file(EXTENSION_SCIB_PATH):
        raise ValueError("external-baseline scIB table differs from its strict audit")
    multiseed_audit = audit_sclsc_multiseed(artifact_audit)

    scib = load_scib(config)
    query = load_query(config)
    summary = query_summary(query)
    summary.to_csv(DATA / "baseline_extension_annotation_summary.csv", index=False)

    outputs: dict[str, pd.DataFrame] = {}
    for label, cohort, stochastic, broadcast in (
        ("main", MAIN_COHORT, MAIN_STOCHASTIC, MAIN_BROADCAST),
        (
            "supplementary_embedding",
            SUPPLEMENTARY_EMBEDDING_COHORT,
            SUPPLEMENTARY_STOCHASTIC,
            SUPPLEMENTARY_BROADCAST,
        ),
    ):
        for include_kbet in (True, False):
            suffix = "with_kbet" if include_kbet else "without_kbet"
            table = aggregate_cohort(
                scib,
                cohort,
                stochastic,
                broadcast,
                include_kbet=include_kbet,
            )
            name = f"{label}_scib_composite_v1_1_{suffix}.csv"
            table.to_csv(DATA / name, index=False)
            outputs[name] = table

    main_scib = outputs["main_scib_composite_v1_1_with_kbet.csv"]
    main_scib_no_kbet = outputs["main_scib_composite_v1_1_without_kbet.csv"].rename(
        columns={
            "A_per_replicate_scale": "A_without_kbet",
            "B_seedmean_then_scale": "B_without_kbet",
            "C_dataset_level_scale": "C_without_kbet",
        }
    )[["method", "A_without_kbet", "B_without_kbet", "C_without_kbet"]]
    common_query = summary[
        (summary["readout"] == "common_distance_weighted_15nn")
        & summary["method"].isin(MAIN_COHORT)
    ]
    if set(common_query["method"]) != set(MAIN_COHORT):
        raise ValueError(
            "main query summary method mismatch: "
            f"{sorted(set(common_query['method']))}"
        )
    main_table = (
        main_scib.merge(main_scib_no_kbet, on="method", validate="one_to_one")
        .merge(common_query, on="method", validate="one_to_one")
        .drop(columns=["readout", "kbet_included"])
    )
    main_table.to_csv(DATA / "main_table_v1_1.csv", index=False)

    supplementary_scib = outputs[
        "supplementary_embedding_scib_composite_v1_1_with_kbet.csv"
    ]
    symphony_scib = supplementary_scib[
        supplementary_scib["method"] == "symphony_matched_pca40"
    ].copy()
    symphony_query = summary[
        (summary["method"] == "symphony_matched_pca40")
        & (summary["readout"] == "common_distance_weighted_15nn")
    ].copy()
    symphony = symphony_scib.merge(symphony_query, on="method", validate="one_to_one")
    symphony["table_role"] = "supplementary_embedding_and_annotation"
    scmap = summary[
        (summary["method"] == "scmap_cell")
        & (summary["readout"] == "official_scmap_cell")
    ].copy()
    scmap["table_role"] = "supplementary_native_annotation_only"
    sclsc_native = summary[
        (summary["method"] == "sclsc_refonly")
        & (summary["readout"] == "official_sclsc_unweighted_10nn_training_subset")
    ].copy()
    sclsc_native["method"] = "sclsc_refonly_native10"
    sclsc_native["table_role"] = "supplementary_method_native_readout"
    supplementary = pd.concat([symphony, scmap, sclsc_native], ignore_index=True, sort=False)
    supplementary.to_csv(DATA / "supplementary_table_v1_1.csv", index=False)

    detailed_query = query.sort_values(
        ["method", "readout", "dataset_id", "heldout_batch", "model_seed"],
        kind="stable",
    )
    detailed_query.to_csv(DATA / "baseline_extension_query_metrics_audited.csv", index=False)
    table_paths = [
        DATA / "baseline_extension_annotation_summary.csv",
        DATA / "main_table_v1_1.csv",
        DATA / "supplementary_table_v1_1.csv",
        DATA / "baseline_extension_query_metrics_audited.csv",
        *(DATA / name for name in outputs),
    ]
    missing_scib_metrics: dict[str, dict[str, int]] = {}
    for method, method_rows in scib.groupby("method", sort=False):
        counts = {
            metric: int(method_rows[metric].isna().sum())
            for metric in metric_columns(method_rows)
            if method_rows[metric].isna().any()
        }
        if counts:
            missing_scib_metrics[str(method)] = counts
    audit = {
        "schema_version": 1,
        "state": "passed",
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": sha256_file(CONFIG_PATH),
        "source_files": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                BASE_SCIB_PATH,
                BASE_QUERY_PATH,
                EXTENSION_SCIB_PATH,
                EXTENSION_QUERY_PATH,
                ARTIFACT_AUDIT_PATH,
                SCIB_AUDIT_PATH,
            )
        },
        "cohorts": {
            "main": list(MAIN_COHORT),
            "supplementary_embedding": list(SUPPLEMENTARY_EMBEDDING_COHORT),
            "supplementary_single_seed_broadcast_for_ordering_A": [
                "symphony_matched_pca40"
            ],
        },
        "sclsc_multiseed_independence": multiseed_audit,
        "expected_scib_counts": EXPECTED_SCIB_COUNTS,
        "expected_query_counts": {
            f"{method}::{readout}": count
            for (method, readout), count in EXPECTED_QUERY_COUNTS.items()
        },
        "observed_scib_rows": len(scib),
        "observed_query_rows": len(query),
        "missing_scib_metric_counts_by_method": missing_scib_metrics,
        "aggregation": (
            "Dataset-wise method-cohort min-max scaling under orderings A/B/C; "
            "overall=0.4*batch_correction+0.6*bio_conservation. A metric missing "
            "for any row is excluded for every method in that ordering's matching "
            "normalization group; unavailable metrics never change method-specific "
            "composite denominators."
        ),
        "outputs": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in table_paths
        },
    }
    write_json(DATA / "baseline_extension_table_audit.json", audit)
    print(
        json.dumps(
            {
                "state": "passed",
                "main_methods": len(main_table),
                "supplementary_rows": len(supplementary),
                "scib_rows": len(scib),
                "query_rows": len(query),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
