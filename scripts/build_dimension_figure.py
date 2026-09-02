#!/usr/bin/env python3
"""Rebuild the corrected SOLAR 30D/64D/128D sensitivity panel."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_scib_metrics_all_runs.csv"
OUT = ROOT / "data"
DIMENSION_BY_RUN = {"dim30": 30, "dim64": 64, "primary_128d": 128}
DATASETS = [
    "immune_cell_human",
    "lung_atlas",
    "pancreas",
    "simulation_1",
    "simulation_2",
]
DATASET_LABELS = ["Immune", "Lung", "Pancreas", "Sim1", "Sim2"]
METRICS = [
    ("metric__ASW_label", "Label ASW"),
    ("metric__NMI_cluster/label", "NMI"),
    ("metric__ARI_cluster/label", "ARI"),
    ("metric__graph_conn", "Graph connectivity"),
    ("metric__iLISI", "iLISI"),
    ("metric__PCR_batch", "PCR"),
    ("metric__trajectory", "Trajectory"),
]
COLORS = {30: "#56B4E9", 64: "#0072B2", 128: "#000000"}
MARKERS = {30: "o", 64: "s", 128: "^"}


def load_dimension_rows() -> pd.DataFrame:
    raw = pd.read_csv(RAW)
    rows = raw.loc[
        raw["run"].isin(DIMENSION_BY_RUN)
        & (raw["method"] == "solar_orthogonal")
        & (raw["track"] == "core")
        & (raw["preprocessing_profile"] == "hvg2000_pca40")
    ].copy()
    expected = rows["run"].map(DIMENSION_BY_RUN)
    if rows["dimension"].isna().any():
        raise ValueError("Dimension-sweep rows lack manifest-derived dimensions")
    rows["dimension"] = rows["dimension"].astype(int)
    if not (rows["dimension"] == expected).all():
        bad = rows.loc[
            rows["dimension"] != expected,
            ["run", "job_id", "dimension"],
        ]
        raise ValueError(f"Run-label/dimension mismatch: {bad.to_dict('records')}")
    keys = rows[["dataset_id", "split_seed", "dimension"]]
    if len(rows) != 75 or keys.duplicated().any():
        raise ValueError(
            "Expected 75 unique dataset/seed/dimension rows; "
            f"found {len(rows)} rows and {int(keys.duplicated().sum())} duplicates"
        )
    return rows


def draw_metric(ax: plt.Axes, rows: pd.DataFrame, metric: str, title: str) -> None:
    for index, dimension in enumerate((30, 64, 128)):
        selected = rows.loc[rows["dimension"] == dimension]
        xs: list[float] = []
        means: list[float] = []
        errors: list[float] = []
        for dataset_index, dataset in enumerate(DATASETS):
            values = selected.loc[selected["dataset_id"] == dataset, metric].dropna()
            if values.empty:
                continue
            xs.append(dataset_index + (index - 1) * 0.22)
            means.append(float(values.mean()))
            errors.append(float(values.std(ddof=1)))
        ax.errorbar(
            xs,
            means,
            yerr=errors,
            fmt=MARKERS[dimension],
            color=COLORS[dimension],
            markersize=4,
            elinewidth=0.8,
            capsize=1.8,
            markeredgecolor="none",
            linestyle="none",
        )
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASET_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> int:
    rows = load_dimension_rows()
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.5), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, METRICS):
        draw_metric(ax, rows, metric, title)
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=MARKERS[dimension],
            color=COLORS[dimension],
            linestyle="none",
            label=f"{dimension}D",
            markersize=7,
        )
        for dimension in (30, 64, 128)
    ]
    legend_ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        title="Output dimension",
    )
    pdf = OUT / "Figure_dimension_sensitivity.pdf"
    png = OUT / "Figure_dimension_sensitivity.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
