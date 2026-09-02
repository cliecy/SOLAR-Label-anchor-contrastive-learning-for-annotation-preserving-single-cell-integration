#!/usr/bin/env python3
"""Regenerate the main-text figure showing query-label recovery (macro-F1 and
balanced accuracy) across the 9 real held-out batches, from
data/classifier_unified_summary.csv (unchanged machine-readable data from the
canonical result set; see PROVENANCE.md). This is a self-contained,
simplified re-implementation of the manuscript figure script for
reproducibility-package purposes -- colors/markers/layout match the
manuscript figure but this file does not depend on any other script in this
package besides this data table.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "data" / "classifier_unified_summary.csv"
OUT_DIR = HERE.parent / "data"

REAL_DATASETS = ["immune_cell_human", "lung_atlas", "pancreas"]
METHOD_ORDER = ["unintegrated_pca_matched", "solar_orthogonal", "frozen_reference_scvi", "frozen_reference_scanvi"]

# Okabe-Ito colorblind-safe palette, matching the manuscript figures.
COLOR = {
    "unintegrated_pca_matched": "#999999",
    "solar_orthogonal": "#0072B2",
    "frozen_reference_scvi": "#E69F00",
    "frozen_reference_scanvi": "#D55E00",
}
MARKER = {
    "unintegrated_pca_matched": "D",
    "solar_orthogonal": "o",
    "frozen_reference_scvi": "s",
    "frozen_reference_scanvi": "^",
}
LABEL = {
    "unintegrated_pca_matched": "Matched PCA (unintegrated)",
    "solar_orthogonal": "SOLAR (repeated anchor, primary)",
    "frozen_reference_scvi": "Frozen-reference scVI",
    "frozen_reference_scanvi": "Frozen-reference scANVI",
}
DATASET_LABEL = {"immune_cell_human": "Immune", "lung_atlas": "Lung", "pancreas": "Pancreas"}
BATCH_DISPLAY = {
    "10X": "10X", "Oetjen_A": "Oetjen A", "Villani": "Villani",
    "3": "3", "4": "4", "B1": "B1",
    "fluidigmc1": "Fluidigm C1", "inDrop2": "inDrop 2", "inDrop3": "inDrop 3",
}


def panel(ax, data, batch_order, x_pos, metric_col, ylabel, letter):
    n_methods = len(METHOD_ORDER)
    jitter = np.linspace(-0.26, 0.26, n_methods)
    for mi, method in enumerate(METHOD_ORDER):
        sub = data[data.method == method]
        xs_mean, ys_mean, yerr = [], [], []
        for key in batch_order:
            ds, batch = key
            row = sub[(sub.dataset_id == ds) & (sub.heldout_batch == batch)]
            if row.empty:
                continue
            x0 = x_pos[key] + jitter[mi]
            vals = row[metric_col].to_numpy(dtype=float)
            if method == "unintegrated_pca_matched":
                ax.scatter([x0], vals, marker=MARKER[method], color=COLOR[method],
                           s=22, zorder=4, linewidths=0.4, edgecolors="black")
            else:
                order = np.argsort(row["seed"].to_numpy())
                n = len(vals)
                offsets = np.linspace(-0.05, 0.05, n) if n > 1 else np.zeros(1)
                seed_x = np.empty(n)
                seed_x[order] = x0 + offsets
                ax.scatter(seed_x, vals, marker=MARKER[method], color=COLOR[method],
                           s=7, alpha=0.45, zorder=3, linewidths=0)
                xs_mean.append(x0)
                ys_mean.append(vals.mean())
                yerr.append(vals.std(ddof=1))
        if xs_mean:
            ax.errorbar(xs_mean, ys_mean, yerr=yerr, fmt=MARKER[method], color=COLOR[method],
                         ms=4.2, mec="black", mew=0.4, elinewidth=0.6, capsize=1.5,
                         zorder=5, linestyle="none")

    for key in batch_order:
        ax.axvline(x_pos[key] - 0.5, color="0.88", lw=0.4, zorder=0)
    ax.axvline(len(batch_order) - 0.5, color="0.88", lw=0.4, zorder=0)
    ax.set_xlim(-0.5, len(batch_order) - 0.5)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.text(-0.115, 1.05, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")

    prev_ds = None
    for key in batch_order:
        ds, _ = key
        if ds != prev_ds:
            ax.axvline(x_pos[key] - 0.5, color="0.6", lw=0.8, zorder=1)
            prev_ds = ds


def main() -> int:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"missing {DATA_PATH} -- run from the package root")
    data = pd.read_csv(DATA_PATH)
    data = data[(data.track == "heldout") & data.dataset_id.isin(REAL_DATASETS) & data.method.isin(METHOD_ORDER)]

    batch_order = []
    for ds in REAL_DATASETS:
        batches = sorted(data[data.dataset_id == ds]["heldout_batch"].unique())
        batch_order.extend([(ds, b) for b in batches])
    x_pos = {key: i for i, key in enumerate(batch_order)}

    fig, axes = plt.subplots(2, 1, figsize=(170 / 25.4, 95 / 25.4), sharex=True)
    panel(axes[0], data, batch_order, x_pos, "macro_f1", "Macro-F1 (query)", "A")
    panel(axes[1], data, batch_order, x_pos, "balanced_accuracy", "Balanced accuracy (query)", "B")

    axes[1].set_xticks(range(len(batch_order)))
    axes[1].set_xticklabels([BATCH_DISPLAY.get(b, b) for _, b in batch_order], rotation=45, ha="right")

    legend_handles = [
        plt.Line2D([0], [0], marker=MARKER[m], color="none", markerfacecolor=COLOR[m],
                   markeredgecolor="black", markeredgewidth=0.3, label=LABEL[m], ms=5.5)
        for m in METHOD_ORDER
    ]
    axes[0].legend(handles=legend_handles, loc="lower right", fontsize=5.3, frameon=True,
                   framealpha=0.9, edgecolor="0.8", handletextpad=0.5, borderpad=0.5, labelspacing=0.4)

    for ds in REAL_DATASETS:
        idxs = [i for i, (d, _) in enumerate(batch_order) if d == ds]
        center = np.mean(idxs)
        axes[0].text(center, 1.1, DATASET_LABEL[ds], transform=axes[0].get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=8, fontweight="bold", clip_on=False)

    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.10, right=0.995, top=0.90, bottom=0.20, hspace=0.16)

    out_pdf = OUT_DIR / "Figure_query_label_recovery.pdf"
    out_png = OUT_DIR / "Figure_query_label_recovery.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    print(f"wrote {out_pdf}, {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
