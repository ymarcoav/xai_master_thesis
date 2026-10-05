"""Combined SHAP-vs-LIME local agreement figure for the thesis.

A 2 x 2 grid of heatmaps - columns UNSW-NB15 / CIC-IDS2017, rows feature / rank agreement -
with models on the y-axis and k on the x-axis. Reads every value from local_agreement.csv
(scripts/explanation_report.py's thesis table) and computes nothing itself.

Styled to sit next to the confusion matrices: "Blues" colormap (on a linear, fixed [0, 1] scale
here, not log), matplotlib's default font, 10 pt cell annotations. Sized for the full text width
of an A4 thesis page (15.5 cm), with roughly square cells. No figure title: the caption is set
in LaTeX.

  python scripts/thesis_agreement_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

DATASETS = {"unsw_nb15": "UNSW-NB15", "cic_ids2017": "CIC-IDS2017"}
METRICS = {"fa_mean": "Feature agreement", "ra_mean": "Rank agreement"}
MODELS = {"random_forest": "Random forest", "xgboost": "XGBoost", "neural_network": "MLP", "svm": "Linear SVM"}
K_VALUES = (1, 5, 10)
CM_TO_INCH = 1 / 2.54


def agreement_matrix(table: pd.DataFrame, dataset: str, column: str) -> np.ndarray:
    """[model, k] matrix of one metric for one dataset, in MODELS x K_VALUES order."""
    rows = table[table["dataset"] == dataset].set_index(["model", "k"])[column]
    return np.array([[rows.loc[(model, k)] for k in K_VALUES] for model in MODELS])


# Layout in inches, placed by hand so the row labels and the shared "k" never get clipped and
# every cell is exactly square: the panel width fixes the cell size, and the figure height follows.
WIDTH = 15.5 * CM_TO_INCH
LEFT = 1.40  # row label + model tick labels
RIGHT = 0.85  # colorbar, its ticks and its label
COL_GAP = 0.15
ROW_GAP = 0.15
TOP = 0.35  # column headers
BOTTOM = 0.60  # k tick labels + shared "k"


def plot(table: pd.DataFrame, output_dir: Path) -> list[Path]:
    cmap = plt.get_cmap("Blues")
    norm = Normalize(vmin=0, vmax=1)
    cell = (WIDTH - LEFT - RIGHT - COL_GAP) / (2 * len(K_VALUES))
    panel_w, panel_h = cell * len(K_VALUES), cell * len(MODELS)
    height = TOP + 2 * panel_h + ROW_GAP + BOTTOM
    fig = plt.figure(figsize=(WIDTH, height))

    def box(x: float, y: float, w: float, h: float) -> list[float]:
        return [x / WIDTH, y / height, w / WIDTH, h / height]

    axes = np.empty((2, 2), dtype=object)
    for row in range(2):
        for col in range(2):
            y = BOTTOM + (1 - row) * (panel_h + ROW_GAP)
            axes[row, col] = fig.add_axes(box(LEFT + col * (panel_w + COL_GAP), y, panel_w, panel_h))

    for row, (column, metric_label) in enumerate(METRICS.items()):
        for col, (dataset, dataset_label) in enumerate(DATASETS.items()):
            ax = axes[row, col]
            values = agreement_matrix(table, dataset, column)
            image = ax.imshow(values, cmap=cmap, norm=norm, aspect="equal")
            for (i, j), value in np.ndenumerate(values):
                # White on dark cells, black on light ones, by the cell colour's luminance.
                r, g, b, _ = cmap(norm(value))
                text_color = "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=10)
            # k tick labels on the bottom row only, model labels on the left column only.
            ax.set_xticks(range(len(K_VALUES)), [str(k) for k in K_VALUES] if row == 1 else [])
            ax.set_yticks(range(len(MODELS)), list(MODELS.values()) if col == 0 else [])
            ax.tick_params(length=0, labelsize=10)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(dataset_label, fontsize=11)
        # Row label, centred on its row, left of the model labels.
        fig.text(
            0.12 / WIDTH, (BOTTOM + (1 - row) * (panel_h + ROW_GAP) + panel_h / 2) / height, metric_label,
            rotation=90, ha="left", va="center", fontsize=11,
        )
    # One shared "k", centred under both bottom panels.
    fig.text((LEFT + panel_w + COL_GAP / 2) / WIDTH, 0.20 / height, "k", ha="center", va="bottom", fontsize=10)

    cbar_ax = fig.add_axes(box(WIDTH - RIGHT + 0.15, BOTTOM, 0.14, 2 * panel_h + ROW_GAP))
    fig.colorbar(image, cax=cbar_ax, label="Agreement")
    cbar_ax.tick_params(labelsize=10)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / "local_agreement_heatmaps.pdf", output_dir / "local_agreement_heatmaps.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="outputs/metrics/explanation_report/local_agreement.csv")
    parser.add_argument("--output-dir", default="outputs/plots/thesis_figures")
    args = parser.parse_args()
    for path in plot(pd.read_csv(args.csv), Path(args.output_dir)):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
