from pathlib import Path

import matplotlib

# Force the non-interactive backend: this module only saves figures to disk, and the
# default backend can try to initialize a Tk display that isn't available headlessly.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402


def evaluate_classifier(model, x_test, y_test) -> dict:
    """Return common classification metrics."""
    predictions = model.predict(x_test)
    return classification_report(y_test, predictions, output_dict=True)


def plot_confusion_matrix(
    y_true,
    y_pred,
    output_path: str | Path,
    *,
    normalize: bool = True,
) -> Path:
    """Plot a confusion matrix and save it as a PNG.

    Row-normalized (share of each true class, i.e. per-class recall) by default: with
    the severe class imbalance in these datasets, raw counts are dominated entirely by
    the majority class and hide exactly the rare-class confusion patterns we care about.
    """
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        display_matrix = np.divide(
            matrix, row_sums, out=np.zeros(matrix.shape, dtype=float), where=row_sums != 0
        )
        fmt = ".2f"
        vmax = 1.0
        colorbar_label = "Share of true class (recall)"
    else:
        display_matrix = matrix
        fmt = "d"
        vmax = None
        colorbar_label = "Count"

    figure_size = max(6, len(labels) * 0.6)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    sns.heatmap(
        display_matrix,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": colorbar_label},
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_label_distribution(labels, output_path: str | Path, *, title: str | None = None) -> Path:
    """Plot a class count bar chart and save it as a PNG.

    Log-scaled and sorted descending: these datasets' class counts span multiple orders
    of magnitude (single digits to millions), which a linear scale would render as a
    single visible bar with everything else invisible at zero.
    """
    counts = labels.value_counts().sort_values(ascending=True)

    figure_height = max(3, len(counts) * 0.5)
    fig, ax = plt.subplots(figsize=(8, figure_height))
    bars = ax.barh(counts.index.astype(str), counts.to_numpy(), color="#4C72B0")
    ax.set_xscale("log")
    ax.set_xlabel("Count (log scale)")
    ax.set_ylabel("Class")
    if title:
        ax.set_title(title)
    ax.bar_label(bars, labels=[f"{count:,}" for count in counts], padding=3, fontsize=9)
    ax.margins(x=0.15)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path
