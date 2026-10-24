"""Classification metrics and the data, performance and evaluation plots."""

from pathlib import Path

import matplotlib

# Non-interactive backend: figures are only saved to disk.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import SymLogNorm  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize  # noqa: E402


def evaluate_classifier(model, x_test, y_test) -> dict:
    """Return classification_report's per-class and macro metrics, plus the macro-averaged
    one-vs-rest ROC-AUC and PR-AUC (average precision).

    The AUCs are computed for more than two classes only, because the one-vs-rest averaging used
    here does not accept two-class input.
    """
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, output_dict=True)

    classes = list(model.classes_)
    if len(classes) > 2 and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)
        report["roc_auc_macro"] = roc_auc_score(
            y_test, probabilities, multi_class="ovr", average="macro", labels=classes
        )
        y_true_binarized = label_binarize(y_test, classes=classes)
        report["pr_auc_macro"] = average_precision_score(y_true_binarized, probabilities, average="macro")

    return report


def _label_sort_key(label: object) -> tuple[int, str]:
    """Benign first, then the other classes alphabetically."""
    if label == "Benign":
        return (0, "")
    return (1, str(label))


def confusion_matrix_frame(y_true, y_pred) -> pd.DataFrame:
    """Confusion-matrix counts as a labelled frame: rows are true classes, columns predicted
    ones, both in plot_confusion_matrix's order (Benign first, then alphabetical)."""
    labels = sorted(set(y_true) | set(y_pred), key=_label_sort_key)
    frame = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)
    frame.index.name = "true_class"
    frame.columns.name = "predicted_class"
    return frame


# The confusion matrices are printed two side by side at 0.49\textwidth (about 2.85 in) in the
# thesis, so they are drawn at that size and their fonts come out at these point sizes in print.
CONFUSION_MATRIX_SIZE_INCHES = 2.9
# Counts are set in Arial Narrow so that the six-digit Benign-as-Benign count fits a cell of the
# ten-class UNSW-NB15 grid at 6 pt; plain Arial fits only at 5 pt and touches the cell edges.
CONFUSION_MATRIX_ANNOTATION_SIZE = 6.0
CONFUSION_MATRIX_ANNOTATION_FONT = "Arial Narrow"
CONFUSION_MATRIX_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.labelsize": 7,
}
# Tick labels too long to leave the cells room for six-digit counts.
CONFUSION_MATRIX_LABEL_ABBREVIATIONS = {"Reconnaissance": "Recon."}


def plot_confusion_matrix(
    y_true,
    y_pred,
    output_path: str | Path,
    *,
    normalize: bool = False,
) -> Path:
    """Plot a confusion matrix at its printed size and save it as a PNG.

    Cells are annotated with exact counts; the colour scale is symmetric-logarithmic because
    the counts span several orders of magnitude. The scale's maximum is the second-largest class
    total, so the Benign diagonal does not use up the whole colour range, and every confusion
    matrix of a dataset shares the same scale. There is no colorbar, to leave the grid as much
    width as possible, so the figure caption has to state the log scale. Pass normalize=True to
    colour by each true class's row share (recall) instead.
    """
    frame = confusion_matrix_frame(y_true, y_pred)
    labels, matrix = list(frame.index), frame.to_numpy()

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        display_matrix = np.divide(
            matrix, row_sums, out=np.zeros(matrix.shape, dtype=float), where=row_sums != 0
        )
        norm = None
        vmin, vmax = 0, 1.0
    else:
        display_matrix = matrix
        # Second-largest row sum (see the docstring); the largest cell for a single-class matrix.
        class_totals = np.sort(matrix.sum(axis=1))
        robust_vmax = int(class_totals[-2]) if class_totals.size > 1 else int(matrix.max())
        robust_vmax = max(robust_vmax, 1)
        norm = SymLogNorm(linthresh=max(1, robust_vmax * 0.05), vmin=0, vmax=robust_vmax)
        vmin = vmax = None

    tick_labels = [CONFUSION_MATRIX_LABEL_ABBREVIATIONS.get(str(label), label) for label in labels]
    with plt.rc_context(CONFUSION_MATRIX_STYLE):
        fig, ax = plt.subplots(figsize=(CONFUSION_MATRIX_SIZE_INCHES, CONFUSION_MATRIX_SIZE_INCHES))
        sns.heatmap(
            display_matrix,
            annot=matrix,
            fmt="d",
            annot_kws={"size": CONFUSION_MATRIX_ANNOTATION_SIZE, "family": CONFUSION_MATRIX_ANNOTATION_FONT},
            cmap="Blues",
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            square=True,
            cbar=False,
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted label", labelpad=2)
        ax.set_ylabel("True label", labelpad=2)
        ax.tick_params(length=0, pad=1.5)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0)
        fig.tight_layout(pad=0.1)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return output_path


def plot_roc_curves(
    y_true,
    y_score: np.ndarray,
    class_names: list,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Plot the macro- and micro-averaged one-vs-rest ROC curves and the chance diagonal.

    The macro average weights every class equally; the micro average is dominated by the Benign
    class. y_score is the (n_samples, n_classes) predict_proba matrix, columns in class_names
    order. A class without positive instances in y_true is skipped.
    """
    class_names = list(class_names)
    y_binarized = label_binarize(y_true, classes=class_names)
    if y_binarized.shape[1] == 1:  # label_binarize collapses exactly 2 classes to one column
        y_binarized = np.hstack([1 - y_binarized, y_binarized])
    y_score = np.asarray(y_score)

    fpr_grid = np.linspace(0.0, 1.0, 1000)
    per_class_tpr = [
        np.interp(fpr_grid, *roc_curve(y_binarized[:, index], y_score[:, index])[:2])
        for index in range(y_binarized.shape[1])
        if y_binarized[:, index].sum() > 0
    ]
    micro_fpr, micro_tpr, _ = roc_curve(y_binarized.ravel(), y_score.ravel())

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if per_class_tpr:
        macro_tpr = np.mean(per_class_tpr, axis=0)
        ax.plot(
            fpr_grid, macro_tpr, color="#1f3b73", lw=2.5,
            label=f"Macro-average  (AUC = {auc(fpr_grid, macro_tpr):.3f})",
        )
    ax.plot(
        micro_fpr, micro_tpr, color="#c44e52", lw=2.0, ls="--",
        label=f"Micro-average  (AUC = {auc(micro_fpr, micro_tpr):.3f})",
    )
    ax.plot([0, 1], [0, 1], color="gray", lw=1.0, ls=":", label="Chance")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_missing_values(data: pd.DataFrame, output_path: str | Path, *, title: str | None = None) -> Path:
    """Plot the percentage of missing values per column, for columns with any missing value. If
    there are none, the figure says so.
    """
    missing_percent = (data.isna().mean() * 100).sort_values(ascending=True)
    missing_percent = missing_percent[missing_percent > 0]

    if missing_percent.empty:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(
            0.5,
            0.5,
            f"No missing values in any of {data.shape[1]} columns",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.axis("off")
    else:
        figure_height = max(3, len(missing_percent) * 0.35)
        fig, ax = plt.subplots(figsize=(8, figure_height))
        bars = ax.barh(missing_percent.index.astype(str), missing_percent.to_numpy(), color="#C44E52")
        ax.set_xlabel("Missing (%)")
        ax.set_ylabel("Column")
        ax.bar_label(bars, labels=[f"{value:.2f}%" for value in missing_percent], padding=3, fontsize=9)
        ax.margins(x=0.15)

    if title:
        ax.set_title(title)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_train_test_split_distribution(
    train_labels, test_labels, output_path: str | Path, *, title: str | None = None
) -> Path:
    """Plot per-class train and test counts as grouped bars."""
    train_counts = train_labels.value_counts()
    test_counts = test_labels.value_counts()
    classes = sorted(
        set(train_counts.index) | set(test_counts.index),
        key=lambda label: train_counts.get(label, 0) + test_counts.get(label, 0),
    )

    y_positions = np.arange(len(classes))
    bar_height = 0.35

    figure_height = max(3, len(classes) * 0.5)
    fig, ax = plt.subplots(figsize=(9, figure_height))
    ax.barh(
        y_positions + bar_height / 2,
        [train_counts.get(label, 0) for label in classes],
        height=bar_height,
        label="Train",
        color="#4C72B0",
    )
    ax.barh(
        y_positions - bar_height / 2,
        [test_counts.get(label, 0) for label in classes],
        height=bar_height,
        label="Test",
        color="#DD8452",
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(label) for label in classes])
    ax.set_xscale("log")
    ax.set_xlabel("Count (log scale)")
    ax.set_ylabel("Class")
    ax.legend()
    if title:
        ax.set_title(title)
    ax.margins(x=0.15)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_feature_correlation(data: pd.DataFrame, output_path: str | Path, *, title: str | None = None) -> Path:
    """Correlation heatmap over the numeric feature columns."""
    numeric_data = data.select_dtypes(include=[np.number])
    correlation = numeric_data.corr()

    figure_size = min(28, max(6, len(correlation.columns) * 0.3))
    label_size = max(4, min(9, 300 // max(1, len(correlation.columns))))
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    sns.heatmap(
        correlation,
        ax=ax,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        cbar_kws={"shrink": 0.7},
        xticklabels=True,
        yticklabels=True,
    )
    ax.tick_params(axis="x", labelsize=label_size, rotation=90)
    ax.tick_params(axis="y", labelsize=label_size, rotation=0)
    if title:
        ax.set_title(title)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_label_distribution(labels, output_path: str | Path, *, title: str | None = None) -> Path:
    """Plot class counts as a horizontal bar chart on a logarithmic scale."""
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
