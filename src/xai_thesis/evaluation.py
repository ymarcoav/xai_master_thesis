from pathlib import Path

import matplotlib

# Force the non-interactive backend: this module only saves figures to disk, and the
# default backend can try to initialize a Tk display that isn't available headlessly.
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
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402


def evaluate_classifier(model, x_test, y_test) -> dict:
    """Return classification_report's per-class/macro metrics, plus macro one-vs-rest ROC-AUC
    and PR-AUC (average precision) - every dataset in this project's catalog is multiclass, so
    these are always the macro variants. This is the same masking-of-rare-class-failure concern
    that already motivates macro-averaging P/R/F1: macro PR-AUC in particular tends to track a
    struggling rare class more honestly than macro ROC-AUC, which can look deceptively strong
    under severe imbalance since the true-negative rate stays high almost by default when the
    negative class dominates.

    The len(classes) > 2 guard isn't a binary/multiclass special-case (that was removed - every
    real dataset here has well over 2 classes) - it reflects a genuine sklearn API constraint:
    roc_auc_score's multi_class="ovr" and average_precision_score's macro averaging both reject
    exactly-2-class input outright (confirmed directly - label_binarize collapses 2 classes to a
    single column, which then shape-mismatches against predict_proba's 2 columns). Toy 2-class
    test fixtures elsewhere in this project exist for reasons unrelated to metric computation, so
    this guard keeps them from crashing rather than expecting every fixture to use 3+ classes.
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
    """Benign first (this project's convention across every dataset), then everything else
    alphabetically.
    """
    if label == "Benign":
        return (0, "")
    return (1, str(label))


def plot_confusion_matrix(
    y_true,
    y_pred,
    output_path: str | Path,
    *,
    normalize: bool = False,
) -> Path:
    """Plot a confusion matrix and save it as a PNG.

    Absolute counts for both the annotated numbers and the color scale by default, the
    latter on a symmetric-log scale: with severe class imbalance, a *linear* absolute
    color scale is dominated entirely by the single largest cell, making every other cell
    (including the rare-class confusion patterns we actually care about) look uniformly
    pale. Log-scaling the color (not the annotated numbers, which stay exact) keeps cells
    that differ by orders of magnitude more distinct than a linear scale would. SymLogNorm
    (rather than plain LogNorm) is used specifically so exact-zero cells - common once a
    class has few enough errors - render as a normal, visible white rather than being
    masked out entirely, which plain log scaling does to non-positive values.

    The scale's vmax is deliberately the largest value *excluding* the single most extreme
    cell (almost always the majority class's own diagonal entry, often orders of magnitude
    bigger than everything else), not that cell itself: anchoring vmax to that one outlier
    left it consuming most of the color range on its own, compressing every other cell -
    including the off-diagonal error cells that are the actual point of a confusion matrix -
    into a visually similar dark band near the top, exactly where the eye needs the most
    separation. Letting that one outlier clip to the same top color (the colorbar's "extend"
    arrow signals this) frees up the full light-to-dark range for everything else instead.

    linthresh (SymLogNorm's linear-region half-width around zero) is set to 5% of that
    robust vmax rather than a fixed constant like 1: with linthresh=1, every value >=1
    immediately starts climbing the log curve, so cells that are still small relative to
    vmax (tens to low hundreds) rendered as a noticeably dark blue rather than the pale
    shade their actual share of the range would suggest - checked empirically against a
    real confusion matrix and tuned by comparing several linthresh values side by side, not
    guessed. A percentage of vmax (rather than one fixed number) keeps this behavior
    consistent whether vmax is a few thousand (DAPT2020) or several hundred thousand
    (CIC-IDS2017).

    Pass normalize=True to color by row-normalized share of each true class (per-class
    recall) instead, while still annotating the absolute count in each cell either way.
    """
    labels = sorted(set(y_true) | set(y_pred), key=_label_sort_key)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        display_matrix = np.divide(
            matrix, row_sums, out=np.zeros(matrix.shape, dtype=float), where=row_sums != 0
        )
        norm = None
        vmin, vmax = 0, 1.0
        colorbar_label = "Share of true class (recall)"
        cbar_extend = "neither"
    else:
        display_matrix = matrix
        below_peak = matrix[matrix < matrix.max()]
        robust_vmax = below_peak.max() if below_peak.size > 0 else matrix.max()
        robust_vmax = max(robust_vmax, 1)
        norm = SymLogNorm(linthresh=max(1, robust_vmax * 0.05), vmin=0, vmax=robust_vmax)
        vmin = vmax = None
        colorbar_label = "Count (log scale)"
        cbar_extend = "max" if matrix.max() > robust_vmax else "neither"

    figure_size = max(6, len(labels) * 0.6)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    # A colorbar built via sns.heatmap's own default placement is sized relative to the
    # whole figure, not the heatmap's actual plotted box - with square=True forcing the
    # heatmap into a square (which need not fill the figure height), the two drift apart and
    # the colorbar ends up visibly taller than the grid. make_axes_locatable ties the
    # colorbar's own axes to the heatmap axes' actual size, so it always matches.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.3)
    sns.heatmap(
        display_matrix,
        annot=matrix,
        fmt="d",
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar_ax=cax,
        cbar_kws={"label": colorbar_label, "extend": cbar_extend},
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


def plot_roc_curves(
    y_true,
    y_score: np.ndarray,
    class_names: list,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """One-vs-rest ROC curve per class plus micro- and macro-averaged curves, on one axes.

    y_score is the (n_samples, n_classes) predict_proba matrix; its columns must be in
    class_names order (i.e. model.classes_). A class with no positive instances in y_true is
    skipped - undefined ROC - which does not happen on the real stratified test splits but
    keeps toy fixtures from crashing.

    Under this project's severe class imbalance the per-class and macro curves are the
    informative ones: the micro-average is dominated by the majority Benign class and hugs
    the top-left corner almost regardless of rare-class separation. It is drawn for
    completeness, but the macro-average (equal weight per class) is the summary that tracks
    rare-class performance, mirroring the macro-F1 / macro-PR-AUC choice elsewhere.
    """
    class_names = list(class_names)
    y_binarized = label_binarize(y_true, classes=class_names)
    if y_binarized.shape[1] == 1:  # label_binarize collapses exactly 2 classes to one column
        y_binarized = np.hstack([1 - y_binarized, y_binarized])

    fig, ax = plt.subplots(figsize=(7.5, 7))
    fpr_grid = np.linspace(0.0, 1.0, 500)
    interpolated_tpr_sum = np.zeros_like(fpr_grid)
    curves_counted = 0

    for index, name in enumerate(class_names):
        if y_binarized[:, index].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_binarized[:, index], y_score[:, index])
        ax.plot(fpr, tpr, lw=1.2, alpha=0.85, color=plt.cm.tab10(index % 10), label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
        interpolated_tpr_sum += np.interp(fpr_grid, fpr, tpr)
        curves_counted += 1

    if curves_counted:
        macro_tpr = interpolated_tpr_sum / curves_counted
        ax.plot(
            fpr_grid, macro_tpr, lw=2.8, color="black",
            label=f"macro-average (AUC = {auc(fpr_grid, macro_tpr):.3f})",
        )

    micro_fpr, micro_tpr, _ = roc_curve(y_binarized.ravel(), np.asarray(y_score).ravel())
    ax.plot(
        micro_fpr, micro_tpr, lw=2.0, ls="--", color="dimgray",
        label=f"micro-average (AUC = {auc(micro_fpr, micro_tpr):.3f})",
    )

    ax.plot([0, 1], [0, 1], lw=1.0, ls=":", color="gray")
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_missing_values(data: pd.DataFrame, output_path: str | Path, *, title: str | None = None) -> Path:
    """Plot the % of missing values per column, for columns with at least one missing value.

    Sorted descending. If no column has any missing values, saves a placeholder figure
    stating that instead of an empty chart - a "verified clean" result is still worth
    recording, not just a silent no-op.
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
    """Plot per-class train vs test counts as grouped bars.

    A visual check that every class carries reasonable representation on both sides of a
    dataset's split - the stratified random split should hold class proportions, but the
    rarest classes (e.g. UNSW-NB15 Worms, CIC-IDS2017 Infiltration) have few enough rows
    that it is still worth confirming.
    """
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
    """Correlation heatmap over numeric feature columns, to spot redundant/highly correlated
    features before modeling. Non-numeric (categorical) columns are dropped automatically.
    """
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
