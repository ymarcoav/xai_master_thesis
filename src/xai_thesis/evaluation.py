from pathlib import Path

import matplotlib

# Force the non-interactive backend: this module only saves figures to disk, and the
# default backend can try to initialize a Tk display that isn't available headlessly.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import SymLogNorm  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# The project's binary datasets (UNSW-NB15, CIC-IDS2017) always name their positive class
# "Attack" (see the dataset adapters' clean() methods); DAPT2020 stays multiclass and has no
# single positive class, so ROC-AUC/PR-AUC are only ever computed when this label is present.
_POSITIVE_LABEL = "Attack"


def evaluate_classifier(model, x_test, y_test) -> dict:
    """Return classification_report's per-class/macro metrics, plus score-based metrics that
    accuracy and macro-averaged P/R/F1 alone can miss under the severe class imbalance in
    these datasets: ROC-AUC and PR-AUC (average precision) for the Attack class when the
    problem is binary Attack/Benign, or macro one-vs-rest ROC-AUC when it's multiclass (only
    DAPT2020 currently) - the latter is the same masking-of-rare-class-failure concern that
    already motivates macro-averaging P/R/F1 there, just applied to a threshold-independent
    score instead of a thresholded one.
    """
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, output_dict=True)

    classes = list(model.classes_)
    if len(classes) == 2 and _POSITIVE_LABEL in classes and hasattr(model, "predict_proba"):
        positive_index = classes.index(_POSITIVE_LABEL)
        attack_scores = model.predict_proba(x_test)[:, positive_index]
        y_true_binary = (np.asarray(y_test) == _POSITIVE_LABEL).astype(int)
        report["roc_auc"] = roc_auc_score(y_true_binary, attack_scores)
        report["pr_auc"] = average_precision_score(y_true_binary, attack_scores)
    elif len(classes) > 2 and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)
        report["roc_auc_macro"] = roc_auc_score(
            y_test, probabilities, multi_class="ovr", average="macro", labels=classes
        )

    return report


def _label_sort_key(label: object) -> tuple[int, str]:
    """Benign first, Attack second (this project's binary convention: Benign=0, Attack=1),
    then everything else alphabetically - applies equally to the multiclass DAPT2020 case,
    where Benign is still a natural reference class even though there's no numbered
    convention for the remaining kill-chain stages.
    """
    if label == "Benign":
        return (0, "")
    if label == "Attack":
        return (1, "")
    return (2, str(label))


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
    that differ by orders of magnitude all visibly distinct. SymLogNorm (rather than plain
    LogNorm) is used specifically so exact-zero cells - common once a class has few enough
    errors - render as a normal, visible white rather than being masked out entirely, which
    plain log scaling does to non-positive values.

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
    else:
        display_matrix = matrix
        norm = SymLogNorm(linthresh=1, vmin=0, vmax=max(matrix.max(), 1))
        vmin = vmax = None
        colorbar_label = "Count (log scale)"

    figure_size = max(6, len(labels) * 0.6)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    sns.heatmap(
        display_matrix,
        annot=matrix,
        fmt="d",
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
        norm=norm,
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
