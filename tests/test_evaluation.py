import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from xai_thesis.evaluation import (
    _label_sort_key,
    evaluate_classifier,
    plot_confusion_matrix,
    plot_feature_correlation,
    plot_label_distribution,
    plot_missing_values,
    plot_roc_curves,
    plot_train_test_split_distribution,
)


def test_label_sort_key_orders_benign_first_then_alphabetical() -> None:
    labels = ["Reconnaissance", "Worms", "Data Exfiltration", "Benign"]

    assert sorted(labels, key=_label_sort_key) == ["Benign", "Data Exfiltration", "Reconnaissance", "Worms"]


def test_plot_confusion_matrix_normalized_writes_png(tmp_path) -> None:
    y_true = ["benign", "benign", "benign", "attack", "attack"]
    y_pred = ["benign", "benign", "attack", "attack", "benign"]
    output_path = tmp_path / "confusion_matrix.png"

    result_path = plot_confusion_matrix(y_true, y_pred, output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_confusion_matrix_raw_counts_writes_png(tmp_path) -> None:
    y_true = ["benign", "benign", "attack"]
    y_pred = ["benign", "attack", "attack"]
    output_path = tmp_path / "confusion_matrix.png"

    result_path = plot_confusion_matrix(y_true, y_pred, output_path, normalize=False)

    assert result_path == output_path
    assert output_path.exists()


def test_plot_roc_curves_writes_png(tmp_path) -> None:
    rng = np.random.default_rng(0)
    class_names = ["Benign", "attack_a", "attack_b"]
    y_true = rng.choice(class_names, size=120)
    # a plausible predict_proba: rows sum to 1, columns follow class_names order
    logits = rng.normal(size=(120, 3))
    y_score = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    output_path = tmp_path / "roc_curves.png"

    result_path = plot_roc_curves(y_true, y_score, class_names, output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_roc_curves_skips_class_absent_from_true(tmp_path) -> None:
    # "attack_b" is a valid column but never occurs in y_true - its ROC is undefined and
    # must be skipped rather than crash roc_curve.
    class_names = ["Benign", "attack_a", "attack_b"]
    y_true = ["Benign", "Benign", "attack_a", "attack_a", "Benign", "attack_a"]
    y_score = np.tile([0.5, 0.3, 0.2], (6, 1))
    output_path = tmp_path / "roc_curves.png"

    result_path = plot_roc_curves(y_true, y_score, class_names, output_path)

    assert result_path.exists()


def test_plot_confusion_matrix_handles_label_absent_from_true(tmp_path) -> None:
    # "unseen" is predicted but never actually occurs; its row would divide by zero
    # when normalizing unless guarded.
    y_true = ["benign", "benign", "attack"]
    y_pred = ["benign", "unseen", "attack"]
    output_path = tmp_path / "confusion_matrix.png"

    result_path = plot_confusion_matrix(y_true, y_pred, output_path)

    assert result_path.exists()


def test_evaluate_classifier_adds_macro_ovr_roc_and_pr_auc() -> None:
    x_train = [[0.0], [0.4], [0.5], [1.0]]
    y_train = ["Benign", "Reconnaissance", "Reconnaissance", "Data Exfiltration"]
    model = LogisticRegression().fit(x_train, y_train)

    report = evaluate_classifier(model, x_train, y_train)

    assert "roc_auc_macro" in report
    assert "pr_auc_macro" in report


def test_plot_label_distribution_writes_png(tmp_path) -> None:
    labels = pd.Series(["benign"] * 100 + ["attack_a"] * 10 + ["attack_b"] * 2)
    output_path = tmp_path / "label_distribution.png"

    result_path = plot_label_distribution(labels, output_path, title="test dataset")

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_missing_values_writes_png_with_missing_data(tmp_path) -> None:
    data = pd.DataFrame({"a": [1.0, None, 3.0], "b": [1.0, 2.0, 3.0]})
    output_path = tmp_path / "missing_values.png"

    result_path = plot_missing_values(data, output_path, title="test dataset")

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_missing_values_writes_placeholder_when_nothing_missing(tmp_path) -> None:
    data = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
    output_path = tmp_path / "missing_values.png"

    result_path = plot_missing_values(data, output_path, title="test dataset")

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_train_test_split_distribution_writes_png(tmp_path) -> None:
    train_labels = pd.Series(["benign"] * 8 + ["attack"] * 2)
    test_labels = pd.Series(["benign"] * 2 + ["attack"] * 1)
    output_path = tmp_path / "train_test_split.png"

    result_path = plot_train_test_split_distribution(train_labels, test_labels, output_path, title="test dataset")

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_train_test_split_distribution_handles_class_absent_from_one_split(tmp_path) -> None:
    # "rare" only appears in train, never in test - still possible under a stratified split
    # when a class has very few rows, and the plot must not crash on it.
    train_labels = pd.Series(["benign"] * 5 + ["rare"] * 1)
    test_labels = pd.Series(["benign"] * 2)
    output_path = tmp_path / "train_test_split.png"

    result_path = plot_train_test_split_distribution(train_labels, test_labels, output_path)

    assert result_path.exists()


def test_plot_feature_correlation_writes_png(tmp_path) -> None:
    data = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [4.0, 3.0, 2.0, 1.0],
            "categorical": ["x", "y", "x", "y"],
        }
    )
    output_path = tmp_path / "feature_correlation.png"

    result_path = plot_feature_correlation(data, output_path, title="test dataset")

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
