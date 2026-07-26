import pandas as pd
from sklearn.linear_model import LogisticRegression

from xai_thesis.evaluation import _label_sort_key, evaluate_classifier, plot_confusion_matrix, plot_label_distribution


def test_label_sort_key_orders_benign_first_attack_second_then_alphabetical() -> None:
    labels = ["Reconnaissance", "Attack", "Data Exfiltration", "Benign"]

    assert sorted(labels, key=_label_sort_key) == ["Benign", "Attack", "Data Exfiltration", "Reconnaissance"]


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


def test_plot_confusion_matrix_handles_label_absent_from_true(tmp_path) -> None:
    # "unseen" is predicted but never actually occurs; its row would divide by zero
    # when normalizing unless guarded.
    y_true = ["benign", "benign", "attack"]
    y_pred = ["benign", "unseen", "attack"]
    output_path = tmp_path / "confusion_matrix.png"

    result_path = plot_confusion_matrix(y_true, y_pred, output_path)

    assert result_path.exists()


def test_evaluate_classifier_adds_roc_auc_and_pr_auc_for_binary_attack_benign() -> None:
    x_train = [[0.0], [0.1], [0.9], [1.0]]
    y_train = ["Benign", "Benign", "Attack", "Attack"]
    model = LogisticRegression().fit(x_train, y_train)

    report = evaluate_classifier(model, x_train, y_train)

    assert "roc_auc" in report
    assert "pr_auc" in report
    assert report["Attack"]["precision"] == report["Attack"]["precision"]  # present, not KeyError


def test_evaluate_classifier_omits_roc_auc_for_multiclass() -> None:
    x_train = [[0.0], [0.4], [0.5], [1.0]]
    y_train = ["Benign", "Reconnaissance", "Reconnaissance", "Data Exfiltration"]
    model = LogisticRegression().fit(x_train, y_train)

    report = evaluate_classifier(model, x_train, y_train)

    assert "roc_auc" not in report
    assert "pr_auc" not in report


def test_plot_label_distribution_writes_png(tmp_path) -> None:
    labels = pd.Series(["benign"] * 100 + ["attack_a"] * 10 + ["attack_b"] * 2)
    output_path = tmp_path / "label_distribution.png"

    result_path = plot_label_distribution(labels, output_path, title="test dataset")

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
