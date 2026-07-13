import pandas as pd

from xai_thesis.evaluation import plot_confusion_matrix, plot_label_distribution


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


def test_plot_label_distribution_writes_png(tmp_path) -> None:
    labels = pd.Series(["benign"] * 100 + ["attack_a"] * 10 + ["attack_b"] * 2)
    output_path = tmp_path / "label_distribution.png"

    result_path = plot_label_distribution(labels, output_path, title="test dataset")

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
