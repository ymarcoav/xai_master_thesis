from pathlib import Path

import pandas as pd
import pytest
import yaml

from xai_thesis.pipeline import preprocess_dataset, run_experiment


def _write_unsw_fixture(raw_dir: Path) -> None:
    train = pd.DataFrame(
        {
            "id": [1, 2],
            "dur": [0.1, 0.2],
            "proto": ["tcp", "udp"],
            "service": ["http", "-"],
            "state": ["FIN", "CON"],
            "is_ftp_login": [0, 2],
            "attack_cat": ["Normal", "Exploits"],
            "label": [0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "id": [3, 4],
            "dur": [0.3, 0.4],
            "proto": ["tcp", "tcp"],
            "service": ["ftp", "http"],
            "state": ["INT", "FIN"],
            "is_ftp_login": [1, 0],
            "attack_cat": ["Normal", "DoS"],
            "label": [0, 1],
        }
    )
    train.to_csv(raw_dir / "UNSW_NB15_training-set.csv", index=False)
    test.to_csv(raw_dir / "UNSW_NB15_testing-set.csv", index=False)


def test_preprocess_dataset_writes_cleaned_train_and_test_csv(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    _write_unsw_fixture(raw_dir)

    config_path = tmp_path / "unsw_nb15.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "unsw_nb15",
                "raw_path": raw_dir.as_posix(),
                "processed_path": processed_dir.as_posix(),
                "label_column": "attack_cat",
                "binary_label_column": "label",
                "drop_columns": ["id", "label"],
                "categorical_columns": ["proto", "service", "state"],
            }
        ),
        encoding="utf-8",
    )

    train_path, test_path = preprocess_dataset(config_path)

    assert train_path == processed_dir / "train.csv"
    assert test_path == processed_dir / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    assert "id" not in train_df.columns
    assert "label" not in train_df.columns
    assert "split" not in train_df.columns
    assert len(train_df) == 2
    assert len(test_df) == 2
    assert "unknown" in train_df["service"].tolist()  # "-" sentinel normalized
    assert train_df["is_ftp_login"].max() == 1  # clipped from 2


def test_run_experiment_trains_and_saves_model_and_metrics(tmp_path) -> None:
    dataset_config_dir = tmp_path / "datasets"
    dataset_config_dir.mkdir()
    processed_dir = tmp_path / "processed" / "toy"
    processed_dir.mkdir(parents=True)

    train_df = pd.DataFrame(
        {
            "feature": [0.0, 1.0, 0.0, 1.0, 0.5, 0.6],
            "label": ["benign", "attack", "benign", "attack", "benign", "attack"],
        }
    )
    test_df = pd.DataFrame({"feature": [0.1, 0.9], "label": ["benign", "attack"]})
    train_df.to_csv(processed_dir / "train.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    (dataset_config_dir / "toy.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "toy",
                "raw_path": "unused",
                "processed_path": processed_dir.as_posix(),
                "label_column": "label",
            }
        ),
        encoding="utf-8",
    )

    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        yaml.safe_dump(
            {
                "name": "toy_experiment",
                "dataset": "toy",
                "model": "random_forest",
                "random_state": 42,
                "model_params": {"n_estimators": 5},
                "preprocessing": {"strategy": "default_tabular"},
                "metrics": ["accuracy", "f1_macro"],
            }
        ),
        encoding="utf-8",
    )

    models_dir = tmp_path / "models"
    metrics_dir = tmp_path / "metrics"

    metrics = run_experiment(
        experiment_path,
        dataset_config_dir=dataset_config_dir,
        models_dir=models_dir,
        metrics_dir=metrics_dir,
    )

    assert set(metrics) == {"accuracy", "f1_macro"}
    assert (models_dir / "toy_experiment.joblib").exists()
    assert (metrics_dir / "toy_experiment.json").exists()


def test_run_experiment_rejects_unknown_preprocessing_strategy(tmp_path) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        yaml.safe_dump(
            {
                "name": "toy_experiment",
                "dataset": "toy",
                "model": "random_forest",
                "preprocessing": {"strategy": "fancy"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(NotImplementedError):
        run_experiment(experiment_path)
