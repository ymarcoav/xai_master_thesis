from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from xai_thesis.pipeline import (
    generate_confusion_matrix,
    generate_explanations,
    generate_label_distribution,
    preprocess_dataset,
    run_experiment,
    tune_hyperparameters,
)


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
    assert set(train_df["attack_cat"]) <= {"Benign", "Exploits", "DoS"}  # "Normal" mapped to "Benign"
    assert "unknown" in train_df["service"].tolist()  # "-" sentinel normalized
    assert train_df["is_ftp_login"].max() == 1  # clipped from 2


def test_generate_label_distribution_writes_png(tmp_path) -> None:
    processed_dir = tmp_path / "processed" / "toy"
    processed_dir.mkdir(parents=True)
    pd.DataFrame({"feature": [0.0, 1.0, 0.0], "label": ["benign", "attack", "benign"]}).to_csv(
        processed_dir / "train.csv", index=False
    )
    pd.DataFrame({"feature": [0.2], "label": ["benign"]}).to_csv(processed_dir / "test.csv", index=False)

    dataset_config_path = tmp_path / "toy.yaml"
    dataset_config_path.write_text(
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

    plots_dir = tmp_path / "plots"
    output_path = generate_label_distribution(dataset_config_path, plots_dir=plots_dir)

    assert output_path == plots_dir / "toy" / "label_distribution.png"
    assert output_path.exists()


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
                "model": "gradient_boosting",
                "random_state": 42,
                "model_params": {"max_iter": 5},
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


def test_run_experiment_neural_network_fits_preprocessing_before_resampling(tmp_path) -> None:
    # 8 "benign" rows with feature=1.0 and 2 "attack" rows with feature=100.0. The true,
    # unresampled median of this column is 1.0. balance_classes resamples both classes to
    # the median class count (5), which shifts the *resampled* 10-row median to 50.5 - if
    # preprocessing's imputer fit on that resampled data instead of the original
    # distribution (the bug being fixed), this test would see 50.5 instead of 1.0.
    dataset_config_dir = tmp_path / "datasets"
    dataset_config_dir.mkdir()
    processed_dir = tmp_path / "processed" / "toy"
    processed_dir.mkdir(parents=True)

    train_df = pd.DataFrame(
        {
            "feature": [1.0] * 8 + [100.0] * 2,
            "label": ["benign"] * 8 + ["attack"] * 2,
        }
    )
    test_df = pd.DataFrame({"feature": [1.0, 100.0], "label": ["benign", "attack"]})
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
                "model": "neural_network",
                "random_state": 42,
                "model_params": {"max_iter": 5},
                "preprocessing": {"strategy": "default_tabular"},
                "metrics": ["accuracy"],
            }
        ),
        encoding="utf-8",
    )

    models_dir = tmp_path / "models"
    metrics_dir = tmp_path / "metrics"

    run_experiment(
        experiment_path, dataset_config_dir=dataset_config_dir, models_dir=models_dir, metrics_dir=metrics_dir
    )

    pipeline = joblib.load(models_dir / "toy_experiment.joblib")
    imputer = pipeline.named_steps["preprocessing"].named_transformers_["numeric"].named_steps["imputer"]
    assert imputer.statistics_[0] == 1.0

    # And the fitted pipeline must still be directly usable for prediction, since it was
    # assembled from separately-fitted preprocessing + model rather than via pipeline.fit().
    predictions = pipeline.predict(test_df[["feature"]])
    assert len(predictions) == 2


def test_generate_confusion_matrix_writes_png_for_trained_experiment(tmp_path) -> None:
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
                "model": "gradient_boosting",
                "random_state": 42,
                "model_params": {"max_iter": 5},
                "preprocessing": {"strategy": "default_tabular"},
                "metrics": ["accuracy"],
            }
        ),
        encoding="utf-8",
    )

    models_dir = tmp_path / "models"
    metrics_dir = tmp_path / "metrics"
    plots_dir = tmp_path / "plots"

    run_experiment(
        experiment_path, dataset_config_dir=dataset_config_dir, models_dir=models_dir, metrics_dir=metrics_dir
    )
    output_path = generate_confusion_matrix(
        experiment_path, dataset_config_dir=dataset_config_dir, models_dir=models_dir, plots_dir=plots_dir
    )

    assert output_path == plots_dir / "toy_experiment" / "confusion_matrix.png"
    assert output_path.exists()


def test_tune_hyperparameters_writes_best_params_into_experiment_yaml(tmp_path) -> None:
    dataset_config_dir = tmp_path / "datasets"
    dataset_config_dir.mkdir()
    processed_dir = tmp_path / "processed" / "toy"
    processed_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    feature = rng.normal(size=60)
    label = ["attack" if value > 0 else "benign" for value in feature]
    pd.DataFrame({"feature": feature, "label": label}).to_csv(processed_dir / "train.csv", index=False)
    pd.DataFrame({"feature": [0.1, -0.1], "label": ["attack", "benign"]}).to_csv(
        processed_dir / "test.csv", index=False
    )

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
                "model": "gradient_boosting",
                "random_state": 42,
                "preprocessing": {"strategy": "default_tabular"},
                "metrics": ["accuracy"],
            }
        ),
        encoding="utf-8",
    )

    tuned_params = tune_hyperparameters(experiment_path, dataset_config_dir=dataset_config_dir, n_iter=3, cv=3)

    assert set(tuned_params) == {
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "max_depth",
        "l2_regularization",
        "min_samples_leaf",
    }

    updated_experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    assert updated_experiment["model_params"] == tuned_params


def test_tune_hyperparameters_logistic_regression_round_trip_retrains_successfully(tmp_path) -> None:
    dataset_config_dir = tmp_path / "datasets"
    dataset_config_dir.mkdir()
    processed_dir = tmp_path / "processed" / "toy"
    processed_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    feature = rng.normal(size=60)
    label = ["attack" if value > 0 else "benign" for value in feature]
    pd.DataFrame({"feature": feature, "label": label}).to_csv(processed_dir / "train.csv", index=False)
    pd.DataFrame({"feature": [0.1, -0.1], "label": ["attack", "benign"]}).to_csv(
        processed_dir / "test.csv", index=False
    )

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
                "model": "logistic_regression",
                "random_state": 42,
                "preprocessing": {"strategy": "default_tabular"},
                "metrics": ["accuracy"],
            }
        ),
        encoding="utf-8",
    )

    tuned_params = tune_hyperparameters(experiment_path, dataset_config_dir=dataset_config_dir, n_iter=3, cv=3)

    assert set(tuned_params) == {"C"}

    run_experiment(
        experiment_path,
        dataset_config_dir=dataset_config_dir,
        models_dir=tmp_path / "models",
        metrics_dir=tmp_path / "metrics",
    )


def test_run_experiment_rejects_unknown_preprocessing_strategy(tmp_path) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        yaml.safe_dump(
            {
                "name": "toy_experiment",
                "dataset": "toy",
                "model": "gradient_boosting",
                "preprocessing": {"strategy": "fancy"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(NotImplementedError):
        run_experiment(experiment_path)


def test_generate_explanations_rejects_dataset_without_class_of_interest_mapping(tmp_path) -> None:
    dataset_config_dir = tmp_path / "datasets"
    dataset_config_dir.mkdir()
    processed_dir = tmp_path / "processed" / "toy"
    processed_dir.mkdir(parents=True)

    pd.DataFrame({"feature": [0.0, 1.0], "label": ["benign", "attack"]}).to_csv(
        processed_dir / "train.csv", index=False
    )
    pd.DataFrame({"feature": [0.1, 0.9], "label": ["benign", "attack"]}).to_csv(
        processed_dir / "test.csv", index=False
    )

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
                "model": "gradient_boosting",
                "preprocessing": {"strategy": "default_tabular"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(NotImplementedError):
        generate_explanations(experiment_path, dataset_config_dir=dataset_config_dir)


def test_generate_explanations_writes_all_expected_plots(tmp_path) -> None:
    dataset_config_dir = tmp_path / "datasets"
    dataset_config_dir.mkdir()
    processed_dir = tmp_path / "processed" / "dapt2020"
    processed_dir.mkdir(parents=True)

    # Reconnaissance/Benign overlap deliberately, so the model both classifies some
    # Reconnaissance instances correctly and confuses others with Benign - exercising both
    # the "correct" and "misclassified" curated-instance paths (see _CLASS_OF_INTEREST).
    # protocol_type is categorical and strongly predictive, so it ranks high in permutation
    # importance - a regression check that PDP correctly excludes categorical columns from
    # its numeric-only feature selection (sklearn's PDP errors outright on string columns).
    rng = np.random.default_rng(0)
    n = 60
    protocol_train = rng.choice(["tcp", "udp"], n)
    stage_train = np.where(protocol_train == "tcp", "Reconnaissance", "Benign")
    noisy_mask = rng.random(n) < 0.2
    stage_train[noisy_mask] = rng.choice(["Benign", "Reconnaissance", "Establish Foothold"], noisy_mask.sum())
    train_df = pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1.5, n),
            "feature_b": rng.normal(0, 1.5, n),
            "protocol_type": protocol_train,
            "Stage": stage_train,
        }
    )
    protocol_test = rng.choice(["tcp", "udp"], 20)
    test_df = pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1.5, 20),
            "feature_b": rng.normal(0, 1.5, 20),
            "protocol_type": protocol_test,
            "Stage": np.where(protocol_test == "tcp", "Reconnaissance", "Benign"),
        }
    )
    train_df.to_csv(processed_dir / "train.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    (dataset_config_dir / "dapt2020.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "dapt2020",
                "raw_path": "unused",
                "processed_path": processed_dir.as_posix(),
                "label_column": "Stage",
                "categorical_columns": ["protocol_type"],
            }
        ),
        encoding="utf-8",
    )

    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        yaml.safe_dump(
            {
                "name": "dapt2020_test_experiment",
                "dataset": "dapt2020",
                "model": "gradient_boosting",
                "random_state": 42,
                "model_params": {"max_iter": 10},
                "preprocessing": {"strategy": "default_tabular"},
                "metrics": ["accuracy"],
            }
        ),
        encoding="utf-8",
    )

    models_dir = tmp_path / "models"
    metrics_dir = tmp_path / "metrics"
    plots_dir = tmp_path / "plots"

    run_experiment(
        experiment_path, dataset_config_dir=dataset_config_dir, models_dir=models_dir, metrics_dir=metrics_dir
    )
    paths = generate_explanations(
        experiment_path, dataset_config_dir=dataset_config_dir, models_dir=models_dir, plots_dir=plots_dir
    )

    assert {"shap_bar", "shap_beeswarm", "permutation_importance", "partial_dependence"} <= set(paths)
    for path in paths.values():
        assert path.exists()
