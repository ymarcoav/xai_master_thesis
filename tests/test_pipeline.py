from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from xai_thesis.pipeline import (
    generate_confusion_matrix,
    generate_explanations,
    generate_feature_correlation,
    generate_label_distribution,
    generate_missing_value_summary,
    generate_train_test_split_comparison,
    preprocess_dataset,
    run_experiment,
    tune_hyperparameters,
)


def _unsw_row(*, dur: str, service: str, attack_cat: str, label: str, stime: str, is_ftp_login: str = "0") -> list[str]:
    """One raw UNSW-NB15 row, positionally matching datasets/unsw_nb15.py's _COLUMN_NAMES
    order exactly (the raw files are headerless - column identity comes purely from
    position), with everything not under test held at a fixed, valid, non-corrupted value."""
    return [
        "59.166.0.0",  # srcip
        "1390",  # sport
        "149.171.126.6",  # dstip
        "53",  # dsport
        "udp",  # proto
        "CON",  # state
        dur,
        "132",  # sbytes
        "164",  # dbytes
        "31",  # sttl
        "29",  # dttl
        "0",  # sloss
        "0",  # dloss
        service,
        "500473.9",  # sload
        "621800.9",  # dload
        "2",  # spkts
        "2",  # dpkts
        "0",  # swin
        "0",  # dwin
        "0",  # stcpb
        "0",  # dtcpb
        "66",  # smean
        "82",  # dmean
        "0",  # trans_depth
        "0",  # response_body_len
        "0",  # sjit
        "0",  # djit
        stime,
        stime,  # ltime
        "0.017",  # sinpkt
        "0.013",  # dinpkt
        "0",  # tcprtt
        "0",  # synack
        "0",  # ackdat
        "0",  # is_sm_ips_ports
        "3",  # ct_state_ttl
        "",  # ct_flw_http_mthd
        is_ftp_login,
        "",  # ct_ftp_cmd
        "7",  # ct_srv_src
        "1",  # ct_srv_dst
        "3",  # ct_dst_ltm
        "1",  # ct_src_ltm
        "1",  # ct_src_dport_ltm
        "1",  # ct_dst_sport_ltm
        "1",  # ct_dst_src_ltm
        attack_cat,
        label,
    ]


def _write_unsw_fixture(raw_dir: Path) -> None:
    # File 1 carries most of the data: 5 Benign + 5 Exploits rows (dur varies per row so
    # none collide as exact duplicates), enough per (source_file, attack_cat) group for the
    # chronological split's 80/20 cutoff to land on a non-degenerate split (4 train/1 test
    # each). Files 2-4 each add one more row (still with distinct dur/stime), one of them
    # exercising the "-" service sentinel and the is_ftp_login=2 data-entry bug, so the split
    # covers all 4 required raw files without every group being a same-sized clone of file 1.
    file1_rows = [
        _unsw_row(dur=f"0.{i}", service="dns", attack_cat="" if i < 5 else "Exploits", label="0" if i < 5 else "1", stime=str(1421927414 + i))
        for i in range(10)
    ]
    pd.DataFrame(file1_rows).to_csv(raw_dir / "UNSW-NB15_1.csv", header=False, index=False)

    pd.DataFrame([_unsw_row(dur="1.0", service="-", attack_cat="", label="0", stime="1421955795", is_ftp_login="2")]).to_csv(
        raw_dir / "UNSW-NB15_2.csv", header=False, index=False
    )
    pd.DataFrame([_unsw_row(dur="1.1", service="dns", attack_cat="DoS", label="1", stime="1424231099")]).to_csv(
        raw_dir / "UNSW-NB15_3.csv", header=False, index=False
    )
    pd.DataFrame([_unsw_row(dur="1.2", service="dns", attack_cat="", label="0", stime="1424250100")]).to_csv(
        raw_dir / "UNSW-NB15_4.csv", header=False, index=False
    )


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
                "drop_columns": ["srcip", "sport", "dstip", "dsport", "ltime", "label", "ct_ftp_cmd"],
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

    assert "srcip" not in train_df.columns
    assert "sport" not in train_df.columns
    assert "ltime" not in train_df.columns
    assert "label" not in train_df.columns
    assert "ct_ftp_cmd" not in train_df.columns
    assert "stime" not in train_df.columns  # dropped by the split itself, only used to order it
    assert "source_file" not in train_df.columns
    # 5 Benign + 5 Exploits in file 1 (80/20 cutoff -> 4 train/1 test each) + one more row per
    # file in files 2-4 (each its own size-1 group -> 0 train/1 test).
    assert len(train_df) == 8
    assert len(test_df) == 5
    assert set(train_df["attack_cat"]) | set(test_df["attack_cat"]) == {"Benign", "Exploits", "DoS"}
    assert "unknown" in (set(train_df["service"]) | set(test_df["service"]))  # "-" sentinel normalized
    assert pd.concat([train_df["is_ftp_login"], test_df["is_ftp_login"]]).max() == 1  # clipped from 2


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


def _write_toy_dataset_config(tmp_path) -> tuple[Path, Path]:
    """Shared toy processed dataset + config for the dataset-diagnostics plot tests."""
    processed_dir = tmp_path / "processed" / "toy"
    processed_dir.mkdir(parents=True)
    pd.DataFrame(
        {"feature_a": [0.0, 1.0, None], "feature_b": [1.0, 2.0, 3.0], "label": ["benign", "attack", "benign"]}
    ).to_csv(processed_dir / "train.csv", index=False)
    pd.DataFrame({"feature_a": [0.2], "feature_b": [4.0], "label": ["benign"]}).to_csv(
        processed_dir / "test.csv", index=False
    )

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
    return dataset_config_path, processed_dir


def test_generate_missing_value_summary_writes_png(tmp_path) -> None:
    dataset_config_path, _ = _write_toy_dataset_config(tmp_path)

    plots_dir = tmp_path / "plots"
    output_path = generate_missing_value_summary(dataset_config_path, plots_dir=plots_dir)

    assert output_path == plots_dir / "toy" / "missing_values.png"
    assert output_path.exists()


def test_generate_train_test_split_comparison_writes_png(tmp_path) -> None:
    dataset_config_path, _ = _write_toy_dataset_config(tmp_path)

    plots_dir = tmp_path / "plots"
    output_path = generate_train_test_split_comparison(dataset_config_path, plots_dir=plots_dir)

    assert output_path == plots_dir / "toy" / "train_test_split.png"
    assert output_path.exists()


def test_generate_feature_correlation_writes_png(tmp_path) -> None:
    dataset_config_path, _ = _write_toy_dataset_config(tmp_path)

    plots_dir = tmp_path / "plots"
    output_path = generate_feature_correlation(dataset_config_path, plots_dir=plots_dir)

    assert output_path == plots_dir / "toy" / "feature_correlation.png"
    assert output_path.exists()


def test_generate_feature_correlation_subsamples_large_datasets(tmp_path) -> None:
    processed_dir = tmp_path / "processed" / "toy"
    processed_dir.mkdir(parents=True)
    rng = np.random.default_rng(42)
    pd.DataFrame(
        {
            "feature_a": rng.normal(size=1000),
            "feature_b": rng.normal(size=1000),
            "label": ["benign"] * 1000,
        }
    ).to_csv(processed_dir / "train.csv", index=False)
    pd.DataFrame({"feature_a": [0.0], "feature_b": [0.0], "label": ["benign"]}).to_csv(
        processed_dir / "test.csv", index=False
    )

    dataset_config_path = tmp_path / "toy.yaml"
    dataset_config_path.write_text(
        yaml.safe_dump(
            {"name": "toy", "raw_path": "unused", "processed_path": processed_dir.as_posix(), "label_column": "label"}
        ),
        encoding="utf-8",
    )

    plots_dir = tmp_path / "plots"
    output_path = generate_feature_correlation(dataset_config_path, plots_dir=plots_dir, sample_size=100)

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


def _write_svm_subsample_fixture(tmp_path: Path, *, dataset_name: str, model: str) -> Path:
    """A toy (dataset, model) experiment set up the same way `run_experiment` would see it for
    a real (model, dataset) pair possibly in `_FINAL_FIT_SUBSAMPLE` - the actual row counts don't
    matter here (that mechanics is covered by test_preprocessing.py's stratified_subsample
    tests); this only needs to exist so run_experiment can be exercised end to end."""
    dataset_config_dir = tmp_path / "datasets"
    dataset_config_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = tmp_path / "processed" / dataset_name
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

    (dataset_config_dir / f"{dataset_name}.yaml").write_text(
        yaml.safe_dump(
            {
                "name": dataset_name,
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
                "name": f"{dataset_name}_{model}_baseline",
                "dataset": dataset_name,
                "model": model,
                "random_state": 42,
                "model_params": {"max_iter": 5} if model == "svm" else {},
                "preprocessing": {"strategy": "default_tabular"},
                "metrics": ["accuracy"],
            }
        ),
        encoding="utf-8",
    )
    return experiment_path


def _run_and_capture_subsample_calls(tmp_path: Path, *, dataset_name: str, model: str) -> list[tuple[int, int | None]]:
    """Run one experiment with `stratified_subsample` spied on, in its own MonkeyPatch context
    (rather than the test's shared one) so each call gets a clean, unchained spy - a loop
    reusing one MonkeyPatch across iterations would let a later spy wrap an earlier one instead
    of the real function, double-counting calls via Python's late-binding closures."""
    import pytest

    from xai_thesis import pipeline as pipeline_module

    calls: list[tuple[int, int | None]] = []
    with pytest.MonkeyPatch.context() as mp:
        real_stratified_subsample = pipeline_module.stratified_subsample

        def spy(*args, **kwargs):
            calls.append((args[2], kwargs.get("min_per_class")))
            return real_stratified_subsample(*args, **kwargs)

        mp.setattr(pipeline_module, "stratified_subsample", spy)
        experiment_path = _write_svm_subsample_fixture(tmp_path, dataset_name=dataset_name, model=model)
        run_experiment(
            experiment_path,
            dataset_config_dir=tmp_path / "datasets",
            models_dir=tmp_path / "models",
            metrics_dir=tmp_path / "metrics",
        )
    return calls


def test_run_experiment_subsamples_final_fit_for_svm_on_the_two_large_datasets(tmp_path) -> None:
    for dataset_name in ["unsw_nb15", "cic_ids2017"]:
        calls = _run_and_capture_subsample_calls(tmp_path / dataset_name, dataset_name=dataset_name, model="svm")
        assert calls == [(150_000, 200)]


def test_run_experiment_does_not_subsample_other_model_dataset_combinations(tmp_path, monkeypatch) -> None:
    from xai_thesis import pipeline as pipeline_module

    # svm on a dataset not in _FINAL_FIT_SUBSAMPLE, and gradient_boosting (not svm) on one that
    # is - neither should trigger a subsample of the final training set.
    for dataset_name, model in [("nsl_kdd", "svm"), ("unsw_nb15", "gradient_boosting")]:
        calls = []
        monkeypatch.setattr(pipeline_module, "stratified_subsample", lambda *a, **k: calls.append(1))
        experiment_path = _write_svm_subsample_fixture(
            tmp_path / f"{dataset_name}_{model}", dataset_name=dataset_name, model=model
        )

        run_experiment(
            experiment_path,
            dataset_config_dir=tmp_path / f"{dataset_name}_{model}" / "datasets",
            models_dir=tmp_path / f"{dataset_name}_{model}" / "models",
            metrics_dir=tmp_path / f"{dataset_name}_{model}" / "metrics",
        )

        assert calls == []


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

    assert output_path == plots_dir / "toy" / "experiment" / "confusion_matrix.png"
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
