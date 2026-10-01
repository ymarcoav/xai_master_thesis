import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from xai_thesis.pipeline import (
    _class_balanced_sample,
    generate_confusion_matrix,
    generate_explanations,
    generate_feature_correlation,
    generate_interview_stimuli,
    generate_label_distribution,
    generate_missing_value_summary,
    generate_roc_curves,
    generate_train_test_split_comparison,
    preprocess_dataset,
    run_experiment,
    tune_hyperparameters,
)


def test_class_balanced_sample_spreads_evenly_and_caps_at_class_size() -> None:
    features = pd.DataFrame({"f": range(1000)})
    target = pd.Series(["benign"] * 950 + ["rare_a"] * 40 + ["rare_b"] * 10, name="label")

    sample = _class_balanced_sample(features, target, total=90, random_state=42)
    counts = target.loc[sample.index].value_counts()

    # even share is 90 // 3 = 30 per class; rare_b only has 10 rows, so it contributes all 10
    assert counts["benign"] == 30
    assert counts["rare_a"] == 30
    assert counts["rare_b"] == 10
    # rows are a real subset of the input, drawn from the right classes
    assert set(sample["f"]) <= set(features["f"])


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
    # 20 rows across the 4 required raw part files, after dedup: 8 Benign + 8 Exploits + 4 DoS
    # (dur/stime vary per row so none collide as exact duplicates). The per-class counts are
    # chosen so the stratified 70/30 split lands every class in both partitions (DoS's
    # continuous test allocation is 4 * 0.3 = 1.2 >= 1). File 2 exercises the "-" service
    # sentinel and the is_ftp_login=2 data-entry bug; files 2-4 also keep the multi-file
    # load_raw concat under test.
    file1_rows = [
        _unsw_row(
            dur=f"0.{i}",
            service="dns",
            attack_cat="" if i < 7 else "Exploits",
            label="0" if i < 7 else "1",
            stime=str(1421927414 + i),
        )
        for i in range(15)
    ]
    pd.DataFrame(file1_rows).to_csv(raw_dir / "UNSW-NB15_1.csv", header=False, index=False)

    pd.DataFrame([_unsw_row(dur="1.0", service="-", attack_cat="", label="0", stime="1421955795", is_ftp_login="2")]).to_csv(
        raw_dir / "UNSW-NB15_2.csv", header=False, index=False
    )
    pd.DataFrame(
        [_unsw_row(dur=f"1.{i}", service="dns", attack_cat="DoS", label="1", stime=str(1424231099 + i)) for i in range(3)]
    ).to_csv(raw_dir / "UNSW-NB15_3.csv", header=False, index=False)
    pd.DataFrame([_unsw_row(dur="2.0", service="dns", attack_cat="DoS", label="1", stime="1424250100")]).to_csv(
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
                "drop_columns": ["srcip", "sport", "dstip", "dsport", "ltime", "label", "ct_ftp_cmd", "stime"],
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
    assert "stime" not in train_df.columns  # not a model feature - dropped in clean() via drop_columns
    assert "source_file" not in train_df.columns
    # 20 rows after dedup (8 Benign + 8 Exploits + 4 DoS), partitioned 70/30 by the stratified
    # random split; every class lands in both partitions.
    assert len(train_df) == 14
    assert len(test_df) == 6
    assert set(train_df["attack_cat"]) == {"Benign", "Exploits", "DoS"}
    assert set(test_df["attack_cat"]) == {"Benign", "Exploits", "DoS"}
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

    assert output_path == plots_dir / "toy" / "data" / "label_distribution.png"
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

    assert output_path == plots_dir / "toy" / "data" / "missing_values.png"
    assert output_path.exists()


def test_generate_train_test_split_comparison_writes_png(tmp_path) -> None:
    dataset_config_path, _ = _write_toy_dataset_config(tmp_path)

    plots_dir = tmp_path / "plots"
    output_path = generate_train_test_split_comparison(dataset_config_path, plots_dir=plots_dir)

    assert output_path == plots_dir / "toy" / "data" / "train_test_split.png"
    assert output_path.exists()


def test_generate_feature_correlation_writes_png(tmp_path) -> None:
    dataset_config_path, _ = _write_toy_dataset_config(tmp_path)

    plots_dir = tmp_path / "plots"
    output_path = generate_feature_correlation(dataset_config_path, plots_dir=plots_dir)

    assert output_path == plots_dir / "toy" / "data" / "feature_correlation.png"
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


def test_run_experiment_trains_xgboost_and_passes_balanced_sample_weight(tmp_path) -> None:
    # xgboost has no class_weight constructor param (see models.py) - pipeline.py must compute
    # and forward sample_weight instead (_MODELS_REQUIRING_SAMPLE_WEIGHT). Spies on
    # compute_sample_weight rather than re-deriving the expected array, so this test breaks if
    # the call is ever dropped without depending on the exact weighting formula.
    from xai_thesis import pipeline as pipeline_module

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
            {"name": "toy", "raw_path": "unused", "processed_path": processed_dir.as_posix(), "label_column": "label"}
        ),
        encoding="utf-8",
    )
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        yaml.safe_dump(
            {
                "name": "toy_xgboost_experiment",
                "dataset": "toy",
                "model": "xgboost",
                "random_state": 42,
                "model_params": {"n_estimators": 5, "max_depth": 2},
                "preprocessing": {"strategy": "default_tabular"},
                "metrics": ["accuracy"],
            }
        ),
        encoding="utf-8",
    )

    calls = []
    real_compute_sample_weight = pipeline_module.compute_sample_weight

    def spy(*args, **kwargs):
        result = real_compute_sample_weight(*args, **kwargs)
        calls.append(result)
        return result

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pipeline_module, "compute_sample_weight", spy)
        run_experiment(
            experiment_path,
            dataset_config_dir=dataset_config_dir,
            models_dir=tmp_path / "models",
            metrics_dir=tmp_path / "metrics",
        )

    assert len(calls) == 1
    assert len(calls[0]) == len(train_df)
    assert (tmp_path / "models" / "toy_xgboost_experiment.joblib").exists()


def _write_svm_subsample_fixture(tmp_path: Path, *, dataset_name: str, model: str) -> Path:
    """A toy (dataset, model) experiment set up the same way `run_experiment` would see it for
    a real (model, dataset) pair possibly in `_FINAL_FIT_SUBSAMPLE` - the actual row counts
    don't matter here (that mechanics is covered by test_preprocessing.py's
    stratified_subsample tests); this only needs to exist so run_experiment can be exercised
    end to end."""
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


def test_run_experiment_subsamples_final_fit_for_svm_on_unsw_and_cic(tmp_path) -> None:
    for dataset_name in ["unsw_nb15", "cic_ids2017"]:
        calls = _run_and_capture_subsample_calls(tmp_path / dataset_name, dataset_name=dataset_name, model="svm")
        assert calls == [(200_000, 500)]


def test_run_experiment_does_not_subsample_other_model_dataset_combinations(tmp_path, monkeypatch) -> None:
    from xai_thesis import pipeline as pipeline_module

    # svm on a dataset not in _FINAL_FIT_SUBSAMPLE, and random_forest (not svm) on one that
    # is - neither should trigger a subsample of the final training set.
    for dataset_name, model in [("toy_ids", "svm"), ("unsw_nb15", "random_forest")]:
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


def test_run_experiment_neural_network_trains_through_the_plain_pipeline_path(tmp_path) -> None:
    # neural_network is now TorchMLPClassifier and handles imbalance via a weighted loss, so
    # it takes the same _build_pipeline + train_model path as random_forest - no resampling,
    # so preprocessing fits on the real training rows and the pipeline is a normal fitted
    # Pipeline (not one reassembled from separately-fitted parts).
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

    metrics = run_experiment(
        experiment_path, dataset_config_dir=dataset_config_dir, models_dir=models_dir, metrics_dir=metrics_dir
    )

    assert set(metrics) == {"accuracy"}
    pipeline = joblib.load(models_dir / "toy_experiment.joblib")
    # imputer fits on the real training rows (median of [1]*8 + [100]*2 is 1.0)
    imputer = pipeline.named_steps["preprocessing"].named_transformers_["numeric"].named_steps["imputer"]
    assert imputer.statistics_[0] == 1.0
    assert len(pipeline.predict(test_df[["feature"]])) == 2


def test_generate_confusion_matrix_and_roc_curves_write_pngs_for_trained_experiment(tmp_path) -> None:
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
    cm_path = generate_confusion_matrix(
        experiment_path, dataset_config_dir=dataset_config_dir, models_dir=models_dir, plots_dir=plots_dir
    )
    roc_path = generate_roc_curves(
        experiment_path, dataset_config_dir=dataset_config_dir, models_dir=models_dir, plots_dir=plots_dir
    )

    assert cm_path == plots_dir / "toy" / "experiment" / "performance" / "confusion_matrix.png"
    assert cm_path.exists()
    assert roc_path == plots_dir / "toy" / "experiment" / "performance" / "roc_curves.png"
    assert roc_path.exists()


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
                "model": "random_forest",
                "random_state": 42,
                "preprocessing": {"strategy": "default_tabular"},
                "metrics": ["accuracy"],
            }
        ),
        encoding="utf-8",
    )

    tuned_params = tune_hyperparameters(experiment_path, dataset_config_dir=dataset_config_dir, n_iter=3, cv=3)

    assert set(tuned_params) == {
        "n_estimators",
        "max_depth",
        "max_features",
        "min_samples_leaf",
        "min_samples_split",
    }

    updated_experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    assert updated_experiment["model_params"] == tuned_params


def test_tune_hyperparameters_neural_network_round_trip_retrains_successfully(tmp_path) -> None:
    # TorchMLPClassifier must be cloneable by RandomizedSearchCV and accept the
    # hidden_layer_sizes / alpha / learning_rate_init grid; run_experiment must then retrain
    # from the tuned params. No resampling special-case in the search anymore.
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
                "model": "neural_network",
                "random_state": 42,
                "model_params": {"max_iter": 5},
                "preprocessing": {"strategy": "default_tabular"},
                "metrics": ["accuracy"],
            }
        ),
        encoding="utf-8",
    )

    tuned_params = tune_hyperparameters(experiment_path, dataset_config_dir=dataset_config_dir, n_iter=3, cv=3)

    assert set(tuned_params) == {"hidden_layer_sizes", "alpha", "learning_rate_init", "batch_size"}
    run_experiment(
        experiment_path,
        dataset_config_dir=dataset_config_dir,
        models_dir=tmp_path / "models",
        metrics_dir=tmp_path / "metrics",
    )
    assert (tmp_path / "models" / "toy_experiment.joblib").exists()


def test_tune_hyperparameters_svm_round_trip_retrains_successfully(tmp_path) -> None:
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
                "model": "svm",
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
                "model": "random_forest",
                "preprocessing": {"strategy": "fancy"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(NotImplementedError):
        run_experiment(experiment_path)


def test_generate_explanations_writes_all_expected_plots(tmp_path) -> None:
    dataset_config_dir = tmp_path / "datasets"
    dataset_config_dir.mkdir()
    processed_dir = tmp_path / "processed" / "toy_ids"
    processed_dir.mkdir(parents=True)

    # Reconnaissance/Benign overlap deliberately, so the model is not trivially perfect.
    # protocol_type is categorical and strongly predictive, so it ranks high in permutation
    # importance.
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

    (dataset_config_dir / "toy_ids.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "toy_ids",
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
                "name": "toy_ids_test_experiment",
                "dataset": "toy_ids",
                "model": "random_forest",
                "random_state": 42,
                "model_params": {"n_estimators": 10},
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
        experiment_path,
        dataset_config_dir=dataset_config_dir,
        models_dir=models_dir,
        plots_dir=plots_dir,
        metrics_dir=metrics_dir,
    )

    assert set(paths) == {"shap_bar", "permutation_importance"}
    for path in paths.values():
        assert path.exists()
    # Local SHAP/LIME plots are interview stimuli only - generate_explanations writes none.
    experiment_dir = plots_dir / "toy_ids" / "test_experiment"
    assert all(path.parent == experiment_dir / "global" for path in paths.values())
    assert not list(experiment_dir.rglob("shap_waterfall_*.png"))
    assert not list(experiment_dir.rglob("lime_*.png"))


def _write_toy_dataset_fixture(tmp_path) -> tuple[Path, Path, Path]:
    """Overlapping-classes construction like test_generate_explanations_writes_all_expected_plots
    (protocol_type is the dominant, near-deterministic predictor of Reconnaissance-vs-Benign),
    but with two differences that matter here: stage_train is dtype=object (np.where's dtype is
    inferred from "Reconnaissance"/"Benign" and silently truncates the longer "Establish
    Foothold" noise label otherwise) and test row 1 deliberately mismatches its protocol_type
    against its true label (protocol_type="udp", Stage="Reconnaissance") so a genuine
    misclassification is guaranteed by construction rather than left to chance - the model
    reliably predicts "Benign" from protocol_type="udp", giving generate_interview_stimuli() a
    real (correct, misclassified) pair for the "Reconnaissance" target class."""
    dataset_config_dir = tmp_path / "datasets"
    dataset_config_dir.mkdir()
    processed_dir = tmp_path / "processed" / "toy_ids"
    processed_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    n = 60
    protocol_train = rng.choice(["tcp", "udp"], n)
    stage_train = np.where(protocol_train == "tcp", "Reconnaissance", "Benign").astype(object)
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
    protocol_test[0] = "tcp"
    stage_test = np.where(protocol_test == "tcp", "Reconnaissance", "Benign").astype(object)
    protocol_test[1] = "udp"
    stage_test[1] = "Reconnaissance"
    test_df = pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1.5, 20),
            "feature_b": rng.normal(0, 1.5, 20),
            "protocol_type": protocol_test,
            "Stage": stage_test,
        }
    )
    train_df.to_csv(processed_dir / "train.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    (dataset_config_dir / "toy_ids.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "toy_ids",
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
                "name": "toy_ids_test_experiment",
                "dataset": "toy_ids",
                "model": "random_forest",
                "random_state": 42,
                "model_params": {"n_estimators": 10},
                "preprocessing": {"strategy": "default_tabular"},
                "metrics": ["accuracy"],
            }
        ),
        encoding="utf-8",
    )
    return dataset_config_dir, experiment_path, tmp_path / "models"


def test_generate_interview_stimuli_writes_plots_for_configured_classes(tmp_path, monkeypatch) -> None:
    from xai_thesis import pipeline as pipeline_module

    dataset_config_dir, experiment_path, models_dir = _write_toy_dataset_fixture(tmp_path)
    metrics_dir = tmp_path / "metrics"
    plots_dir = tmp_path / "plots"

    run_experiment(
        experiment_path, dataset_config_dir=dataset_config_dir, models_dir=models_dir, metrics_dir=metrics_dir
    )
    monkeypatch.setattr(pipeline_module, "_INTERVIEW_STIMULUS_CLASSES", {"toy_ids": ["Reconnaissance"]})

    results = generate_interview_stimuli(
        experiment_path, dataset_config_dir=dataset_config_dir, models_dir=models_dir, plots_dir=plots_dir
    )

    assert set(results) == {"Reconnaissance"}
    paths = results["Reconnaissance"]
    assert {"shap_waterfall_correct", "lime_correct", "shap_waterfall_misclassified", "lime_misclassified"} <= set(
        paths
    )
    for path in paths.values():
        assert path.exists()
    assert paths["lime_correct"].parent == plots_dir / "toy_ids" / "test_experiment" / "interview" / "Reconnaissance"


def test_generate_interview_stimuli_skips_class_with_no_misclassifications(tmp_path, monkeypatch) -> None:
    from xai_thesis import pipeline as pipeline_module

    dataset_config_dir, experiment_path, models_dir = _write_toy_dataset_fixture(tmp_path)
    metrics_dir = tmp_path / "metrics"
    plots_dir = tmp_path / "plots"

    run_experiment(
        experiment_path, dataset_config_dir=dataset_config_dir, models_dir=models_dir, metrics_dir=metrics_dir
    )
    # "Establish Foothold" never appears in the test set at all, so the model can never
    # misclassify it there - exercises the "no stimulus pair possible" path rather than erroring.
    monkeypatch.setattr(pipeline_module, "_INTERVIEW_STIMULUS_CLASSES", {"toy_ids": ["Establish Foothold"]})

    results = generate_interview_stimuli(
        experiment_path, dataset_config_dir=dataset_config_dir, models_dir=models_dir, plots_dir=plots_dir
    )

    assert results == {"Establish Foothold": {}}


def test_generate_interview_stimuli_raises_for_unconfigured_dataset(tmp_path) -> None:
    dataset_config_dir, experiment_path, models_dir = _write_toy_dataset_fixture(tmp_path)
    metrics_dir = tmp_path / "metrics"

    run_experiment(
        experiment_path, dataset_config_dir=dataset_config_dir, models_dir=models_dir, metrics_dir=metrics_dir
    )

    with pytest.raises(NotImplementedError):
        generate_interview_stimuli(experiment_path, dataset_config_dir=dataset_config_dir, models_dir=models_dir)


def test_as_importance_vector_clips_negative_permutation_importances() -> None:
    from xai_thesis.pipeline import _as_importance_vector

    # A feature whose permutation improved the score scores slightly negative; that means "no
    # measurable importance", so it must not outrank a small positive one on |value|.
    clipped = _as_importance_vector(np.array([-0.004, 0.001, 0.02]))

    assert clipped.tolist() == [0.0, 0.001, 0.02]


def test_top_features_orders_by_importance_and_ignores_negatives() -> None:
    from xai_thesis.pipeline import top_features

    ranking = np.array([0.1, -0.9, 0.5, 0.3])
    names = ["a", "b", "c", "d"]

    assert top_features(ranking, names, k=3) == ["c", "d", "a"]


def test_global_ranking_sample_balanced_is_even_across_classes(tmp_path) -> None:
    from xai_thesis.pipeline import _global_ranking_sample

    y = pd.Series(["benign"] * 40 + ["attack"] * 10 + ["rare"] * 3)
    x = pd.DataFrame({"f": np.arange(len(y), dtype=float)})

    sample_x, sample_y = _global_ranking_sample(x, y, 12, basis="balanced", random_state=42)

    counts = sample_y.value_counts()
    # 12 // 3 classes = 4 each, except "rare", which contributes all 3 rows it has.
    assert counts["benign"] == 4
    assert counts["attack"] == 4
    assert counts["rare"] == 3
    assert list(sample_x.index) == list(sample_y.index)


def test_global_ranking_sample_stratified_follows_prevalence() -> None:
    from xai_thesis.pipeline import _global_ranking_sample

    y = pd.Series(["benign"] * 90 + ["attack"] * 10)
    x = pd.DataFrame({"f": np.arange(len(y), dtype=float)})

    _, sample_y = _global_ranking_sample(x, y, 50, basis="stratified", random_state=42)

    # Prevalence-proportional: ~45 benign / ~5 attack, i.e. the imbalance the balanced basis
    # exists to avoid.
    assert sample_y.value_counts()["benign"] == 45
    assert sample_y.value_counts()["attack"] == 5


def test_global_ranking_sample_rejects_unknown_basis() -> None:
    from xai_thesis.pipeline import _global_ranking_sample

    y = pd.Series(["a", "b"])
    x = pd.DataFrame({"f": [0.0, 1.0]})

    with pytest.raises(ValueError, match="unknown basis"):
        _global_ranking_sample(x, y, 2, basis="kmeans", random_state=42)


def test_only_cic_svm_global_is_excluded_by_default() -> None:
    """Every local cell is computed; the one global cell not computed is cic_ids2017/svm.

    Pins both lists so a cell cannot be dropped, or the omission lose its stated reason, without
    a test failing. The global exclusion must carry the measured cost, since it is printed in the
    panel's footnote.
    """
    from xai_thesis.pipeline import _AGREEMENT_EXCLUDED, _GLOBAL_AGREEMENT_EXCLUDED

    assert _AGREEMENT_EXCLUDED == {}
    assert list(_GLOBAL_AGREEMENT_EXCLUDED) == [("cic_ids2017", "svm")]
    assert "53 h" in _GLOBAL_AGREEMENT_EXCLUDED[("cic_ids2017", "svm")]


def test_partition_excluded_uses_the_exclusion_list_it_is_given() -> None:
    from xai_thesis.pipeline import _GLOBAL_AGREEMENT_EXCLUDED, _partition_excluded

    experiments = [("random_forest", Path("rf.yaml")), ("svm", Path("svm.yaml"))]

    included, notes = _partition_excluded(
        "cic_ids2017", experiments, excluded=_GLOBAL_AGREEMENT_EXCLUDED
    )
    assert [model for model, _ in included] == ["random_forest"]
    assert notes and notes[0].startswith("svm omitted:")

    # The local panel's default list is empty, so the same call without it keeps svm.
    included, notes = _partition_excluded("cic_ids2017", experiments)
    assert [model for model, _ in included] == ["random_forest", "svm"]
    assert notes == []


def test_partition_excluded_drops_configured_pairs_and_reports_why(monkeypatch) -> None:
    """The exclusion mechanism itself still works, for any future pair that needs it."""
    from xai_thesis import pipeline as pipeline_module
    from xai_thesis.pipeline import _partition_excluded

    monkeypatch.setattr(
        pipeline_module, "_AGREEMENT_EXCLUDED", {("toy", "svm"): "measured at 999 s/instance"}
    )
    experiments = [("random_forest", Path("rf.yaml")), ("svm", Path("svm.yaml"))]

    included, notes = _partition_excluded("toy", experiments)

    assert [model for model, _ in included] == ["random_forest"]
    assert notes == ["svm omitted: measured at 999 s/instance"]

    # A dataset with no exclusions keeps every model and reports nothing.
    included, notes = _partition_excluded("other", experiments)
    assert [model for model, _ in included] == ["random_forest", "svm"]
    assert notes == []


def test_partition_excluded_honours_an_explicit_model_filter() -> None:
    from xai_thesis.pipeline import _partition_excluded

    experiments = [("random_forest", Path("rf.yaml")), ("svm", Path("svm.yaml"))]

    included, notes = _partition_excluded("unsw_nb15", experiments, models=["random_forest"])

    assert [model for model, _ in included] == ["random_forest"]
    assert notes == ["svm not requested"]


def test_generate_global_agreement_writes_heatmaps_and_csv(tmp_path) -> None:
    from xai_thesis.pipeline import generate_global_agreement

    dataset_config_dir, experiment_path, models_dir = _write_toy_dataset_fixture(tmp_path)
    metrics_dir = tmp_path / "metrics"
    plots_dir = tmp_path / "plots"
    run_experiment(
        experiment_path, dataset_config_dir=dataset_config_dir, models_dir=models_dir, metrics_dir=metrics_dir
    )

    paths = generate_global_agreement(
        "toy_ids",
        config_dir=tmp_path,
        dataset_config_dir=dataset_config_dir,
        models_dir=models_dir,
        metrics_dir=metrics_dir,
        plots_dir=plots_dir,
        k_values=[1, 3],
        size=9,
        # The fixture's test split is far smaller than _GLOBAL_EXPLANATION_SAMPLE_SIZE, so this
        # run is technically a non-default basis and would be CSV-only; ask for the heatmaps
        # explicitly, since drawing them is what this test is about.
        write_plots=True,
    )

    assert set(paths) == {"feature", "rank", "csv"}
    for path in paths.values():
        assert path.exists()

    rows = list(csv.DictReader(paths["csv"].open(encoding="utf-8")))
    # One row per (metric, k) for the single model this fixture defines.
    assert len(rows) == 4
    assert {row["metric"] for row in rows} == {"feature", "rank"}
    assert {row["k"] for row in rows} == {"1", "3"}
    for row in rows:
        assert 0.0 <= float(row["value"]) <= 1.0
        assert row["basis"] == "balanced"
        assert row["shap_top5"] and row["permutation_top5"]
        # Rank agreement is the stricter metric and can never exceed feature agreement.
    by_key = {(row["metric"], row["k"]): float(row["value"]) for row in rows}
    for k in ("1", "3"):
        assert by_key[("rank", k)] <= by_key[("feature", k)]


def test_compute_global_rankings_reuses_its_cache(tmp_path) -> None:
    from xai_thesis.pipeline import compute_global_rankings

    dataset_config_dir, experiment_path, models_dir = _write_toy_dataset_fixture(tmp_path)
    metrics_dir = tmp_path / "metrics"
    run_experiment(
        experiment_path, dataset_config_dir=dataset_config_dir, models_dir=models_dir, metrics_dir=metrics_dir
    )

    kwargs = {
        "dataset_config_dir": dataset_config_dir,
        "models_dir": models_dir,
        "metrics_dir": metrics_dir,
        "size": 9,
    }
    first = compute_global_rankings(experiment_path, **kwargs)
    stamp = first.stat().st_mtime_ns
    second = compute_global_rankings(experiment_path, **kwargs)

    assert first == second
    assert second.stat().st_mtime_ns == stamp, "cached ranking should not be recomputed"

    # The two bases write to separate cache files, so one never silently serves the other.
    stratified = compute_global_rankings(experiment_path, basis="stratified", **kwargs)
    assert stratified != first
    assert "stratified9" in stratified.name


def test_generate_explanations_caches_the_global_rankings_it_plots(tmp_path) -> None:
    """generate_explanations() and generate_global_agreement() must describe one ranking.

    The plot path drops its two rankings into the global cache, so the agreement panel reads
    back exactly the numbers behind shap_bar.png / permutation_importance.png instead of
    recomputing them - which for svm would be ~12 h and could in principle drift.
    """
    from xai_thesis.pipeline import _global_cache_path, compute_global_rankings

    dataset_config_dir, experiment_path, models_dir = _write_toy_dataset_fixture(tmp_path)
    metrics_dir = tmp_path / "metrics"
    plots_dir = tmp_path / "plots"
    run_experiment(
        experiment_path, dataset_config_dir=dataset_config_dir, models_dir=models_dir, metrics_dir=metrics_dir
    )

    generate_explanations(
        experiment_path,
        dataset_config_dir=dataset_config_dir,
        models_dir=models_dir,
        metrics_dir=metrics_dir,
        plots_dir=plots_dir,
        sample_size=9,
    )

    cache_path = _global_cache_path(metrics_dir, "toy_ids_test_experiment", "balanced", 9)
    assert cache_path.exists(), "plotting the global explanations should populate the cache"
    stamp = cache_path.stat().st_mtime_ns

    reused = compute_global_rankings(
        experiment_path,
        dataset_config_dir=dataset_config_dir,
        models_dir=models_dir,
        metrics_dir=metrics_dir,
        size=9,
    )

    assert reused == cache_path
    assert reused.stat().st_mtime_ns == stamp, "agreement must reuse the plotted ranking"

    cached = np.load(cache_path, allow_pickle=False)
    assert cached["shap"].shape == cached["permutation"].shape == cached["feature_names"].shape
    assert (cached["shap"] >= 0).all(), "mean |SHAP| is a magnitude"


def test_global_agreement_on_a_non_default_basis_writes_csv_but_no_heatmap(tmp_path) -> None:
    """Only the default basis gets a heatmap.

    The alternative bases exist to justify the default, and are reported as numbers; emitting a
    near-identical heatmap for each would bury the one figure the thesis uses.
    """
    from xai_thesis.pipeline import generate_global_agreement

    dataset_config_dir, experiment_path, models_dir = _write_toy_dataset_fixture(tmp_path)
    metrics_dir = tmp_path / "metrics"
    plots_dir = tmp_path / "plots"
    run_experiment(
        experiment_path, dataset_config_dir=dataset_config_dir, models_dir=models_dir, metrics_dir=metrics_dir
    )
    shared = {
        "config_dir": tmp_path,
        "dataset_config_dir": dataset_config_dir,
        "models_dir": models_dir,
        "metrics_dir": metrics_dir,
        "plots_dir": plots_dir,
        "k_values": [1, 3],
    }

    alternative = generate_global_agreement("toy_ids", basis="stratified", size=8, **shared)
    assert set(alternative) == {"csv"}
    assert alternative["csv"].exists()
    assert not list(plots_dir.rglob("global_agreement*.png"))

    # ...but it can still be asked for explicitly.
    forced = generate_global_agreement(
        "toy_ids", basis="stratified", size=8, write_plots=True, **shared
    )
    assert set(forced) == {"feature", "rank", "csv"}
    assert all(path.exists() for path in forced.values())
