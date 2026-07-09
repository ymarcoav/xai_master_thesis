import numpy as np
import pandas as pd

from xai_thesis.preprocessing import (
    build_preprocessing_pipeline,
    split_by_official_split_column,
    split_chronologically_by_source,
    split_features_target,
)


def test_split_features_target_separates_label_column() -> None:
    data = pd.DataFrame({"feature": [1, 2], "label": ["benign", "attack"]})

    features, target = split_features_target(data, "label")

    assert list(features.columns) == ["feature"]
    assert target.tolist() == ["benign", "attack"]


def test_split_by_official_split_column_separates_train_and_test() -> None:
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "split": ["train", "train", "test", "test"],
        }
    )

    train, test = split_by_official_split_column(data)

    assert train["feature"].tolist() == [1, 2]
    assert test["feature"].tolist() == [3, 4]
    assert "split" not in train.columns
    assert "split" not in test.columns


def test_split_chronologically_by_source_keeps_every_source_in_both_splits() -> None:
    data = pd.DataFrame(
        {
            "feature": list(range(10)),
            "label": ["a"] * 5 + ["b"] * 5,
            "source_file": ["monday"] * 5 + ["friday"] * 5,
        }
    )

    train, test = split_chronologically_by_source(data, "source_file", "label", test_fraction=0.2)

    assert len(train) == 8
    assert len(test) == 2
    # each source file's last rows (by original order) go to test
    assert train["feature"].tolist() == [0, 1, 2, 3, 5, 6, 7, 8]
    assert test["feature"].tolist() == [4, 9]
    assert "source_file" not in train.columns
    assert "source_file" not in test.columns


def test_split_chronologically_by_source_keeps_class_in_both_splits_despite_time_clustering() -> None:
    # A rare attack class occurs in one contiguous burst near the end of a file's row
    # order; a per-source-file-only cutoff would drop it entirely from train.
    data = pd.DataFrame(
        {
            "feature": list(range(10)),
            "label": ["benign"] * 8 + ["rare_attack"] * 2,
            "source_file": ["tuesday"] * 10,
        }
    )

    train, test = split_chronologically_by_source(data, "source_file", "label", test_fraction=0.2)

    assert "rare_attack" in train["label"].tolist()
    assert "rare_attack" in test["label"].tolist()


def test_build_preprocessing_pipeline_imputes_and_scales_numeric_columns() -> None:
    features = pd.DataFrame({"dur": [1.0, 2.0, np.nan, 400.0]})

    pipeline = build_preprocessing_pipeline(features, categorical_columns=[])
    transformed = pipeline.fit_transform(features)

    assert not np.isnan(transformed).any()
    assert transformed.shape == (4, 1)


def test_build_preprocessing_pipeline_one_hot_encodes_categorical_columns() -> None:
    features = pd.DataFrame({"proto": ["tcp", "udp", "tcp", "udp"]})

    pipeline = build_preprocessing_pipeline(features, categorical_columns=["proto"])
    transformed = pipeline.fit_transform(features)

    assert transformed.shape == (4, 2)


def test_build_preprocessing_pipeline_buckets_rare_categories() -> None:
    features = pd.DataFrame({"proto": ["tcp"] * 97 + ["rare_a", "rare_b", "rare_c"]})

    pipeline = build_preprocessing_pipeline(features, categorical_columns=["proto"], min_frequency=0.05)
    transformed = pipeline.fit_transform(features)

    # tcp (97%) gets its own column; the three rare protocols (1% each) share one "infrequent" column.
    assert transformed.shape == (100, 2)


def test_build_preprocessing_pipeline_ignores_unseen_category_at_transform() -> None:
    train = pd.DataFrame({"proto": ["tcp", "udp"]})
    test = pd.DataFrame({"proto": ["tcp", "icmp"]})

    pipeline = build_preprocessing_pipeline(train, categorical_columns=["proto"])
    pipeline.fit(train)

    transformed = pipeline.transform(test)

    assert transformed.shape == (2, 2)
