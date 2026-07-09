import numpy as np
import pandas as pd

from xai_thesis.preprocessing import build_preprocessing_pipeline, split_features_target


def test_split_features_target_separates_label_column() -> None:
    data = pd.DataFrame({"feature": [1, 2], "label": ["benign", "attack"]})

    features, target = split_features_target(data, "label")

    assert list(features.columns) == ["feature"]
    assert target.tolist() == ["benign", "attack"]


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
