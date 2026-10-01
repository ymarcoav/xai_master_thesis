import numpy as np
import pandas as pd

from xai_thesis.preprocessing import (
    build_preprocessing_pipeline,
    split_features_target,
    stratified_random_split,
    stratified_subsample,
)


def test_split_features_target_separates_label_column() -> None:
    data = pd.DataFrame({"feature": [1, 2], "label": ["benign", "attack"]})

    features, target = split_features_target(data, "label")

    assert list(features.columns) == ["feature"]
    assert target.tolist() == ["benign", "attack"]


def test_stratified_random_split_preserves_class_proportions_and_ratio() -> None:
    data = pd.DataFrame(
        {
            "feature": list(range(100)),
            "label": ["benign"] * 80 + ["attack"] * 20,
            "source_file": ["monday"] * 100,
        }
    )

    train, test = stratified_random_split(data, "label", drop_columns=("source_file",))

    assert len(train) == 70
    assert len(test) == 30
    # the 80/20 class balance is held on both sides of the split
    assert train["label"].value_counts().to_dict() == {"benign": 56, "attack": 14}
    assert test["label"].value_counts().to_dict() == {"benign": 24, "attack": 6}
    assert "source_file" not in train.columns
    assert "source_file" not in test.columns


def test_stratified_random_split_is_deterministic_for_fixed_seed() -> None:
    data = pd.DataFrame({"feature": list(range(50)), "label": ["a", "b"] * 25})

    first_train, first_test = stratified_random_split(data, "label")
    second_train, second_test = stratified_random_split(data, "label")

    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_test, second_test)


def test_stratified_random_split_ignores_absent_drop_columns() -> None:
    data = pd.DataFrame({"feature": list(range(20)), "label": ["a", "b"] * 10})

    # "stime"/"Timestamp" aren't in this frame - listing them must be a no-op, not an error.
    train, test = stratified_random_split(data, "label", drop_columns=("source_file", "stime", "Timestamp"))

    assert list(train.columns) == ["feature", "label"]
    assert len(train) + len(test) == 20


def test_stratified_subsample_keeps_a_floor_for_rare_classes() -> None:
    # A stratified 10%-ish subsample would proportionally reduce a 10-row rare class to
    # ~1 row, too few for a downstream 3-fold split; min_per_class should override that.
    features = pd.DataFrame({"feature": list(range(1000))})
    target = pd.Series(["common"] * 990 + ["rare"] * 10, name="label")

    x_sampled, y_sampled = stratified_subsample(features, target, target_size=100, min_per_class=3)

    counts = y_sampled.value_counts()
    assert counts["rare"] == 3
    assert counts["common"] == 99
    assert len(x_sampled) == len(y_sampled)


def test_stratified_subsample_keeps_every_row_of_a_class_smaller_than_the_floor() -> None:
    features = pd.DataFrame({"feature": list(range(1002))})
    target = pd.Series(["common"] * 1000 + ["tiny"] * 2, name="label")

    x_sampled, y_sampled = stratified_subsample(features, target, target_size=100, min_per_class=5)

    assert y_sampled.value_counts()["tiny"] == 2  # only 2 exist, floor of 5 can't exceed that


def test_build_preprocessing_pipeline_imputes_and_scales_numeric_columns() -> None:
    features = pd.DataFrame({"dur": [1.0, 2.0, np.nan, 400.0]})

    pipeline = build_preprocessing_pipeline(features, categorical_columns=[])
    transformed = pipeline.fit_transform(features)

    assert not np.isnan(transformed).any()
    assert transformed.shape == (4, 1)
    assert transformed.min() == 0.0  # min-max scaled: smallest value maps to 0
    assert transformed.max() == 1.0  # and the largest to 1


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


def test_build_preprocessing_pipeline_unseen_category_is_all_zeros_when_no_infrequent_bucket() -> None:
    # train has only frequent categories, so no "infrequent" bucket exists; an unseen test
    # category then has nowhere to go and encodes as an all-zero vector.
    train = pd.DataFrame({"proto": ["tcp", "udp"]})
    test = pd.DataFrame({"proto": ["tcp", "icmp"]})

    pipeline = build_preprocessing_pipeline(train, categorical_columns=["proto"])
    pipeline.fit(train)

    transformed = np.asarray(pipeline.transform(test))

    assert transformed.shape == (2, 2)
    assert transformed[1].tolist() == [0.0, 0.0]  # icmp: no active category


def test_build_preprocessing_pipeline_unseen_category_folds_into_infrequent_bucket() -> None:
    # handle_unknown="infrequent_if_exist": when the column DOES have an infrequent bucket
    # (from rare training categories), a category seen only at transform time is routed to it
    # rather than dropped to all-zeros - a rare unseen value behaves like a rare seen one.
    train = pd.DataFrame({"proto": ["tcp"] * 97 + ["rare_a", "rare_b", "rare_c"]})
    test = pd.DataFrame({"proto": ["tcp", "never_seen_in_train"]})

    pipeline = build_preprocessing_pipeline(train, categorical_columns=["proto"], min_frequency=0.05)
    pipeline.fit(train)
    encoder = pipeline.named_transformers_["categorical"].named_steps["encoder"]

    transformed = np.asarray(pipeline.transform(test))
    infrequent_index = list(encoder.get_feature_names_out()).index("proto_infrequent_sklearn")

    assert transformed[0].tolist() == [1.0, 0.0]  # tcp
    assert transformed[1][infrequent_index] == 1.0  # unseen -> infrequent bucket
    assert transformed[1].sum() == 1.0
