import pandas as pd

from xai_thesis.preprocessing import split_features_target


def test_split_features_target_separates_label_column() -> None:
    data = pd.DataFrame({"feature": [1, 2], "label": ["benign", "attack"]})

    features, target = split_features_target(data, "label")

    assert list(features.columns) == ["feature"]
    assert target.tolist() == ["benign", "attack"]
