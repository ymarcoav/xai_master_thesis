import pandas as pd

from xai_thesis.datasets.base import DatasetConfig
from xai_thesis.datasets.unsw_nb15 import UNSWNB15Dataset


def test_unsw_clean_drops_identifier_and_binary_label() -> None:
    config = DatasetConfig(
        name="unsw_nb15",
        raw_path="unused",
        processed_path="unused",
        label_column="attack_cat",
        binary_label_column="label",
        drop_columns=("id", "label"),
    )
    dataset = UNSWNB15Dataset(config)
    data = pd.DataFrame(
        {
            "id": [1],
            "dur": [0.1],
            "label": [1],
            "attack_cat": ["Exploits"],
        }
    )

    cleaned = dataset.clean(data)

    assert list(cleaned.columns) == ["dur", "attack_cat"]
