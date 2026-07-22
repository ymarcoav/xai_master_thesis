import pandas as pd

from xai_thesis.datasets.base import DatasetConfig
from xai_thesis.datasets.unsw_nb15 import UNSWNB15Dataset


def _make_dataset() -> UNSWNB15Dataset:
    config = DatasetConfig(
        name="unsw_nb15",
        raw_path="unused",
        processed_path="unused",
        label_column="label",
        drop_columns=("id", "attack_cat"),
    )
    return UNSWNB15Dataset(config)


def test_unsw_clean_drops_identifier_and_multiclass_label() -> None:
    dataset = _make_dataset()
    data = pd.DataFrame(
        {
            "id": [1],
            "dur": [0.1],
            "label": [1],
            "attack_cat": ["Exploits"],
            "service": ["http"],
            "is_ftp_login": [0],
            "split": ["train"],
        }
    )

    cleaned = dataset.clean(data)

    assert list(cleaned.columns) == ["dur", "label", "service", "is_ftp_login", "split"]


def test_unsw_clean_maps_binary_label_to_readable_strings() -> None:
    dataset = _make_dataset()
    data = pd.DataFrame(
        {
            "id": [1, 2],
            "dur": [0.1, 0.2],
            "label": [1, 0],
            "attack_cat": ["Exploits", "Normal"],
            "service": ["http", "http"],
            "is_ftp_login": [0, 0],
            "split": ["train", "train"],
        }
    )

    cleaned = dataset.clean(data)

    assert cleaned["label"].tolist() == ["Attack", "Benign"]


def test_unsw_clean_drops_duplicates_within_split_but_keeps_cross_split_overlap() -> None:
    dataset = _make_dataset()
    data = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "dur": [0.1, 0.1, 0.1, 0.5],
            "label": [1, 1, 1, 0],
            "attack_cat": ["Exploits", "Exploits", "Exploits", "Normal"],
            "service": ["-", "-", "-", "http"],
            "is_ftp_login": [2, 2, 2, 0],
            "split": ["train", "train", "test", "train"],
        }
    )

    cleaned = dataset.clean(data)

    assert len(cleaned) == 3
    assert cleaned["split"].value_counts().to_dict() == {"train": 2, "test": 1}


def test_unsw_clean_normalizes_service_sentinel_and_clips_ftp_login_flag() -> None:
    dataset = _make_dataset()
    data = pd.DataFrame(
        {
            "id": [1],
            "dur": [0.1],
            "label": [1],
            "attack_cat": ["Exploits"],
            "service": ["-"],
            "is_ftp_login": [2],
            "split": ["train"],
        }
    )

    cleaned = dataset.clean(data)

    assert cleaned["service"].tolist() == ["unknown"]
    assert cleaned["is_ftp_login"].tolist() == [1]
