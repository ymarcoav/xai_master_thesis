import pandas as pd
import pytest

from xai_thesis.datasets.base import DatasetConfig
from xai_thesis.datasets.nsl_kdd import NSLKDDDataset


def _make_dataset() -> NSLKDDDataset:
    config = DatasetConfig(
        name="nsl_kdd",
        raw_path="unused",
        processed_path="unused",
        label_column="attack_category",
        drop_columns=("difficulty_level",),
    )
    return NSLKDDDataset(config)


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "duration": [0, 0, 0],
            "protocol_type": ["tcp", "tcp", "udp"],
            "class": ["normal", "neptune", "buffer_overflow"],
            "difficulty_level": [20, 19, 21],
            "split": ["train", "train", "test"],
        }
    )


def test_nsl_kdd_clean_drops_difficulty_level_and_native_class_column() -> None:
    dataset = _make_dataset()
    data = _base_frame()

    cleaned = dataset.clean(data)

    assert "difficulty_level" not in cleaned.columns
    assert "class" not in cleaned.columns
    assert "attack_category" in cleaned.columns


def test_nsl_kdd_clean_maps_normal_and_specific_attack_types_to_categories() -> None:
    dataset = _make_dataset()
    data = _base_frame()

    cleaned = dataset.clean(data)

    assert cleaned["attack_category"].tolist() == ["Benign", "DoS", "U2R"]


def test_nsl_kdd_clean_maps_a_probe_and_r2l_example() -> None:
    dataset = _make_dataset()
    data = pd.DataFrame(
        {
            "duration": [0, 0],
            "class": ["portsweep", "guess_passwd"],
            "difficulty_level": [10, 10],
            "split": ["train", "train"],
        }
    )

    cleaned = dataset.clean(data)

    assert cleaned["attack_category"].tolist() == ["Probe", "R2L"]


def test_nsl_kdd_clean_raises_on_unrecognized_class_value() -> None:
    dataset = _make_dataset()
    data = pd.DataFrame(
        {
            "duration": [0],
            "class": ["totally_unknown_attack"],
            "difficulty_level": [10],
            "split": ["train"],
        }
    )

    with pytest.raises(KeyError):
        dataset.clean(data)


def test_nsl_kdd_clean_drops_exact_duplicate_rows() -> None:
    dataset = _make_dataset()
    data = _base_frame()
    data.loc[1] = data.loc[0]  # exact duplicate of the first row, including split

    cleaned = dataset.clean(data)

    assert len(cleaned) == 2
