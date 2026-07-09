import numpy as np
import pandas as pd

from xai_thesis.datasets.base import DatasetConfig
from xai_thesis.datasets.cic_ids2017 import CICIDS2017Dataset


def _make_dataset() -> CICIDS2017Dataset:
    config = DatasetConfig(
        name="cic_ids2017",
        raw_path="unused",
        processed_path="unused",
        label_column="Label",
    )
    return CICIDS2017Dataset(config)


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            " Destination Port": [80, 443, 22, 80],
            "Flow Duration": [100, 200, 300, 100],
            " Flow IAT Min": [1, 2, 3, 1],
            "Flow Bytes/s": [10.0, float("inf"), 5.0, 10.0],
            " Flow Packets/s": [1.0, 2.0, float("-inf"), 1.0],
            " Fwd Header Length": [20, 20, 20, 20],
            " Fwd Header Length.1": [20, 20, 20, 20],
            " Label": ["BENIGN", "BENIGN", "BENIGN", "BENIGN"],
        }
    )


def test_cic_ids2017_clean_strips_whitespace_and_drops_duplicate_column() -> None:
    dataset = _make_dataset()
    data = _base_frame()

    cleaned = dataset.clean(data)

    assert "Fwd Header Length.1" not in cleaned.columns
    assert "Fwd Header Length" in cleaned.columns
    assert "Label" in cleaned.columns
    assert all(not column.startswith(" ") for column in cleaned.columns)


def test_cic_ids2017_clean_converts_infinity_to_nan() -> None:
    dataset = _make_dataset()
    data = _base_frame()

    cleaned = dataset.clean(data)

    assert not np.isinf(cleaned["Flow Bytes/s"]).any()
    assert not np.isinf(cleaned["Flow Packets/s"]).any()


def test_cic_ids2017_clean_drops_rows_with_negative_duration_or_iat() -> None:
    dataset = _make_dataset()
    data = _base_frame()
    data.loc[0, "Flow Duration"] = -1
    data.loc[1, " Flow IAT Min"] = -5

    cleaned = dataset.clean(data)

    assert (cleaned["Flow Duration"] >= 0).all()
    assert (cleaned["Flow IAT Min"] >= 0).all()
    assert len(cleaned) == 2


def test_cic_ids2017_clean_repairs_corrupted_label_characters() -> None:
    dataset = _make_dataset()
    data = _base_frame()
    data[" Label"] = ["Web Attack � Brute Force", "BENIGN", "BENIGN", "BENIGN"]

    cleaned = dataset.clean(data)

    assert "Web Attack - Brute Force" in cleaned["Label"].tolist()
    assert "�" not in "".join(cleaned["Label"].tolist())


def test_cic_ids2017_clean_drops_exact_duplicate_rows() -> None:
    dataset = _make_dataset()
    data = _base_frame()  # rows 0 and 3 are exact duplicates; rows 1 and 2 are unique

    cleaned = dataset.clean(data)

    assert len(cleaned) == 3
