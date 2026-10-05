import numpy as np
import pandas as pd
import pytest

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
            "Bwd Header Length": [20, 20, 20, 20],
            "min_seg_size_forward": [20, 20, 20, 20],
            " Label": ["BENIGN", "BENIGN", "BENIGN", "BENIGN"],
            "source_file": ["Monday", "Tuesday", "Wednesday", "Friday"],
        }
    )


def test_cic_ids2017_clean_drops_configured_drop_columns() -> None:
    config = DatasetConfig(
        name="cic_ids2017",
        raw_path="unused",
        processed_path="unused",
        label_column="Label",
        drop_columns=("Fwd Avg Bytes/Bulk", "Bwd PSH Flags"),
    )
    dataset = CICIDS2017Dataset(config)
    data = _base_frame()
    data["Fwd Avg Bytes/Bulk"] = 0
    data["Bwd PSH Flags"] = 0

    cleaned = dataset.clean(data)

    assert "Fwd Avg Bytes/Bulk" not in cleaned.columns
    assert "Bwd PSH Flags" not in cleaned.columns


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


def test_cic_ids2017_clean_drops_rows_with_negative_header_length_or_segment_size() -> None:
    # Negative header lengths or segment sizes are physically impossible; such rows are dropped.
    dataset = _make_dataset()
    data = _base_frame()
    data.loc[0, "Bwd Header Length"] = -1
    data.loc[1, "min_seg_size_forward"] = -5

    cleaned = dataset.clean(data)

    assert (cleaned["Bwd Header Length"] >= 0).all()
    assert (cleaned["min_seg_size_forward"] >= 0).all()
    assert len(cleaned) == 2


def test_cic_ids2017_clean_maps_labels_to_seven_categories() -> None:
    dataset = _make_dataset()
    data = _base_frame()
    data[" Label"] = ["Web Attack - Brute Force", "DDoS", "PortScan", "BENIGN"]

    cleaned = dataset.clean(data)

    labels = cleaned["Label"].tolist()
    assert labels == ["Web Attack", "DoS/DDoS", "PortScan", "Benign"]


def test_cic_ids2017_clean_maps_corrupted_web_attack_separator_byte() -> None:
    dataset = _make_dataset()
    data = _base_frame()
    # The real raw files use a corrupted non-ASCII separator byte here, not a plain hyphen -
    # confirmed via raw byte inspection (literal UTF-8 encoding of U+FFFD).
    data[" Label"] = ["Web Attack � Brute Force", "Web Attack � Sql Injection", "Web Attack � XSS", "BENIGN"]

    cleaned = dataset.clean(data)

    labels = cleaned["Label"].tolist()
    assert labels == ["Web Attack", "Web Attack", "Web Attack", "Benign"]


def test_cic_ids2017_raw_labels_consolidate_every_raw_row_without_removing_any() -> None:
    dataset = _make_dataset()
    data = _base_frame()
    # Row 1 has a negative duration (clean() drops it) and rows 0 and 3 are duplicates; raw_labels
    # keeps all four.
    data["Flow Duration"] = [100, -1, 300, 100]
    data[" Label"] = ["DoS Hulk", "Web Attack � XSS", "BENIGN", "DoS Hulk"]

    assert dataset.raw_labels(data).tolist() == ["DoS/DDoS", "Web Attack", "Benign", "DoS/DDoS"]


def test_cic_ids2017_clean_drops_screen_drop_columns_like_drop_columns() -> None:
    config = DatasetConfig(
        name="cic_ids2017",
        raw_path="unused",
        processed_path="unused",
        label_column="Label",
        drop_columns=("Destination Port",),
        screen_drop_columns=("Fwd Header Length",),
    )

    cleaned = CICIDS2017Dataset(config).clean(_base_frame())

    assert "Destination Port" not in cleaned.columns
    assert "Fwd Header Length" not in cleaned.columns
    assert "Bwd Header Length" in cleaned.columns


def test_cic_ids2017_clean_raises_on_unrecognized_label() -> None:
    dataset = _make_dataset()
    data = _base_frame()
    data[" Label"] = ["Not A Real Label", "DDoS", "PortScan", "BENIGN"]

    with pytest.raises(KeyError):
        dataset.clean(data)


def test_cic_ids2017_clean_drops_exact_duplicate_rows_across_source_files() -> None:
    dataset = _make_dataset()
    # rows 0 (Monday) and 3 (Friday) have identical content but different source files;
    # rows 1 and 2 are unique. Content duplicates should be dropped regardless of source file.
    data = _base_frame()

    cleaned = dataset.clean(data)

    assert len(cleaned) == 3
    assert "source_file" in cleaned.columns
    assert set(cleaned["source_file"]) == {"Monday", "Tuesday", "Wednesday"}
