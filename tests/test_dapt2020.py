import pandas as pd

from xai_thesis.datasets.base import DatasetConfig
from xai_thesis.datasets.dapt2020 import DAPT2020Dataset


def _make_dataset() -> DAPT2020Dataset:
    config = DatasetConfig(
        name="dapt2020",
        raw_path="unused",
        processed_path="unused",
        label_column="Stage",
        drop_columns=("Flow ID", "Activity", "Src IP", "Dst IP", "Timestamp"),
    )
    return DAPT2020Dataset(config)


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Flow ID": ["1.1.1.1-2.2.2.2-1-2-6", "1.1.1.1-2.2.2.2-1-2-6"],
            "Src IP": ["1.1.1.1", "1.1.1.1"],
            "Dst IP": ["2.2.2.2", "2.2.2.2"],
            "Timestamp": ["01/01/2020 01:00:00 AM", "01/01/2020 01:00:01 AM"],
            "Flow Duration": [100, 200],
            "Flow IAT Min": [1, 2],
            "Activity": ["Normal", "Network Scan"],
            "Stage": ["Benign", "BENIGN"],
            "source_file": ["enp0s3-monday", "enp0s3-monday"],
        }
    )


def test_dapt2020_clean_drops_identifier_activity_identity_and_timestamp_columns() -> None:
    dataset = _make_dataset()
    data = _base_frame()

    cleaned = dataset.clean(data)

    assert list(cleaned.columns) == ["Flow Duration", "Flow IAT Min", "Stage", "source_file"]


def test_dapt2020_clean_normalizes_benign_casing() -> None:
    dataset = _make_dataset()
    data = _base_frame()

    cleaned = dataset.clean(data)

    assert set(cleaned["Stage"]) == {"Benign"}


def test_dapt2020_clean_drops_rows_with_negative_duration_or_iat() -> None:
    dataset = _make_dataset()
    data = _base_frame()
    data.loc[1, "Flow Duration"] = -1

    cleaned = dataset.clean(data)

    assert len(cleaned) == 1
    assert (cleaned["Flow Duration"] >= 0).all()


def test_dapt2020_clean_drops_exact_duplicate_rows_across_source_files() -> None:
    dataset = _make_dataset()
    data = _base_frame()
    data.loc[1] = data.loc[0]
    data.loc[1, "source_file"] = "enp0s3-tuesday"  # same content, different source file

    cleaned = dataset.clean(data)

    assert len(cleaned) == 1
    assert cleaned["source_file"].tolist() == ["enp0s3-monday"]


def test_dapt2020_clean_dedup_ignores_timestamp_differences() -> None:
    # Same flow content recorded twice at different real times is still the same duplicate
    # flow for modeling purposes - Timestamp is dropped in the column-drop step, so the two
    # rows are byte-identical by the time dedup runs.
    dataset = _make_dataset()
    data = _base_frame()
    data.loc[1] = data.loc[0]
    data.loc[1, "Timestamp"] = "01/01/2020 02:30:00 PM"

    cleaned = dataset.clean(data)

    assert len(cleaned) == 1
