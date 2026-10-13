import pandas as pd

from xai_thesis.datasets.base import DatasetConfig
from xai_thesis.datasets.unsw_nb15 import UNSWNB15Dataset


def _make_dataset() -> UNSWNB15Dataset:
    config = DatasetConfig(
        name="unsw_nb15",
        raw_path="unused",
        processed_path="unused",
        label_column="attack_cat",
        drop_columns=("srcip", "sport", "dstip", "dsport", "ltime", "stime", "service", "label"),
    )
    return UNSWNB15Dataset(config)


def _row(**overrides: object) -> dict[str, object]:
    """A minimal-but-complete raw row (every column clean() touches), as raw strings -
    matching how load_raw() actually reads the headerless files (dtype=str,
    keep_default_na=False) - with overrides layered on top per test."""
    base = {
        "srcip": "59.166.0.0",
        "sport": "1390",
        "dstip": "149.171.126.6",
        "dsport": "53",
        "proto": "udp",
        "state": "CON",
        "dur": "0.001",
        "sbytes": "132",
        "dbytes": "164",
        "sttl": "31",
        "dttl": "29",
        "sloss": "0",
        "dloss": "0",
        "service": "dns",
        "sload": "500473.9",
        "dload": "621800.9",
        "spkts": "2",
        "dpkts": "2",
        "swin": "0",
        "dwin": "0",
        "stcpb": "0",
        "dtcpb": "0",
        "smean": "66",
        "dmean": "82",
        "trans_depth": "0",
        "response_body_len": "0",
        "sjit": "0",
        "djit": "0",
        "stime": "1421927414",
        "ltime": "1421927414",
        "sinpkt": "0.017",
        "dinpkt": "0.013",
        "tcprtt": "0",
        "synack": "0",
        "ackdat": "0",
        "is_sm_ips_ports": "0",
        "ct_state_ttl": "3",
        "ct_flw_http_mthd": "",
        "is_ftp_login": "0",
        "ct_ftp_cmd": "",
        "ct_srv_src": "7",
        "ct_srv_dst": "1",
        "ct_dst_ltm": "3",
        "ct_src_ltm": "1",
        "ct_src_dport_ltm": "1",
        "ct_dst_sport_ltm": "1",
        "ct_dst_src_ltm": "1",
        "attack_cat": "",
        "label": "0",
        "source_file": "UNSW-NB15_1.csv",
    }
    base.update(overrides)
    return base


def test_unsw_clean_drops_identifiers_timestamps_service_and_native_label() -> None:
    dataset = _make_dataset()
    data = pd.DataFrame([_row()])

    cleaned = dataset.clean(data)

    assert "srcip" not in cleaned.columns
    assert "sport" not in cleaned.columns
    assert "dstip" not in cleaned.columns
    assert "dsport" not in cleaned.columns
    assert "ltime" not in cleaned.columns
    assert "label" not in cleaned.columns
    assert "stime" not in cleaned.columns  # absolute timestamp, not a model feature
    assert "service" not in cleaned.columns  # proxy for the destination port
    assert "ct_ftp_cmd" in cleaned.columns


def test_unsw_clean_drops_truncated_rows_via_invalid_label() -> None:
    dataset = _make_dataset()
    # Simulates the one genuinely corrupted row found in the real raw files: truncated
    # mid-record, so label (like every later column) reads as blank rather than 0/1.
    data = pd.DataFrame([_row(), _row(attack_cat="Exploits", dur="0.5")])
    data.loc[1, "label"] = ""

    cleaned = dataset.clean(data)

    assert len(cleaned) == 1


def test_unsw_clean_maps_blank_attack_cat_to_benign_and_normalizes_backdoors_spelling() -> None:
    dataset = _make_dataset()
    data = pd.DataFrame(
        [
            _row(attack_cat=""),
            _row(attack_cat=" Backdoors ", dur="0.2"),
            _row(attack_cat="Backdoor", dur="0.3"),
        ]
    )

    cleaned = dataset.clean(data)

    assert cleaned["attack_cat"].tolist() == ["Benign", "Backdoor", "Backdoor"]


def test_unsw_raw_labels_normalize_every_raw_row_without_removing_any() -> None:
    dataset = _make_dataset()
    # Two identical rows: clean() would deduplicate them, raw_labels() must keep both.
    data = pd.DataFrame([_row(attack_cat=" Backdoors "), _row(attack_cat=" Backdoors "), _row(attack_cat="")])

    assert dataset.raw_labels(data).tolist() == ["Backdoor", "Backdoor", "Benign"]


def test_unsw_clean_fills_non_applicable_blanks_with_zero() -> None:
    dataset = _make_dataset()
    data = pd.DataFrame([_row(), _row(ct_flw_http_mthd="1", is_ftp_login="1", ct_ftp_cmd="2", dur="0.2")])

    cleaned = dataset.clean(data)

    assert cleaned["ct_flw_http_mthd"].tolist() == [0, 1]
    assert cleaned["is_ftp_login"].tolist() == [0, 1]
    assert cleaned["ct_ftp_cmd"].tolist() == [0, 2]


def test_unsw_clean_clips_ftp_login_bug_values() -> None:
    dataset = _make_dataset()
    data = pd.DataFrame([_row(is_ftp_login="2"), _row(is_ftp_login="4", dur="0.2")])

    cleaned = dataset.clean(data)

    assert cleaned["is_ftp_login"].tolist() == [1, 1]


def test_unsw_clean_fills_not_applicable_counts_with_zero_not_shared_imputer() -> None:
    dataset = _make_dataset()
    data = pd.DataFrame([_row(ct_flw_http_mthd="", is_ftp_login="")])

    cleaned = dataset.clean(data)

    assert cleaned["ct_flw_http_mthd"].tolist() == [0]
    assert cleaned["is_ftp_login"].tolist() == [0]


def test_unsw_clean_drops_duplicates_ignoring_source_file() -> None:
    # stime is dropped in step (5), so two rows identical in every flow feature but recorded
    # in different source files at different times collapse to one exact duplicate.
    dataset = _make_dataset()
    data = pd.DataFrame(
        [
            _row(source_file="UNSW-NB15_1.csv", stime="1421927414"),
            _row(source_file="UNSW-NB15_2.csv", stime="1424231129"),
            _row(attack_cat="Exploits", dur="0.9", source_file="UNSW-NB15_1.csv"),
        ]
    )

    cleaned = dataset.clean(data)

    assert len(cleaned) == 2
