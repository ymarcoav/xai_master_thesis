"""UNSW-NB15 adapter: loads the four raw CSV files and applies the dataset-specific cleaning."""

from pathlib import Path

import pandas as pd

from xai_thesis.datasets.base import BaseDataset

_SOURCE_COLUMN = "source_file"

_RAW_FILES = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv",
]

# The raw files have no header. Column order as documented in NUSW-NB15_features.csv, lowercased,
# with a few names shortened (smeansz/dmeansz -> smean/dmean, res_bdy_len -> response_body_len,
# Sintpkt/Dintpkt -> sinpkt/dinpkt) and the typo "ct_src_ ltm" corrected to ct_src_ltm.
_COLUMN_NAMES = [
    "srcip",
    "sport",
    "dstip",
    "dsport",
    "proto",
    "state",
    "dur",
    "sbytes",
    "dbytes",
    "sttl",
    "dttl",
    "sloss",
    "dloss",
    "service",
    "sload",
    "dload",
    "spkts",
    "dpkts",
    "swin",
    "dwin",
    "stcpb",
    "dtcpb",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "sjit",
    "djit",
    "stime",
    "ltime",
    "sinpkt",
    "dinpkt",
    "tcprtt",
    "synack",
    "ackdat",
    "is_sm_ips_ports",
    "ct_state_ttl",
    "ct_flw_http_mthd",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_srv_src",
    "ct_srv_dst",
    "ct_dst_ltm",
    "ct_src_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "attack_cat",
    "label",
]

# Columns converted to numbers; the identifiers, the categorical fields (proto, service, state)
# and the two label columns stay strings. service is dropped in clean() (see the dataset config).
_NUMERIC_COLUMNS = [
    "dur",
    "sbytes",
    "dbytes",
    "sttl",
    "dttl",
    "sloss",
    "dloss",
    "sload",
    "dload",
    "spkts",
    "dpkts",
    "swin",
    "dwin",
    "stcpb",
    "dtcpb",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "sjit",
    "djit",
    "stime",
    "ltime",
    "sinpkt",
    "dinpkt",
    "tcprtt",
    "synack",
    "ackdat",
    "is_sm_ips_ports",
    "ct_state_ttl",
    "ct_flw_http_mthd",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_srv_src",
    "ct_srv_dst",
    "ct_dst_ltm",
    "ct_src_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
]

# ct_flw_http_mthd, is_ftp_login and ct_ftp_cmd are blank in more than half of the raw records
# (53.1%, 56.3% and 56.3%), concentrated in non-HTTP and non-FTP flows: the feature does not apply,
# so blank means 0 rather than a missing value for the imputer.
_APPLICABILITY_ZERO_COLUMNS = ["ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd"]


def _normalize_labels(labels: pd.Series) -> pd.Series:
    """Normalize attack_cat: strip whitespace, map blank (benign) records to "Benign", and unify
    the two spellings "Backdoor" and "Backdoors"."""
    return labels.str.strip().replace({"": "Benign", "Backdoors": "Backdoor"})


class UNSWNB15Dataset(BaseDataset):
    """UNSW-NB15 adapter over the full raw capture (the four UNSW-NB15_*.csv files)."""

    def load_raw(self) -> pd.DataFrame:
        raw_path = Path(self.config.raw_path)
        frames = []
        for file_name in _RAW_FILES:
            # Read everything as text so that placeholders (blank entries) are kept as written and
            # handled explicitly in clean().
            frame = pd.read_csv(raw_path / file_name, header=None, names=_COLUMN_NAMES, dtype=str, keep_default_na=False)
            frame[_SOURCE_COLUMN] = file_name
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()

        # (1) Convert numeric columns; unparseable entries become NaN for the median imputer.
        for column in _NUMERIC_COLUMNS:
            data[column] = pd.to_numeric(data[column], errors="coerce")

        # (2) Drop corrupted rows: keep only records whose native binary label is "0" or "1"
        #     (guards against truncated records; the current files contain none).
        data = data.loc[data["label"].isin(["0", "1"])].copy()

        # (3) Repair invalid and non-applicable values. Blank ct_flw_http_mthd / is_ftp_login /
        #     ct_ftp_cmd mean "not applicable" and become 0 (see _APPLICABILITY_ZERO_COLUMNS).
        data[_APPLICABILITY_ZERO_COLUMNS] = data[_APPLICABILITY_ZERO_COLUMNS].fillna(0)
        # is_ftp_login is documented as binary but contains values of 2 and 4.
        data["is_ftp_login"] = data["is_ftp_login"].clip(upper=1)

        # (4) Normalize the labels (see _normalize_labels).
        label_column = self.config.label_column
        data[label_column] = _normalize_labels(data[label_column])

        # (5) Drop the configured columns: connection identifiers, capture timestamps, service
        #     and the native binary label (see configs/datasets/unsw_nb15.yaml).
        data = self.drop_configured_columns(data)

        # (6) Remove exact duplicate records, ignoring the source-file tag.
        content_columns = [column for column in data.columns if column != _SOURCE_COLUMN]
        return data.loc[~data.duplicated(subset=content_columns)].reset_index(drop=True)

    def raw_labels(self, data: pd.DataFrame) -> pd.Series:
        return _normalize_labels(data[self.config.label_column])
