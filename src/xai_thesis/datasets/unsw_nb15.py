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

# UNSW-NB15's raw files ship as headerless CSVs; this is NUSW-NB15_features.csv's documented
# column order, transcribed with two adjustments: a handful of columns are renamed to match
# this project's own already-established UNSW-NB15 convention (smeansz/dmeansz -> smean/dmean,
# res_bdy_len -> response_body_len, Sintpkt/Dintpkt -> sinpkt/dinpkt) for continuity with
# earlier analysis, and "ct_src_ ltm" (a literal typo - embedded space - in the features file
# itself) is fixed to ct_src_ltm. Everything else is just lowercased. Confirmed against the
# actual files: every row has 49 comma-separated values, matching this list's length.
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

# Every column above except the identifiers (dropped outright, see clean()) and the four
# string/categorical fields (proto/service/state/attack_cat) and label (dropped outright, only
# ever compared as a raw string - see clean()).
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
    "ct_srv_src",
    "ct_srv_dst",
    "ct_dst_ltm",
    "ct_src_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
]

# ct_flw_http_mthd/is_ftp_login are blank for ~53-56% of raw rows (is_ftp_login 56.3%,
# ct_flw_http_mthd 53.1% on the current raw capture) - verified this is concentrated in
# non-HTTP/non-FTP flows, not missing-at-random, so blank means "not applicable" (a real 0),
# not a value for the shared imputer to guess at from the other ~45%. ct_ftp_cmd shared this
# same treatment previously, but is now dropped outright (see
# configs/datasets/unsw_nb15.yaml) rather than cleaned - it turned out to be near-redundant
# with is_ftp_login (r~=0.85 on the current raw capture) and showed no per-class
# differentiation when nonzero.
_APPLICABILITY_ZERO_COLUMNS = ["ct_flw_http_mthd", "is_ftp_login"]


class UNSWNB15Dataset(BaseDataset):
    """UNSW-NB15 adapter, over the full raw capture (4 part files), not the pre-built
    10%-sampled training-set/testing-set files this project used previously."""

    def load_raw(self) -> pd.DataFrame:
        raw_path = Path(self.config.raw_path)
        frames = []
        for file_name in _RAW_FILES:
            # dtype=str + keep_default_na=False: the raw files mix real values with several
            # different "missing" placeholders (blank, single space, "-") across different
            # columns with different meanings (see clean()), so auto-NaN-ing on read would
            # erase which placeholder was actually used before this can react to it.
            frame = pd.read_csv(raw_path / file_name, header=None, names=_COLUMN_NAMES, dtype=str, keep_default_na=False)
            frame[_SOURCE_COLUMN] = file_name
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        # Steps (1)-(6) below follow the shared dataset-cleaning order documented in the
        # methodology chapter and mirrored by CICIDS2017Dataset.clean(); only the per-step
        # detail is UNSW-NB15-specific.
        data = data.copy()

        # (1) Coerce numeric types - the raw files are read as strings (see load_raw), and
        # unparseable entries become NaN for the shared median imputer to fill later.
        for column in _NUMERIC_COLUMNS:
            data[column] = pd.to_numeric(data[column], errors="coerce")

        # (2) Drop corrupted rows: keep only rows whose native binary label is literally
        # "0"/"1". The first download of UNSW-NB15_4.csv had its final record truncated
        # mid-line (6 fields instead of 49); re-downloading file 4 on 2026-08-28 fixed that,
        # and the current raw capture (2,540,044 rows) contains no such row, so this now
        # drops nothing. Kept because it catches that class of export corruption generically,
        # without hardcoding a row index that a fresh export could shift.
        data = data.loc[data["label"].isin(["0", "1"])].copy()

        # (3) Repair domain-specific invalid / non-applicable values.
        # ct_flw_http_mthd / is_ftp_login are blank ("not applicable") for non-HTTP/FTP flows
        # - a real 0, not a value for the imputer to guess (see _APPLICABILITY_ZERO_COLUMNS).
        data[_APPLICABILITY_ZERO_COLUMNS] = data[_APPLICABILITY_ZERO_COLUMNS].fillna(0)
        # is_ftp_login is documented as binary but has a data entry bug (values of 2 and 4).
        data["is_ftp_login"] = data["is_ftp_login"].clip(upper=1)
        # "-" and "" both mark "service not identified" in the raw data.
        data["service"] = data["service"].replace({"-": "unknown", "": "unknown"})

        # (4) Harmonize the label scheme. attack_cat is blank for benign rows in the raw
        # files (unlike the "Normal" string the pre-built files this project used previously
        # carried), and inconsistently whitespace-padded and spelled ("Backdoor" vs
        # "Backdoors") across the 4 files - normalized to "Backdoor" to match
        # _CLASS_OF_INTEREST and every other reference to this class in the project.
        label_column = self.config.label_column
        data[label_column] = data[label_column].str.strip().replace({"": "Benign", "Backdoors": "Backdoor"})

        # (5) Drop leakage / identifier / redundant feature columns (config.drop_columns):
        # srcip/dstip/sport/dsport (raw IP/port would let a model memorize attacker/victim
        # addresses instead of traffic behavior), ltime (redundant with dur/stime), the
        # native binary label (attack_cat is used instead - see load_dataset_config),
        # ct_ftp_cmd, and stime (an absolute timestamp, never a feature - dropped here so a
        # flow recorded twice at different times collapses to one exact duplicate in step 6).
        data = data.drop(columns=list(self.config.drop_columns), errors="ignore")

        # (6) Deduplicate on canonical flow content, ignoring only the source-file tag, so a
        # flow duplicated across two source files is still caught.
        content_columns = [column for column in data.columns if column != _SOURCE_COLUMN]
        return data.loc[~data.duplicated(subset=content_columns)].reset_index(drop=True)
