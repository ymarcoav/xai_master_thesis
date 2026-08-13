from pathlib import Path

import pandas as pd

from xai_thesis.datasets.base import BaseDataset

_TRAIN_FILE = "KDDTrain+.txt"
_TEST_FILE = "KDDTest+.txt"

# NSL-KDD's raw files ship as headerless CSVs; this is the documented column order (41 named
# features + the native "class" label), plus one further, unnamed trailing column ("difficulty
# level" - a KDD-Cup-era record-difficulty score, not a traffic feature) - confirmed against
# the actual files: every row has 43 comma-separated values, not the 42 the named attributes
# alone would suggest.
_COLUMN_NAMES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "class",
    "difficulty_level",
]

# NSL-KDD's native "class" column is the full ~40-way specific attack-type scheme (e.g.
# "neptune", "buffer_overflow"), not the 4-category grouping used here. This project models
# NSL-KDD as multiclass over the conventional DoS/Probe/R2L/U2R categories (+ Benign) rather
# than either the full specific-attack-type scheme or a binary collapse - the standard
# grouping used throughout NSL-KDD literature (e.g. Tavallaee et al. 2009; Dhanabal &
# Shantharajah 2015). Compiled from, and verified against, every distinct label actually
# present in both KDDTrain+.txt and KDDTest+.txt - the test set deliberately introduces attack
# types absent from training (e.g. "mscan", "httptunnel", "sqlattack") to test generalization
# to novel attacks, so the mapping below covers the union of both files, not just training.
_ATTACK_CATEGORY: dict[str, str] = {
    # DoS (Denial of Service)
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "apache2": "DoS",
    "udpstorm": "DoS",
    "processtable": "DoS",
    "mailbomb": "DoS",
    # Probe (surveillance/probing)
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "satan": "Probe",
    "mscan": "Probe",
    "saint": "Probe",
    # R2L (Remote to Local)
    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "phf": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",
    "named": "R2L",
    "sendmail": "R2L",
    "snmpgetattack": "R2L",
    "snmpguess": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "worm": "R2L",
    "httptunnel": "R2L",
    # U2R (User to Root)
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
    "ps": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
}


def _map_to_category(value: str) -> str:
    """Map a native NSL-KDD class value to Benign or its DoS/Probe/R2L/U2R category.

    Raises (via the dict lookup's KeyError) on any value that isn't "normal" and isn't one of
    the specific attack types this mapping was verified against, rather than silently folding
    an unrecognized label into Benign - a data issue here should be loud, not quietly wrong.
    """
    if value == "normal":
        return "Benign"
    return _ATTACK_CATEGORY[value]


class NSLKDDDataset(BaseDataset):
    """NSL-KDD adapter.

    Out of scope for the thesis as of 2026-08-13: kept fully functional, but no longer part
    of the actual thesis analysis - see README.md's Datasets section.
    """

    def load_raw(self) -> pd.DataFrame:
        raw_path = Path(self.config.raw_path)
        train = pd.read_csv(raw_path / _TRAIN_FILE, header=None, names=_COLUMN_NAMES).assign(split="train")
        test = pd.read_csv(raw_path / _TEST_FILE, header=None, names=_COLUMN_NAMES).assign(split="test")
        return pd.concat([train, test], ignore_index=True)

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        # difficulty_level is a KDD-Cup-era per-record difficulty score, not a traffic feature.
        data = data.drop(columns=list(self.config.drop_columns), errors="ignore")

        label_column = self.config.label_column
        data[label_column] = data["class"].map(_map_to_category)
        data = data.drop(columns=["class"], errors="ignore")

        # Drop exact duplicate rows last, but only within a split - NSL-KDD's own stated
        # design goal (over the original KDD Cup '99 data) was already removing duplicates,
        # so this is mostly a safety net, not an expected source of many removals.
        return data.drop_duplicates().reset_index(drop=True)
