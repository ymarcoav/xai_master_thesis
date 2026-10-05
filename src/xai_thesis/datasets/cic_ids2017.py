"""CIC-IDS2017 adapter: loads the eight raw CSV files and applies the dataset-specific cleaning."""

from pathlib import Path

import numpy as np
import pandas as pd

from xai_thesis.datasets.base import BaseDataset

_DUPLICATE_COLUMN = "Fwd Header Length.1"
_SOURCE_COLUMN = "source_file"

# Native labels consolidated into seven classes, following the grouping of Arreche et al. (2024):
# the DoS and DDoS variants and Heartbleed become DoS/DDoS, FTP-Patator and SSH-Patator become
# Brute Force, and the three Web Attack subtypes become Web Attack (see _map_to_category).
_ATTACK_CATEGORY: dict[str, str] = {
    "DoS Hulk": "DoS/DDoS",
    "DDoS": "DoS/DDoS",
    "DoS GoldenEye": "DoS/DDoS",
    "DoS slowloris": "DoS/DDoS",
    "DoS Slowhttptest": "DoS/DDoS",
    "Heartbleed": "DoS/DDoS",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "PortScan": "PortScan",
    "Infiltration": "Infiltration",
    "Bot": "Bot",
}


def _map_to_category(value: str) -> str:
    """Map a native CIC-IDS2017 label to Benign or one of the six attack classes.

    The three "Web Attack - <subtype>" labels are matched by prefix, because the separator in the
    raw files is a corrupted non-ASCII character.
    """
    if value == "BENIGN":
        return "Benign"
    if value.startswith("Web Attack"):
        return "Web Attack"
    return _ATTACK_CATEGORY[value]


class CICIDS2017Dataset(BaseDataset):
    """CIC-IDS2017 adapter over the eight raw CSV files."""

    def load_raw(self) -> pd.DataFrame:
        raw_path = Path(self.config.raw_path)
        frames = []
        for file_path in sorted(raw_path.glob("*.csv")):
            frame = pd.read_csv(file_path)
            frame[_SOURCE_COLUMN] = file_path.stem
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        # Column names carry inconsistent leading spaces, and "Fwd Header Length" appears twice
        # (pandas names the identical copy "Fwd Header Length.1").
        data = data.rename(columns=lambda column: column.strip())
        data = data.drop(columns=[_DUPLICATE_COLUMN], errors="ignore")


        # (1) Flow Bytes/s and Flow Packets/s are infinite for flows with a duration of zero;
        #     infinities are treated as missing and filled by the median imputer.
        data = data.replace([np.inf, -np.inf], np.nan)

        # (2) Drop corrupted rows with physically impossible negative values: negative flow
        #     durations or inter-arrival times (a known CICFlowMeter timestamp error), and
        #     negative header lengths or segment sizes (an integer overflow).
        duration_columns = [column for column in data.columns if column == "Flow Duration" or "IAT" in column]
        data = data.loc[~(data[duration_columns] < 0).any(axis=1)]
        non_negative_columns = ["Fwd Header Length", "Bwd Header Length", "min_seg_size_forward"]
        data = data.loc[~(data[non_negative_columns] < 0).any(axis=1)]

        # (3) Consolidate the labels into seven classes (see _map_to_category).
        label_column = self.config.label_column
        data[label_column] = data[label_column].map(_map_to_category)

        # (4) Drop Destination Port (drop_columns) and the constant and near-duplicate columns
        #     found by the feature screen (screen_drop_columns); see
        #     configs/datasets/cic_ids2017.yaml.
        data = self.drop_configured_columns(data)

        # (5) Remove exact duplicate records, ignoring the source-file tag.
        content_columns = [column for column in data.columns if column != _SOURCE_COLUMN]
        return data.loc[~data.duplicated(subset=content_columns)].reset_index(drop=True)

    def raw_labels(self, data: pd.DataFrame) -> pd.Series:
        # The raw label header is " Label" (leading space, see clean()).
        return data.rename(columns=lambda column: column.strip())[self.config.label_column].map(_map_to_category)
