from pathlib import Path

import numpy as np
import pandas as pd

from xai_thesis.datasets.base import BaseDataset

_HEADER_MARKER_COLUMN = "Flow ID"
_SOURCE_COLUMN = "source_file"


class DAPT2020Dataset(BaseDataset):
    """DAPT2020 adapter.

    Out of scope for the thesis as of 2026-07-31: kept fully functional, but no longer part
    of the actual thesis analysis - see README.md's Datasets section.
    """

    def load_raw(self) -> pd.DataFrame:
        raw_path = Path(self.config.raw_path)
        csv_files = sorted(raw_path.glob("*.csv"))

        # enp0s3-pvt-thursday.pcap_Flow.csv ships without its header row; every
        # other file shares the same 85-column schema, so borrow it from those.
        canonical_columns = next(
            pd.read_csv(file_path, nrows=0).columns.tolist()
            for file_path in csv_files
            if _file_has_header(file_path)
        )

        frames = []
        for file_path in csv_files:
            frame = (
                pd.read_csv(file_path)
                if _file_has_header(file_path)
                else pd.read_csv(file_path, header=None, names=canonical_columns)
            )
            frame[_SOURCE_COLUMN] = file_path.stem
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        # Steps (1)-(6) below follow the shared dataset-cleaning order documented in the
        # methodology chapter and mirrored by CICIDS2017Dataset.clean() (DAPT2020 is
        # CICFlowMeter output too). Step (3), repairing non-applicable values, has no
        # DAPT2020 case.

        # (1) Coerce numeric types. Flow Bytes/s and Flow Packets/s divide by flow duration
        # and yield ±Inf where it is 0 - treat as missing so the shared median imputer fills
        # them, matching CICIDS2017Dataset. (Not re-verified against DAPT2020's own row counts,
        # since the dataset is out of thesis scope - kept for structural parity so both
        # CICFlowMeter adapters handle the exporter's quirks identically.)
        data = data.replace([np.inf, -np.inf], np.nan)

        # (2) Drop corrupted rows. Flow Duration and *IAT columns are occasionally negative
        # due to the same flow-exporter timestamp bug seen in CIC-IDS2017; those rows are
        # corrupted, not valid traffic.
        duration_columns = [column for column in data.columns if column == "Flow Duration" or "IAT" in column]
        data = data.loc[~(data[duration_columns] < 0).any(axis=1)]

        # (4) Harmonize the label scheme: some files spell the benign class "BENIGN".
        label_column = self.config.label_column
        data[label_column] = data[label_column].replace("BENIGN", "Benign")

        # (5) Drop leakage / identifier / redundant feature columns (config.drop_columns): the
        # synthetic flow identifier, the finer-grained Activity label, the fixed testbed
        # identity columns (IPs would let a model memorize attacker/victim addresses instead
        # of traffic behavior), and Timestamp (an absolute capture time, never a feature -
        # dropped here so a flow recorded twice at different times collapses to one exact
        # duplicate in step 6).
        data = data.drop(columns=list(self.config.drop_columns), errors="ignore")

        # (6) Deduplicate on canonical flow content, ignoring the source-file tag, so a flow
        # duplicated across two source files is still caught.
        content_columns = [column for column in data.columns if column != _SOURCE_COLUMN]
        return data.loc[~data.duplicated(subset=content_columns)].reset_index(drop=True)


def _file_has_header(file_path: Path) -> bool:
    with file_path.open("r", encoding="utf-8") as handle:
        first_field = handle.readline().split(",", 1)[0]
    return first_field.strip() == _HEADER_MARKER_COLUMN
