from pathlib import Path

import numpy as np
import pandas as pd

from xai_thesis.datasets.base import BaseDataset

_DUPLICATE_COLUMN = "Fwd Header Length.1"


class CICIDS2017Dataset(BaseDataset):
    """CIC-IDS-2017 adapter."""

    def load_raw(self) -> pd.DataFrame:
        raw_path = Path(self.config.raw_path)
        frames = [pd.read_csv(file_path) for file_path in sorted(raw_path.glob("*.csv"))]
        return pd.concat(frames, ignore_index=True)

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        # Column names are inconsistently prefixed with a leading space in the raw files.
        data = data.rename(columns=lambda column: column.strip())

        # "Fwd Header Length" is duplicated verbatim in every raw file; the second copy
        # (auto-suffixed ".1" by pandas) is byte-for-byte identical to the first.
        data = data.drop(columns=[_DUPLICATE_COLUMN], errors="ignore")

        # Flow Bytes/s and Flow Packets/s divide by a near-zero duration; treat the
        # resulting Infinity the same as any other missing value for the shared imputer.
        data = data.replace([np.inf, -np.inf], np.nan)

        # Flow Duration and *IAT columns are occasionally negative due to a known
        # flow-exporter timestamp bug; these rows are corrupted, not valid traffic.
        duration_columns = [column for column in data.columns if column == "Flow Duration" or "IAT" in column]
        data = data.loc[~(data[duration_columns] < 0).any(axis=1)]

        # The raw files already contain a corrupted replacement character where an
        # en-dash was lost (e.g. "Web Attack � Brute Force").
        label_column = self.config.label_column
        data[label_column] = data[label_column].str.replace("�", "-", regex=False)

        return data.drop_duplicates().reset_index(drop=True)
