from pathlib import Path

import pandas as pd

from xai_thesis.datasets.base import BaseDataset

_HEADER_MARKER_COLUMN = "Flow ID"
_SOURCE_COLUMN = "source_file"
_TIMESTAMP_COLUMN = "Timestamp"


class DAPT2020Dataset(BaseDataset):
    """DAPT2020 adapter."""

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
        # Drop the synthetic flow identifier, the finer-grained Activity label, and the
        # fixed testbed identity columns (IPs would let a model memorize attacker/victim
        # addresses instead of learning traffic behavior). Timestamp is deliberately NOT in
        # drop_columns: it survives clean() so pipeline.py's train/test split can sort by it
        # (raw row order was checked and is not already chronological), and is dropped there
        # immediately after sorting, before it ever reaches preprocessing/the model.
        data = data.drop(columns=list(self.config.drop_columns), errors="ignore")

        # Some files spell the benign class "BENIGN" instead of "Benign".
        label_column = self.config.label_column
        data[label_column] = data[label_column].replace("BENIGN", "Benign")

        # Flow Duration and *IAT columns are occasionally negative due to the same
        # flow-exporter timestamp bug seen in CIC-IDS2017.
        duration_columns = [column for column in data.columns if column == "Flow Duration" or "IAT" in column]
        data = data.loc[~(data[duration_columns] < 0).any(axis=1)]

        # Dedupe on flow content only, ignoring the source-file tag and Timestamp, so a flow
        # duplicated across two source files (or recorded twice at different real times) is
        # still caught.
        content_columns = [column for column in data.columns if column not in (_SOURCE_COLUMN, _TIMESTAMP_COLUMN)]
        return data.loc[~data.duplicated(subset=content_columns)].reset_index(drop=True)


def _file_has_header(file_path: Path) -> bool:
    with file_path.open("r", encoding="utf-8") as handle:
        first_field = handle.readline().split(",", 1)[0]
    return first_field.strip() == _HEADER_MARKER_COLUMN
