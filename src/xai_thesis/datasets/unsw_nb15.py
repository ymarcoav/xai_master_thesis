from pathlib import Path

import pandas as pd

from xai_thesis.datasets.base import BaseDataset

_TRAIN_FILE = "UNSW_NB15_training-set.csv"
_TEST_FILE = "UNSW_NB15_testing-set.csv"


class UNSWNB15Dataset(BaseDataset):
    """UNSW-NB15 adapter."""

    def load_raw(self) -> pd.DataFrame:
        raw_path = Path(self.config.raw_path)
        train = pd.read_csv(raw_path / _TRAIN_FILE).assign(split="train")
        test = pd.read_csv(raw_path / _TEST_FILE).assign(split="test")
        return pd.concat([train, test], ignore_index=True)

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        # attack_cat is UNSW-NB15's native multiclass label; "Normal" is renamed to "Benign"
        # to match this project's convention across every other dataset. The native binary
        # label (0/1, now unused) is dropped below.
        data["attack_cat"] = data["attack_cat"].replace("Normal", "Benign")

        # Drop row identifiers and the (now unused) native binary label.
        data = data.drop(columns=list(self.config.drop_columns), errors="ignore")

        # "-" marks "service not identified" in the raw data; make it explicit.
        data["service"] = data["service"].replace("-", "unknown")

        # is_ftp_login is documented as binary but contains a data entry bug (values of 2).
        data["is_ftp_login"] = data["is_ftp_login"].clip(upper=1)

        # Drop exact duplicate rows last (after the fixes above, in case they reveal new
        # duplicates), but only within a split: the official train/test files overlap by
        # ~4k rows, which we keep so the split stays comparable to published benchmarks.
        return data.drop_duplicates().reset_index(drop=True)
