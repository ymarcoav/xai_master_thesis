import pandas as pd

from xai_thesis.datasets.base import BaseDataset


class UNSWNB15Dataset(BaseDataset):
    """UNSW-NB15 adapter."""

    def load_raw(self) -> pd.DataFrame:
        # Add raw CSV loading once file names are fixed.
        raise NotImplementedError

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        # Drop row identifiers and binary label leakage for multiclass experiments.
        return data.drop(columns=list(self.config.drop_columns), errors="ignore")
