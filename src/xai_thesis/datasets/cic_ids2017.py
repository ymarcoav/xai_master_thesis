import pandas as pd

from xai_thesis.datasets.base import BaseDataset


class CICIDS2017Dataset(BaseDataset):
    """CIC-IDS-2017 adapter."""

    def load_raw(self) -> pd.DataFrame:
        # Add raw CSV loading once file names are fixed.
        raise NotImplementedError

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        # Add CIC-IDS-2017-specific label and column cleanup here.
        return data
