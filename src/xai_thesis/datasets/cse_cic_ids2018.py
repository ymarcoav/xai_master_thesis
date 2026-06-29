import pandas as pd

from xai_thesis.datasets.base import BaseDataset


class CSECICIDS2018Dataset(BaseDataset):
    """CSE-CIC-IDS2018 adapter."""

    def load_raw(self) -> pd.DataFrame:
        # Add raw CSV loading once file names are fixed.
        raise NotImplementedError

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        # Add CIC-specific label and column cleanup here.
        return data
