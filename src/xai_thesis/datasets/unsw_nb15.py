import pandas as pd

from xai_thesis.datasets.base import BaseDataset


class UNSWNB15Dataset(BaseDataset):
    """UNSW-NB15 adapter."""

    def load_raw(self) -> pd.DataFrame:
        # Add raw CSV loading once file names are fixed.
        raise NotImplementedError

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        # Add UNSW-specific label and column cleanup here.
        return data
