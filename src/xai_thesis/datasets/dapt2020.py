import pandas as pd

from xai_thesis.datasets.base import BaseDataset


class DAPT2020Dataset(BaseDataset):
    """DAPT2020 adapter."""

    def load_raw(self) -> pd.DataFrame:
        # Add raw CSV loading once file names are fixed.
        raise NotImplementedError

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        # Add DAPT2020-specific label and column cleanup here.
        return data
