"""Dataset-specific loading and cleaning code."""

from xai_thesis.datasets.base import BaseDataset
from xai_thesis.datasets.cic_ids2017 import CICIDS2017Dataset
from xai_thesis.datasets.unsw_nb15 import UNSWNB15Dataset

DATASET_REGISTRY: dict[str, type[BaseDataset]] = {
    "unsw_nb15": UNSWNB15Dataset,
    "cic_ids2017": CICIDS2017Dataset,
}
