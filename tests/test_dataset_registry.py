from pathlib import Path

from xai_thesis.datasets import DATASET_REGISTRY
from xai_thesis.datasets.base import BaseDataset


def test_dataset_registry_matches_all_dataset_configs() -> None:
    config_names = {path.stem for path in Path("configs/datasets").glob("*.yaml")}

    assert config_names == set(DATASET_REGISTRY.keys())


def test_dataset_registry_values_are_base_dataset_subclasses() -> None:
    assert all(issubclass(cls, BaseDataset) for cls in DATASET_REGISTRY.values())
