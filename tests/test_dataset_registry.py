from pathlib import Path

from xai_thesis.datasets import DATASET_REGISTRY
from xai_thesis.datasets.base import BaseDataset


def test_dataset_registry_matches_dataset_configs() -> None:
    # Every config in configs/datasets/ needs a registered adapter, and every registered
    # adapter needs a config - no orphaned adapters for datasets that left the project.
    config_names = {path.stem for path in Path("configs/datasets").glob("*.yaml")}

    assert config_names == set(DATASET_REGISTRY.keys())


def test_dataset_registry_values_are_base_dataset_subclasses() -> None:
    assert all(issubclass(cls, BaseDataset) for cls in DATASET_REGISTRY.values())
