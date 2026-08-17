from pathlib import Path

from xai_thesis.datasets import DATASET_REGISTRY
from xai_thesis.datasets.base import BaseDataset


def test_dataset_registry_covers_all_active_dataset_configs() -> None:
    # Every active config (configs/datasets/) needs a registered adapter, but the registry
    # can have more entries than that - archived datasets (see configs/_archive/datasets/,
    # e.g. NSL-KDD/DAPT2020) keep their adapter registered and fully functional even though
    # their config moved out of the active directory, by design.
    config_names = {path.stem for path in Path("configs/datasets").glob("*.yaml")}

    assert config_names <= set(DATASET_REGISTRY.keys())


def test_dataset_registry_values_are_base_dataset_subclasses() -> None:
    assert all(issubclass(cls, BaseDataset) for cls in DATASET_REGISTRY.values())
