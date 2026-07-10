from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from xai_thesis.utils import load_yaml


@dataclass(frozen=True)
class DatasetConfig:
    """Minimal dataset settings loaded from YAML."""

    name: str
    raw_path: Path
    processed_path: Path
    label_column: str
    binary_label_column: str | None = None
    drop_columns: tuple[str, ...] = field(default_factory=tuple)
    categorical_columns: tuple[str, ...] = field(default_factory=tuple)


def load_dataset_config(config_path: str | Path) -> DatasetConfig:
    """Load a DatasetConfig from a dataset YAML file."""
    raw = load_yaml(config_path)
    return DatasetConfig(
        name=raw["name"],
        raw_path=Path(raw["raw_path"]),
        processed_path=Path(raw["processed_path"]),
        label_column=raw["label_column"],
        binary_label_column=raw.get("binary_label_column"),
        drop_columns=tuple(raw.get("drop_columns", [])),
        categorical_columns=tuple(raw.get("categorical_columns", [])),
    )


class BaseDataset:
    """Shared interface for dataset-specific adapters."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def load_raw(self) -> pd.DataFrame:
        """Load raw files into one dataframe."""
        raise NotImplementedError

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply dataset-specific cleanup."""
        raise NotImplementedError

    def build(self) -> pd.DataFrame:
        """Return cleaned data ready for shared preprocessing."""
        return self.clean(self.load_raw())
