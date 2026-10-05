"""Dataset configuration and the base class of the dataset adapters."""

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
    drop_columns: tuple[str, ...] = field(default_factory=tuple)
    # Columns removed by the constant / near-duplicate screen. Kept separate from drop_columns so
    # that scripts/feature_screen.py can rerun the screen with them present; clean() drops both.
    screen_drop_columns: tuple[str, ...] = field(default_factory=tuple)
    categorical_columns: tuple[str, ...] = field(default_factory=tuple)


def load_dataset_config(config_path: str | Path) -> DatasetConfig:
    """Load a DatasetConfig from a dataset YAML file."""
    raw = load_yaml(config_path)
    return DatasetConfig(
        name=raw["name"],
        raw_path=Path(raw["raw_path"]),
        processed_path=Path(raw["processed_path"]),
        label_column=raw["label_column"],
        drop_columns=tuple(raw.get("drop_columns", [])),
        screen_drop_columns=tuple(raw.get("screen_drop_columns", [])),
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

    def raw_labels(self, data: pd.DataFrame) -> pd.Series:
        """The normalized label of every row of load_raw()'s output, with no row removed - the
        raw class distribution the datasets chapter plots."""
        raise NotImplementedError

    def drop_configured_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drop config.drop_columns and config.screen_drop_columns (absent names are ignored)."""
        columns = [*self.config.drop_columns, *self.config.screen_drop_columns]
        return data.drop(columns=columns, errors="ignore")

    def build(self) -> pd.DataFrame:
        """Return cleaned data ready for shared preprocessing."""
        return self.clean(self.load_raw())
