from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetConfig:
    """Minimal dataset settings loaded from YAML."""

    name: str
    raw_path: Path
    processed_path: Path
    label_column: str
    binary_label_column: str | None = None
    drop_columns: tuple[str, ...] = field(default_factory=tuple)


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
