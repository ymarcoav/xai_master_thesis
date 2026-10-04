from pathlib import Path

import yaml

from xai_thesis.datasets.base import load_dataset_config


def test_load_dataset_config_reads_yaml(tmp_path) -> None:
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "sample",
                "raw_path": "data/raw/sample",
                "processed_path": "data/processed/sample",
                "label_column": "label",
                "drop_columns": ["id"],
                "categorical_columns": ["proto"],
            }
        ),
        encoding="utf-8",
    )

    config = load_dataset_config(config_path)

    assert config.name == "sample"
    assert config.raw_path == Path("data/raw/sample")
    assert config.processed_path == Path("data/processed/sample")
    assert config.label_column == "label"
    assert config.drop_columns == ("id",)
    assert config.categorical_columns == ("proto",)


def test_load_dataset_config_defaults_optional_fields(tmp_path) -> None:
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "sample",
                "raw_path": "data/raw/sample",
                "processed_path": "data/processed/sample",
                "label_column": "label",
            }
        ),
        encoding="utf-8",
    )

    config = load_dataset_config(config_path)

    assert config.drop_columns == ()
    assert config.screen_drop_columns == ()
    assert config.categorical_columns == ()
