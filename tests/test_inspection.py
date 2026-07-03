import pandas as pd
import yaml

from xai_thesis.inspection import inspect_dataset_config


def test_inspect_dataset_config_writes_summary(tmp_path) -> None:
    raw_path = tmp_path / "raw"
    raw_path.mkdir()
    config_path = tmp_path / "dataset.yaml"
    output_path = tmp_path / "outputs"

    data = pd.DataFrame(
        {
            " feature ": [1.0, None, float("inf")],
            "Label": ["BENIGN", "Attack", "Attack"],
        }
    )
    data.to_csv(raw_path / "sample.csv", index=False)
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "sample",
                "raw_path": raw_path.as_posix(),
                "processed_path": "unused",
                "label_column": "label",
            }
        ),
        encoding="utf-8",
    )

    summary = inspect_dataset_config(config_path, output_path)

    assert summary["aggregate"]["rows_scanned"] == 3
    assert summary["files"][0]["label_column_found"] == "Label"
    assert summary["aggregate"]["label_counts"] == {"Attack": 2, "BENIGN": 1}
    assert summary["aggregate"]["columns_with_whitespace"] == [" feature "]
    assert summary["aggregate"]["numeric_columns_with_non_finite_values"] == [" feature "]
    assert (output_path / "sample" / "inspection.json").exists()
    assert (output_path / "sample" / "inspection.md").exists()
