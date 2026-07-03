from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from xai_thesis.utils import ensure_directory, load_yaml


def inspect_dataset_config(
    config_path: str | Path,
    output_dir: str | Path = "outputs/metrics/dataset_inspection",
    *,
    max_rows: int | None = None,
    chunksize: int = 100_000,
) -> dict[str, Any]:
    """Inspect raw CSV files for one dataset config."""
    config_path = Path(config_path)
    config = load_yaml(config_path)
    raw_path = Path(config["raw_path"])
    csv_files = sorted(raw_path.glob("*.csv"))

    summary = {
        "dataset": config["name"],
        "config_path": config_path.as_posix(),
        "raw_path": raw_path.as_posix(),
        "configured_label_column": config.get("label_column"),
        "max_rows": max_rows,
        "files": [
            inspect_csv_file(file_path, config.get("label_column"), max_rows, chunksize)
            for file_path in csv_files
        ],
    }
    summary["aggregate"] = _aggregate_file_summaries(summary["files"])

    output_path = ensure_directory(Path(output_dir) / config["name"])
    json_path = output_path / "inspection.json"
    markdown_path = output_path / "inspection.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown_summary(summary), encoding="utf-8")

    return summary


def inspect_csv_file(
    file_path: str | Path,
    label_column: str | None,
    max_rows: int | None = None,
    chunksize: int = 100_000,
) -> dict[str, Any]:
    """Inspect one CSV file without loading large datasets fully into memory."""
    file_path = Path(file_path)
    row_count = 0
    columns: list[str] = []
    dtypes: dict[str, set[str]] = {}
    missing_counts: Counter[str] = Counter()
    non_finite_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    label_column_found: str | None = None

    for chunk in _read_csv_chunks(file_path, max_rows=max_rows, chunksize=chunksize):
        if not columns:
            columns = list(chunk.columns)
            label_column_found = _resolve_column_name(columns, label_column)

        row_count += len(chunk)

        for column, dtype in chunk.dtypes.items():
            dtypes.setdefault(column, set()).add(str(dtype))

        missing_counts.update(chunk.isna().sum().to_dict())

        numeric_columns = chunk.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            non_finite = ~np.isfinite(chunk[numeric_columns])
            non_finite_counts.update(non_finite.sum().to_dict())

        if label_column_found is not None:
            labels = chunk[label_column_found].fillna("<missing>").astype(str)
            label_counts.update(labels.value_counts().to_dict())

    stripped_columns = [column.strip() for column in columns]
    duplicate_columns = sorted(_duplicates(columns))
    duplicate_columns_after_strip = sorted(_duplicates(stripped_columns))
    columns_with_whitespace = [
        column for column in columns if column != column.strip()
    ]

    return {
        "file": file_path.as_posix(),
        "rows_scanned": row_count,
        "columns": columns,
        "column_count": len(columns),
        "dtypes": {column: sorted(values) for column, values in dtypes.items()},
        "missing_counts": dict(sorted(missing_counts.items())),
        "non_finite_numeric_counts": dict(sorted(non_finite_counts.items())),
        "label_column_found": label_column_found,
        "label_counts": dict(label_counts.most_common()),
        "duplicate_columns": duplicate_columns,
        "duplicate_columns_after_strip": duplicate_columns_after_strip,
        "columns_with_whitespace": columns_with_whitespace,
        "empty_columns": [
            column for column, count in missing_counts.items() if count == row_count and row_count > 0
        ],
    }


def render_markdown_summary(summary: dict[str, Any]) -> str:
    """Render a compact human-readable inspection report."""
    aggregate = summary["aggregate"]
    lines = [
        f"# Dataset Inspection: {summary['dataset']}",
        "",
        f"- Raw path: `{summary['raw_path']}`",
        f"- Files inspected: {aggregate['file_count']}",
        f"- Rows scanned: {aggregate['rows_scanned']}",
        f"- Unique columns observed: {aggregate['unique_column_count']}",
        f"- Configured label column: `{summary['configured_label_column']}`",
        "",
        "## Files",
        "",
        "| File | Rows | Columns | Label Column | Label Classes |",
        "| --- | ---: | ---: | --- | ---: |",
    ]

    for file_summary in summary["files"]:
        label_count = len(file_summary["label_counts"])
        lines.append(
            "| "
            f"{Path(file_summary['file']).name} | "
            f"{file_summary['rows_scanned']} | "
            f"{file_summary['column_count']} | "
            f"`{file_summary['label_column_found']}` | "
            f"{label_count} |"
        )

    lines.extend(["", "## Labels", ""])
    if aggregate["label_counts"]:
        lines.extend(_counter_table("Label", aggregate["label_counts"]))
    else:
        lines.append("No configured label column was found.")

    lines.extend(["", "## Data Quality Flags", ""])
    lines.append(f"- Columns with leading/trailing whitespace: {aggregate['columns_with_whitespace']}")
    lines.append(f"- Duplicate columns: {aggregate['duplicate_columns']}")
    lines.append(f"- Duplicate columns after stripping whitespace: {aggregate['duplicate_columns_after_strip']}")
    lines.append(f"- Empty columns: {aggregate['empty_columns']}")
    lines.append(f"- Columns with missing values: {aggregate['columns_with_missing_values']}")
    lines.append(f"- Numeric columns with non-finite values: {aggregate['numeric_columns_with_non_finite_values']}")
    lines.append("")

    return "\n".join(lines)


def inspect_all_dataset_configs(
    config_dir: str | Path = "configs/datasets",
    output_dir: str | Path = "outputs/metrics/dataset_inspection",
    *,
    max_rows: int | None = None,
    chunksize: int = 100_000,
) -> list[dict[str, Any]]:
    """Inspect every dataset config in a directory."""
    config_paths = sorted(Path(config_dir).glob("*.yaml"))
    return [
        inspect_dataset_config(path, output_dir, max_rows=max_rows, chunksize=chunksize)
        for path in config_paths
    ]


def _read_csv_chunks(
    file_path: Path,
    *,
    max_rows: int | None,
    chunksize: int,
):
    rows_remaining = max_rows
    reader = pd.read_csv(file_path, chunksize=chunksize, low_memory=False)

    for chunk in reader:
        if rows_remaining is None:
            yield chunk
            continue

        if rows_remaining <= 0:
            break

        yielded = chunk.head(rows_remaining)
        rows_remaining -= len(yielded)
        yield yielded


def _resolve_column_name(columns: list[str], configured_name: str | None) -> str | None:
    if configured_name is None:
        return None
    if configured_name in columns:
        return configured_name

    normalized = configured_name.strip().casefold()
    for column in columns:
        if column.strip().casefold() == normalized:
            return column

    return None


def _aggregate_file_summaries(files: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    non_finite_counts: Counter[str] = Counter()
    all_columns: set[str] = set()

    for file_summary in files:
        label_counts.update(file_summary["label_counts"])
        missing_counts.update(file_summary["missing_counts"])
        non_finite_counts.update(file_summary["non_finite_numeric_counts"])
        all_columns.update(file_summary["columns"])

    return {
        "file_count": len(files),
        "rows_scanned": sum(file_summary["rows_scanned"] for file_summary in files),
        "unique_column_count": len(all_columns),
        "label_counts": dict(label_counts.most_common()),
        "columns_with_whitespace": sorted(
            {
                column
                for file_summary in files
                for column in file_summary["columns_with_whitespace"]
            }
        ),
        "duplicate_columns": sorted(
            {column for file_summary in files for column in file_summary["duplicate_columns"]}
        ),
        "duplicate_columns_after_strip": sorted(
            {
                column
                for file_summary in files
                for column in file_summary["duplicate_columns_after_strip"]
            }
        ),
        "empty_columns": sorted(
            {column for file_summary in files for column in file_summary["empty_columns"]}
        ),
        "columns_with_missing_values": sorted(
            column for column, count in missing_counts.items() if count > 0
        ),
        "numeric_columns_with_non_finite_values": sorted(
            column
            for column, count in non_finite_counts.items()
            if count > 0 and not math.isnan(float(count))
        ),
    }


def _duplicates(values: list[str]) -> set[str]:
    counts = Counter(values)
    return {value for value, count in counts.items() if count > 1}


def _counter_table(name: str, values: dict[str, int]) -> list[str]:
    lines = [
        f"| {name} | Count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| `{label}` | {count} |" for label, count in values.items())
    return lines
