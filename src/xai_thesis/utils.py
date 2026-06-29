from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
