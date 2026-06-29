import pandas as pd


def split_features_target(data: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target label."""
    features = data.drop(columns=[label_column])
    target = data[label_column]
    return features, target


def build_preprocessing_pipeline():
    """Create the shared preprocessing pipeline."""
    # Add imputers, encoders, scalers, and feature selectors here.
    raise NotImplementedError
