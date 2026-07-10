from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer


def split_features_target(data: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target label."""
    features = data.drop(columns=[label_column])
    target = data[label_column]
    return features, target


def split_by_official_split_column(
    data: pd.DataFrame, split_column: str = "split"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data using a dataset's own pre-assigned train/test split column (e.g. UNSW-NB15)."""
    train = data.loc[data[split_column] == "train"].drop(columns=[split_column]).reset_index(drop=True)
    test = data.loc[data[split_column] == "test"].drop(columns=[split_column]).reset_index(drop=True)
    return train, test


def split_chronologically_by_source(
    data: pd.DataFrame, source_column: str, label_column: str, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by holding out the last rows of each (source file, label) group as test.

    Attack categories in CIC-IDS2017 and DAPT2020 are concentrated in specific source files
    and, within a file, in a contiguous time burst. Splitting per source file alone can drop
    a whole class into just one side of the split, so this groups by (source file, label)
    instead: every class stays represented in both splits, while later rows within each
    group are still the ones held out for test.
    """
    train_parts = []
    test_parts = []
    for _, group in data.groupby([source_column, label_column], sort=False):
        cutoff = int(len(group) * (1 - test_fraction))
        train_parts.append(group.iloc[:cutoff])
        test_parts.append(group.iloc[cutoff:])

    train = pd.concat(train_parts).drop(columns=[source_column]).reset_index(drop=True)
    test = pd.concat(test_parts).drop(columns=[source_column]).reset_index(drop=True)
    return train, test


def build_preprocessing_pipeline(
    features: pd.DataFrame,
    categorical_columns: Sequence[str],
    *,
    min_frequency: float = 0.01,
    random_state: int = 42,
) -> ColumnTransformer:
    """Create the shared preprocessing pipeline for numeric and categorical features."""
    numeric_columns = [column for column in features.columns if column not in categorical_columns]

    # Network-flow features (durations, IATs, byte counts) are heavily right-skewed: the
    # typical spread (IQR) can be tiny next to a legitimate but extreme outlier, which made
    # RobustScaler's output for those columns balloon to 10^5-10^6x other features and broke
    # linear-model optimization. QuantileTransformer bounds the output regardless of raw skew.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", QuantileTransformer(output_distribution="normal", random_state=random_state)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, list(categorical_columns)),
        ]
    )
