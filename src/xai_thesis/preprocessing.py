"""Train/test split, subsampling and the shared preprocessing pipeline."""

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils import resample


def split_features_target(data: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target label."""
    features = data.drop(columns=[label_column])
    target = data[label_column]
    return features, target


def stratified_random_split(
    data: pd.DataFrame,
    label_column: str,
    *,
    test_fraction: float = 0.3,
    random_state: int = 42,
    drop_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train/test with a stratified random partition (fixed seed).

    Stratifying on label_column keeps every class's share of the rows the same in both
    partitions. test_fraction defaults to 0.3 (a 70/30 split).

    drop_columns lists helper columns that must not reach the model (e.g. the "source_file"
    loading tag); names not present in the frame are ignored.
    """
    present_drop = [column for column in drop_columns if column in data.columns]
    features = data.drop(columns=present_drop)
    train, test = train_test_split(
        features,
        test_size=test_fraction,
        random_state=random_state,
        stratify=features[label_column],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def stratified_subsample(
    features: pd.DataFrame,
    target: pd.Series,
    target_size: int,
    *,
    min_per_class: int = 1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Subsample to roughly target_size rows, proportionally per class, while every class keeps
    at least min_per_class rows (or all of them, if it has fewer). Raising a class to the minimum
    adds rows on top of the proportional shares, so the result can exceed target_size slightly.
    """
    counts = target.value_counts()
    proportions = counts / counts.sum()

    resampled_features = []
    resampled_target = []
    for label, count in counts.items():
        desired = max(min_per_class, round(proportions[label] * target_size))
        n_samples = min(count, desired)
        mask = target == label
        x_sampled, y_sampled = resample(
            features.loc[mask], target.loc[mask], n_samples=n_samples, replace=False, random_state=random_state
        )
        resampled_features.append(x_sampled)
        resampled_target.append(y_sampled)

    return pd.concat(resampled_features), pd.concat(resampled_target)


def build_preprocessing_pipeline(
    features: pd.DataFrame,
    categorical_columns: Sequence[str],
    *,
    min_frequency: float = 0.01,
) -> ColumnTransformer:
    """Create the shared preprocessing pipeline for numeric and categorical features."""
    numeric_columns = [column for column in features.columns if column not in categorical_columns]

    # Numeric features: median imputation, then min-max scaling to [0, 1]. Both are fitted on the
    # training data; test values outside the training range are not clipped.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )
    # Categorical features: one-hot encoding. Categories below min_frequency of the training rows
    # share one "infrequent" bucket per column, and categories seen only in the test data are
    # assigned to that bucket (or to an all-zero vector if the column has none).
    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, list(categorical_columns)),
        ]
    )
