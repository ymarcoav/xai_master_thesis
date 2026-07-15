from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.utils import resample


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


def balance_classes(
    features: pd.DataFrame, target: pd.Series, *, random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Resample every class to the median class count (training-time rebalancing).

    Some models (e.g. MLPClassifier) have no class_weight/sample_weight support, so they
    can't be rebalanced the way the rest of this project's models are. This is the resampling
    equivalent for those: classes below the median are oversampled with replacement, classes
    above it are undersampled, so no class is duplicated all the way up to the majority
    class's size (risky when the rarest classes have only a handful of rows) nor left as
    imbalanced as the raw data.
    """
    counts = target.value_counts()
    target_count = int(counts.median())

    resampled_features = []
    resampled_target = []
    for label, count in counts.items():
        mask = target == label
        x_resampled, y_resampled = resample(
            features.loc[mask],
            target.loc[mask],
            n_samples=target_count,
            replace=count < target_count,
            random_state=random_state,
        )
        resampled_features.append(x_resampled.reset_index(drop=True))
        resampled_target.append(y_resampled.reset_index(drop=True))

    # ignore_index + reset_index throughout: oversampling with replacement duplicates original
    # row indices, so without this a later .loc lookup on a shuffled index would match every
    # duplicate of a given label instead of the single intended row.
    x_combined = pd.concat(resampled_features, ignore_index=True)
    y_combined = pd.concat(resampled_target, ignore_index=True)
    shuffled_positions = x_combined.sample(frac=1, random_state=random_state).index
    x_balanced = x_combined.loc[shuffled_positions].reset_index(drop=True)
    y_balanced = y_combined.loc[shuffled_positions].reset_index(drop=True)
    return x_balanced, y_balanced


def stratified_subsample(
    features: pd.DataFrame,
    target: pd.Series,
    target_size: int,
    *,
    min_per_class: int = 1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Subsample down to roughly target_size rows, proportionally per class, while
    guaranteeing every class keeps at least min_per_class rows (or all of them, if it has
    fewer). Plain proportional stratified sampling can whittle an already-rare class (e.g. 5
    total rows) down below what a downstream cv-fold split needs; this puts a floor under it.
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
