from collections.abc import Sequence
from datetime import datetime

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
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
    data: pd.DataFrame,
    source_column: str,
    label_column: str,
    test_fraction: float = 0.2,
    *,
    timestamp_column: str | None = None,
    timestamp_format: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by holding out the last rows of each (source file, label) group as test.

    Attack categories in CIC-IDS2017 and DAPT2020 are concentrated in specific source files
    and, within a file, in a contiguous time burst. Splitting per source file alone can drop
    a whole class into just one side of the split, so this groups by (source file, label)
    instead: every class stays represented in both splits, while later rows within each
    group are still the ones held out for test.

    Without timestamp_column, "later rows" means whatever order the raw CSV happens to have -
    a positional split, not a verified temporal one. This was checked empirically against
    DAPT2020's real Timestamp column: raw row order is NOT chronological (25-40% of adjacent
    rows go backwards in time in the files checked), so a positional split there does not
    actually hold out later traffic. When timestamp_column is given, each group is sorted by
    it first, so the held-out rows are genuinely the most recent ones. Two pandas/numpy
    segfaults were hit and worked around while building this (both reproducible on this
    project's installed pandas 3.0.4 / numpy 2.4.6): pandas' vectorized date parser crashes on
    this project's %d/%m/%Y %I:%M:%S %p timestamps (worked around with Python's own
    datetime.strptime instead), and sort_values on a datetime64-dtype column crashes outright
    regardless of how it was parsed (worked around by sorting on each timestamp's numeric
    .timestamp() value instead of the datetime object itself - same ordering, plain float64
    dtype). The timestamp column itself is dropped from the output either way, alongside
    source_column - it is only ever used to order the split, never as a model feature.
    """
    if timestamp_column is not None:
        parsed_timestamps = data[timestamp_column].map(
            lambda value: datetime.strptime(value, timestamp_format).timestamp()
        )
        sort_column = f"_{timestamp_column}_sort_key"
        data = data.assign(**{sort_column: parsed_timestamps})

    train_parts = []
    test_parts = []
    for _, group in data.groupby([source_column, label_column], sort=False):
        if timestamp_column is not None:
            group = group.sort_values(sort_column, kind="stable")
        cutoff = int(len(group) * (1 - test_fraction))
        train_parts.append(group.iloc[:cutoff])
        test_parts.append(group.iloc[cutoff:])

    drop_columns = [source_column] if timestamp_column is None else [source_column, timestamp_column, sort_column]
    train = pd.concat(train_parts).drop(columns=drop_columns).reset_index(drop=True)
    test = pd.concat(test_parts).drop(columns=drop_columns).reset_index(drop=True)
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
) -> ColumnTransformer:
    """Create the shared preprocessing pipeline for numeric and categorical features."""
    numeric_columns = [column for column in features.columns if column not in categorical_columns]

    # Min-max scaling, matching the reference methodology this project now follows. Verified
    # this doesn't reproduce the RobustScaler regression found earlier (which broke LinearSVC
    # outright, since a few heavy-tailed columns' IQR-scaled output ballooned to 10^5-10^6x
    # other features): the tree ensemble is scale-invariant regardless (splits depend on
    # value ordering, not magnitude) and the retained neural network took only a moderate,
    # non-catastrophic hit on the most heavily-skewed dataset - LinearSVC is no longer in the
    # catalog (see models.py), so its specific fragility to this choice no longer applies.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
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
