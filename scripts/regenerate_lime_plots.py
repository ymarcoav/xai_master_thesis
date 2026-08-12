"""One-off: regenerate just the LIME plots for already-explained experiments with the new
three-panel explain_instance_with_lime_full renderer, without recomputing SHAP.

Deliberately not a permanent script (not wired into README/CLI conventions) - this exists
only to catch up existing outputs/plots/*/lime_*.png files after explain_instance_with_lime_full
was introduced, since generate_explanations() itself was already fixed to call it for anything
explained from now on. Delete once the catch-up run is done and confirmed.
"""

from pathlib import Path

import joblib

from xai_thesis.explainability import explain_instance_with_lime_full
from xai_thesis.pipeline import (
    _CLASS_OF_INTEREST,
    _SAFE_TO_DISCRETIZE_LIME,
    _load_experiment_and_data,
    select_curated_instances,
)
from xai_thesis.utils import ensure_directory

EXPERIMENTS = [
    "cic_ids2017_gradient_boosting_baseline",
    "cic_ids2017_logistic_regression_baseline",
    "cic_ids2017_neural_network_baseline",
    "cic_ids2017_random_forest_baseline",
    "dapt2020_gradient_boosting_baseline",
    "dapt2020_logistic_regression_baseline",
    "dapt2020_neural_network_baseline",
    "dapt2020_random_forest_baseline",
    "dapt2020_svm_baseline",
    "nsl_kdd_gradient_boosting_baseline",
    "nsl_kdd_logistic_regression_baseline",
    "nsl_kdd_neural_network_baseline",
    "nsl_kdd_random_forest_baseline",
    "nsl_kdd_svm_baseline",
    "unsw_nb15_gradient_boosting_baseline",
    "unsw_nb15_logistic_regression_baseline",
    "unsw_nb15_neural_network_baseline",
    "unsw_nb15_random_forest_baseline",
    "unsw_nb15_svm_baseline",
]

for experiment_name in EXPERIMENTS:
    experiment_path = f"configs/experiments/{experiment_name}.yaml"
    experiment, dataset_config, x_train, _, x_test, y_test = _load_experiment_and_data(
        experiment_path, "configs/datasets"
    )
    random_state = experiment.get("random_state", 42)

    if dataset_config.name not in _CLASS_OF_INTEREST:
        print(f"skip {experiment_name}: no class-of-interest mapping")
        continue
    target_class, confused_with_class = _CLASS_OF_INTEREST[dataset_config.name]

    pipeline = joblib.load(Path("outputs/models") / f"{experiment_name}.joblib")
    output_dir = ensure_directory(Path("outputs/plots") / experiment_name)
    class_names = list(pipeline.named_steps["model"].classes_)

    instances = select_curated_instances(
        pipeline, x_test, y_test, target_class, confused_with_class, random_state=random_state
    )
    if not instances:
        print(f"skip {experiment_name}: no curated instances")
        continue

    for kind, instance_index in instances.items():
        out_path = explain_instance_with_lime_full(
            pipeline,
            x_train,
            x_test.loc[instance_index],
            class_names,
            output_dir / f"lime_{kind}.png",
            categorical_columns=dataset_config.categorical_columns,
            random_state=random_state,
            discretize_continuous=dataset_config.name in _SAFE_TO_DISCRETIZE_LIME,
        )
        print(f"wrote {out_path}")
