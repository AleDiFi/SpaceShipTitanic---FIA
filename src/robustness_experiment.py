"""EXP-02: compare fixed XGBoost and SVC candidates on development data only.

These hyperparameters are from the completed VAL-01 development CV search.
We do NOT re-tune against the holdout, and do NOT use holdout labels here.
Results are descriptive sensitivity analyses, not a new unbiased test score.

Run: python -m src.robustness_experiment
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    StratifiedGroupKFold, StratifiedKFold, train_test_split,
)

from src.model import ModelConfig, _build_pipeline_for_model, build_model_zoo

SEEDS = (42, 123, 2026)
FOLDS = 5

# Fixed from EXP-01; do not select again using the holdout.
FIXED_PARAMS = {
    "xgboost": {
        "subsample": 0.8, "reg_lambda": 1.0, "n_estimators": 700,
        "min_child_weight": 1, "max_depth": 4, "learning_rate": 0.03,
        "colsample_bytree": 0.9,
    },
    "svc": {"kernel": "rbf", "gamma": 0.01, "class_weight": "balanced", "C": 10.0},
}


def development_partition(data: pd.DataFrame):
    """Reproduce main.py's original 80/20 stratified raw-data split."""
    X = data.drop(columns=["Transported"])
    y = data["Transported"].astype(int)
    X_dev, _, y_dev, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    return X_dev.reset_index(drop=True), y_dev.reset_index(drop=True)


def evaluate_robustness(X: pd.DataFrame, y: pd.Series, seeds=SEEDS, folds=FOLDS):
    """Return per-fold results. Every fold refits all learned preprocessing."""
    cfg = ModelConfig(model_names=("xgboost", "svc"), n_jobs=1)
    models = build_model_zoo(cfg)
    missing = set(FIXED_PARAMS) - set(models)
    if missing:
        raise RuntimeError(f"Missing required estimators: {sorted(missing)}")

    groups = X["PassengerId"].astype("string").str.split("_").str[0]
    rows = []
    for scheme in ("stratified", "stratified_group"):
        for seed in seeds:
            if scheme == "stratified":
                splitter = StratifiedKFold(
                    n_splits=folds, shuffle=True, random_state=seed,
                )
                splits = splitter.split(X, y)
            else:
                splitter = StratifiedGroupKFold(
                    n_splits=folds, shuffle=True, random_state=seed,
                )
                splits = splitter.split(X, y, groups=groups)
            for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
                # Explicitly check group separation in group-aware validation.
                overlap = set(groups.iloc[train_idx]) & set(groups.iloc[valid_idx])
                if scheme == "stratified_group" and overlap:
                    raise AssertionError("Travel group leaked into group-aware validation")
                for model_name in ("xgboost", "svc"):
                    estimator = clone(models[model_name])
                    estimator.set_params(**FIXED_PARAMS[model_name])
                    pipeline = _build_pipeline_for_model(
                        X.iloc[train_idx], model_name, estimator,
                    )
                    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
                    predictions = pipeline.predict(X.iloc[valid_idx])
                    rows.append({
                        "scheme": scheme, "seed": seed, "fold": fold,
                        "model": model_name, "n_train": len(train_idx),
                        "n_valid": len(valid_idx),
                        "shared_groups": len(overlap),
                        "accuracy": accuracy_score(y.iloc[valid_idx], predictions),
                        "f1": f1_score(y.iloc[valid_idx], predictions, zero_division=0),
                    })
    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame):
    """Mean and std across the evaluated folds; scores are correlated."""
    return (
        results.groupby(["scheme", "model"], as_index=False)
        .agg(
            folds=("accuracy", "size"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            total_shared_groups=("shared_groups", "sum"),
        )
        .sort_values(["scheme", "accuracy_mean"], ascending=[True, False])
    )


def main():
    root = Path(__file__).resolve().parent.parent
    data = pd.read_csv(root / "data" / "raw" / "train.csv")
    X_dev, y_dev = development_partition(data)
    results = evaluate_robustness(X_dev, y_dev)
    summary = summarize_results(results)
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "exp02_fold_results.csv", index=False)
    summary.to_csv(output_dir / "exp02_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nSaved EXP-02 results in outputs/. Original holdout is untouched.")


if __name__ == "__main__":
    main()
