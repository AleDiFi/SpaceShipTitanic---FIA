"""FIA review: assess whether Group and Surname features help, without holdout reuse.

Run separately: python -m src.feature_ablation
This diagnostic NEVER selects parameters from the holdout.
"""
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.model import ModelConfig, _build_pipeline_for_model, build_model_zoo
from src.robustness_experiment import FIXED_PARAMS, development_partition


class DropColumns(BaseEstimator, TransformerMixin):
    """Remove selected derived features between feature engineering and encoding."""

    def __init__(self, columns=()):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=list(self.columns), errors="raise")


FEATURE_SETS = {
    "baseline": (),
    "no_group_id": ("Group",),
    "no_surname": ("Surname",),
    "no_group_id_or_surname": ("Group", "Surname"),
}


def assess_features(X, y, models=("svc", "xgboost"), folds=5, seed=42):
    """Evaluate fixed models on identical development folds across feature sets."""
    cfg = ModelConfig(model_names=tuple(models), n_jobs=1)
    zoo = build_model_zoo(cfg)
    missing = set(models) - set(zoo)
    if missing:
        raise RuntimeError(f"Models unavailable: {sorted(missing)}")
    rows = []
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        train_X, valid_X = X.iloc[train_idx], X.iloc[valid_idx]
        train_y, valid_y = y.iloc[train_idx], y.iloc[valid_idx]
        for model_name in models:
            for variant, columns in FEATURE_SETS.items():
                estimator = clone(zoo[model_name])
                estimator.set_params(**FIXED_PARAMS[model_name])
                pipeline = _build_pipeline_for_model(train_X, model_name, estimator)
                if columns:
                    # Place the selector AFTER fold-safe FE and BEFORE One-Hot Encoding.
                    pipeline.steps.insert(1, ("drop_features", DropColumns(columns=columns)))
                pipeline.fit(train_X, train_y)
                prediction = pipeline.predict(valid_X)
                rows.append({
                    "model": model_name, "variant": variant, "fold": fold,
                    "accuracy": accuracy_score(valid_y, prediction),
                })
    return pd.DataFrame(rows)


def summarize_ablation(rows):
    summary = rows.groupby(["model", "variant"], as_index=False).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
    )
    baseline = summary[summary["variant"] == "baseline"][["model", "accuracy_mean"]]
    baseline = baseline.rename(columns={"accuracy_mean": "baseline_accuracy"})
    summary = summary.merge(baseline, on="model")
    summary["delta_vs_baseline"] = summary["accuracy_mean"] - summary["baseline_accuracy"]
    return summary.sort_values(["model", "variant"])


def main():
    root = Path(__file__).resolve().parent.parent
    data = pd.read_csv(root / "data" / "raw" / "train.csv")
    X, y = development_partition(data)
    results = assess_features(X, y)
    summary = summarize_ablation(results)
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    results.to_csv(output / "feature_ablation_folds.csv", index=False)
    summary.to_csv(output / "feature_ablation_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("Only the development partition was used. Do not infer holdout performance.")


if __name__ == "__main__":
    main()
