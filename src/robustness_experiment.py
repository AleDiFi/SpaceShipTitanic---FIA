"""EXP-02: confronto fra XGBoost e SVC sul solo set di sviluppo.

Gli iperparametri sono stati fissati nella precedente ricerca con cross-validation.
L'holdout non viene utilizzato per selezionare parametri né per effettuare stime.
I risultati descrivono la robustezza, non costituiscono un nuovo test indipendente.

Esecuzione: python -m src.robustness_experiment
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

# Parametri fissati in EXP-01: non riselezionarli usando l'holdout.
FIXED_PARAMS = {
    "xgboost": {
        "subsample": 0.8, "reg_lambda": 1.0, "n_estimators": 700,
        "min_child_weight": 1, "max_depth": 4, "learning_rate": 0.03,
        "colsample_bytree": 0.9,
    },
    "svc": {"kernel": "rbf", "gamma": 0.01, "class_weight": "balanced", "C": 10.0},
}


def development_partition(data: pd.DataFrame):
    """Riproduce la suddivisione stratificata 80/20 dei dati originali."""
    X = data.drop(columns=["Transported"])
    y = data["Transported"].astype(int)
    X_dev, _, y_dev, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    return X_dev.reset_index(drop=True), y_dev.reset_index(drop=True)


def evaluate_robustness(X: pd.DataFrame, y: pd.Series, seeds=SEEDS, folds=FOLDS):
    """Restituisce i risultati per fold; ogni fold riapprende il preprocessing."""
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
                # Verifica esplicita della separazione dei gruppi nella validazione per gruppi.
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
    """Calcola media e deviazione standard descrittive di fold correlati."""
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
    print("\nRisultati EXP-02 salvati in outputs/. L'holdout non è stato utilizzato.")


if __name__ == "__main__":
    main()
