"""Training e selezione modello per SpaceShip Titanic.

Contiene:
- confronto modelli su holdout (model selection)
- confronto modelli con cross-validation
- fine-tuning iperparametri del modello selezionato
- fit finale del modello migliore
- salvataggio risultati
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class ModelConfig:
    """Configurazione per confronto modelli e fit finale."""

    target_col: str = "Transported"
    test_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
    cv_folds: int = 5
    cv_selection_metric: str = "accuracy"
    tune_n_iter: int = 40
    tune_cv_folds: int = 5
    tune_refit_metric: str = "f1"
    # Modelli da confrontare (sklearn-only; xgboost opzionale se disponibile)
    model_names: tuple[str, ...] = field(
        default_factory=lambda: (
            "logistic_regression",
            "knn",
            "svc",
            "random_forest",
            "gaussian_nb",
            "xgboost",
        )
    )


# =============================================================================
# SECTION: Helpers (Target + Feature Types)
# =============================================================================
def _ensure_numeric_target(y: pd.Series) -> pd.Series:
    """Converte il target a 0/1 in modo robusto."""
    if y.dtype == bool:
        return y.astype(int)
    if y.dtype == object:
        mapped = y.map({"True": 1, "False": 0})
        if mapped.isna().any():
            raise ValueError("[Model] Target object non mappabile a 0/1.")
        return mapped.astype(int)
    return y.astype(int)


def _split_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Individua colonne categoriche vs numeriche."""
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [c for c in df.columns if c not in categorical_cols]
    return categorical_cols, numeric_cols


# =============================================================================
# SECTION: Preprocessor
# =============================================================================
def build_preprocessor(
    categorical_cols: Iterable[str],
    numeric_cols: Iterable[str],
    *,
    scale_numeric: bool,
    sparse_output: bool,
) -> ColumnTransformer:
    """Costruisce il preprocessor comune (OHE + opzionale scaling)."""
    categorical_cols = list(categorical_cols)
    numeric_cols = list(numeric_cols)

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=sparse_output,
    )

    if scale_numeric:
        numeric_transformer: Pipeline | str = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_transformer = "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numeric_cols),
        ],
        remainder="drop",
    )
    return preprocessor


# =============================================================================
# SECTION: Model Zoo + Rules
# =============================================================================
def build_model_zoo(cfg: ModelConfig) -> dict[str, object]:
    """Crea il dizionario nome -> estimator, rispettando i vincoli del repo."""
    zoo: dict[str, object] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            n_jobs=cfg.n_jobs,
            solver="lbfgs",
        ),
        "knn": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "svc": SVC(
            kernel="rbf",
            C=5.0,
            gamma="scale",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        ),
        "gaussian_nb": GaussianNB(),
    }

    # XGBoost: opzionale, ma è già in requirements.txt.
    try:
        from xgboost import XGBClassifier  # type: ignore

        zoo["xgboost"] = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            reg_lambda=1.0,
        )
        print("[Model] XGBoost disponibile: incluso nel confronto.")
    except Exception:
        print("[Model] XGBoost non disponibile: salto 'xgboost'.")

    # Filtra zoo in base ai nomi richiesti
    requested = set(cfg.model_names)
    filtered = {name: est for name, est in zoo.items() if name in requested}
    missing = requested.difference(filtered.keys())
    if missing:
        print(f"[Model] Modelli richiesti ma non disponibili: {sorted(missing)}")
    return filtered


def _model_requires_scaling(model_name: str) -> bool:
    """Regola: alcuni modelli beneficiano molto dello scaling."""
    return model_name in {"logistic_regression", "knn", "svc"}


def _model_requires_dense_matrix(model_name: str) -> bool:
    """GaussianNB non accetta matrici sparse."""
    return model_name in {"gaussian_nb"}


def _build_pipeline_for_model(
    X: pd.DataFrame,
    model_name: str,
    estimator: object,
) -> Pipeline:
    """Crea la pipeline (preprocessor + estimator) adatta al modello."""
    categorical_cols, numeric_cols = _split_feature_types(X)
    scale_numeric = _model_requires_scaling(model_name)
    dense_matrix = _model_requires_dense_matrix(model_name)

    # Per GaussianNB forziamo dense; per gli altri preferiamo sparse.
    preprocessor = build_preprocessor(
        categorical_cols,
        numeric_cols,
        scale_numeric=scale_numeric,
        sparse_output=not dense_matrix,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )
    return pipeline


def _supported_scoring_metrics() -> tuple[str, ...]:
    """Metriche supportate e allineate con classificazione binaria."""
    return ("accuracy", "precision", "recall", "f1")


def _build_scoring_dict() -> dict[str, object]:
    """Crea scorers coerenti con zero_division=0."""
    return {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }


def _validate_scoring_metric(metric_name: str) -> None:
    """Valida la metrica richiesta per sorting/refit."""
    supported = set(_supported_scoring_metrics())
    if metric_name not in supported:
        raise ValueError(
            f"[Model] Metrica '{metric_name}' non supportata. "
            f"Usa una tra {sorted(supported)}."
        )


# =============================================================================
# SECTION: Hyperparameter Search Spaces
# =============================================================================
def build_param_distributions(model_name: str) -> dict[str, list[object]]:
    """Restituisce gli spazi di ricerca per RandomizedSearchCV per ciascun modello."""
    spaces: dict[str, dict[str, list[object]]] = {
        "logistic_regression": {
            "classifier__C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
        },
        "knn": {
            "classifier__n_neighbors": [5, 9, 13, 17, 21, 31],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],
            "classifier__leaf_size": [20, 30, 40, 50],
        },
        "svc": {
            "classifier__C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
            "classifier__gamma": ["scale", "auto", 0.005, 0.01, 0.05, 0.1],
            "classifier__kernel": ["rbf"],
            "classifier__class_weight": [None, "balanced"],
        },
        "random_forest": {
            "classifier__n_estimators": [300, 500, 700, 900],
            "classifier__max_depth": [None, 8, 12, 20, 30],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2", None],
            "classifier__bootstrap": [True, False],
        },
        "gaussian_nb": {
            "classifier__var_smoothing": [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
        },
        "xgboost": {
            "classifier__n_estimators": [300, 500, 700, 900],
            "classifier__max_depth": [4, 6, 8, 10],
            "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "classifier__min_child_weight": [1, 3, 5],
            "classifier__reg_lambda": [0.5, 1.0, 2.0, 5.0],
        },
    }
    return spaces.get(model_name, {})


# =============================================================================
# SECTION: Model Evaluation
# =============================================================================
def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: ModelConfig | None = None,
) -> tuple[pd.DataFrame, str]:
    """Confronta più modelli su uno split holdout.

    Returns:
        - DataFrame risultati (ordinato)
        - nome del best model
    """
    cfg = cfg or ModelConfig()
    print("\n" + "=" * 60)
    print("SPACESHIP TITANIC - Model Selection (holdout)")
    print("=" * 60)

    y_num = _ensure_numeric_target(y)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y_num,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y_num,
    )

    zoo = build_model_zoo(cfg)
    if not zoo:
        raise ValueError("[Model] Nessun modello disponibile per il confronto.")

    results: list[dict[str, float | str]] = []

    for model_name, estimator in zoo.items():
        print("-" * 60)
        print(f"[Model] Training: {model_name}")

        pipeline = _build_pipeline_for_model(X_train, model_name, estimator)
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_valid)
        acc = accuracy_score(y_valid, preds)
        prec = precision_score(y_valid, preds, zero_division=0)
        rec = recall_score(y_valid, preds, zero_division=0)
        f1 = f1_score(y_valid, preds, zero_division=0)
        print(
            f"[Model] {model_name} -> "
            f"acc: {acc:.4f} | prec: {prec:.4f} | rec: {rec:.4f} | f1: {f1:.4f}"
        )

        results.append(
            {
                "model": model_name,
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
            }
        )

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False).reset_index(drop=True)
    best_model_name = str(results_df.loc[0, "model"])

    print("-" * 60)
    print("[Model] Classifica modelli (top 5):")
    print(results_df.head(5).to_string(index=False))
    print(f"[Model] Best model selezionato: {best_model_name}")
    print("=" * 60 + "\n")

    return results_df, best_model_name


# =============================================================================
# SECTION: Model Evaluation (Cross Validation)
# =============================================================================
def evaluate_models_cv(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: ModelConfig | None = None,
) -> tuple[pd.DataFrame, str]:
    """Confronta più modelli con Stratified K-Fold cross-validation."""
    cfg = cfg or ModelConfig()
    _validate_scoring_metric(cfg.cv_selection_metric)

    print("\n" + "=" * 60)
    print(f"SPACESHIP TITANIC - Model Selection (CV {cfg.cv_folds}-fold)")
    print("=" * 60)

    y_num = _ensure_numeric_target(y)
    cv = StratifiedKFold(
        n_splits=cfg.cv_folds,
        shuffle=True,
        random_state=cfg.random_state,
    )

    zoo = build_model_zoo(cfg)
    if not zoo:
        raise ValueError("[Model] Nessun modello disponibile per il confronto.")

    scoring_metrics = _supported_scoring_metrics()
    scoring = _build_scoring_dict()
    results: list[dict[str, float | str]] = []

    for model_name, estimator in zoo.items():
        print("-" * 60)
        print(f"[Model][CV] Training+Validation: {model_name}")

        pipeline = _build_pipeline_for_model(X, model_name, estimator)
        cv_scores = cross_validate(
            estimator=pipeline,
            X=X,
            y=y_num,
            scoring=scoring,
            cv=cv,
            n_jobs=cfg.n_jobs,
            return_train_score=False,
            error_score="raise",
        )

        row: dict[str, float | str] = {"model": model_name}
        for metric in scoring_metrics:
            mean_val = float(cv_scores[f"test_{metric}"].mean())
            std_val = float(cv_scores[f"test_{metric}"].std())
            row[f"{metric}_mean"] = mean_val
            row[f"{metric}_std"] = std_val

        print(
            f"[Model][CV] {model_name} -> "
            f"acc: {row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f} | "
            f"prec: {row['precision_mean']:.4f}±{row['precision_std']:.4f} | "
            f"rec: {row['recall_mean']:.4f}±{row['recall_std']:.4f} | "
            f"f1: {row['f1_mean']:.4f}±{row['f1_std']:.4f}"
        )

        results.append(row)

    sort_col = f"{cfg.cv_selection_metric}_mean"
    results_df = pd.DataFrame(results).sort_values(sort_col, ascending=False).reset_index(drop=True)
    best_model_name = str(results_df.loc[0, "model"])

    print("-" * 60)
    print("[Model][CV] Classifica modelli (top 5):")
    print(results_df.head(5).to_string(index=False))
    print(f"[Model][CV] Best model selezionato ({cfg.cv_selection_metric}): {best_model_name}")
    print("=" * 60 + "\n")

    return results_df, best_model_name


# =============================================================================
# SECTION: Hyperparameter Tuning
# =============================================================================
def fine_tune_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    cfg: ModelConfig | None = None,
) -> tuple[Pipeline, dict[str, object], pd.DataFrame]:
    """Esegue RandomizedSearchCV sul modello scelto e ritorna best pipeline + report."""
    cfg = cfg or ModelConfig()
    _validate_scoring_metric(cfg.tune_refit_metric)

    print("\n" + "=" * 60)
    print(f"[Model][Tune] Fine-tuning modello: {model_name}")
    print("=" * 60)

    y_num = _ensure_numeric_target(y)
    zoo = build_model_zoo(cfg)
    if model_name not in zoo:
        raise ValueError(f"[Model][Tune] Modello '{model_name}' non presente nel model zoo.")

    param_distributions = build_param_distributions(model_name)
    if not param_distributions:
        raise ValueError(f"[Model][Tune] Nessuno spazio di ricerca definito per '{model_name}'.")

    pipeline = _build_pipeline_for_model(X, model_name, zoo[model_name])
    cv = StratifiedKFold(
        n_splits=cfg.tune_cv_folds,
        shuffle=True,
        random_state=cfg.random_state,
    )
    scoring = _build_scoring_dict()

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=cfg.tune_n_iter,
        scoring=scoring,
        refit=cfg.tune_refit_metric,
        cv=cv,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        verbose=0,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X, y_num)

    best_pipeline: Pipeline = search.best_estimator_
    best_params: dict[str, object] = dict(search.best_params_)
    best_score = float(search.best_score_)

    rank_col = f"rank_test_{cfg.tune_refit_metric}"
    metric_cols = [f"mean_test_{metric}" for metric in _supported_scoring_metrics()]
    tuning_results_df = (
        pd.DataFrame(search.cv_results_)[[rank_col, *metric_cols, "params"]]
        .sort_values(rank_col)
        .reset_index(drop=True)
    )

    print(f"[Model][Tune] Best {cfg.tune_refit_metric}: {best_score:.4f}")
    print(f"[Model][Tune] Best params: {best_params}")
    print("=" * 60 + "\n")

    return best_pipeline, best_params, tuning_results_df


# =============================================================================
# SECTION: Persistence
# =============================================================================
def save_model_results(results_df: pd.DataFrame, output_path: Path) -> Path:
    """Salva la classifica dei modelli in CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"[Model] Risultati modelli salvati in: {output_path}")
    return output_path


# =============================================================================
# SECTION: Final Fit
# =============================================================================
def fit_best_model(
    X: pd.DataFrame,
    y: pd.Series,
    best_model_name: str,
    cfg: ModelConfig | None = None,
    best_params: dict[str, object] | None = None,
) -> Pipeline:
    """Allena su tutto il train il modello selezionato come migliore."""
    cfg = cfg or ModelConfig()
    print("[Model] Fit finale del best model su tutto il training set...")

    y_num = _ensure_numeric_target(y)
    zoo = build_model_zoo(cfg)
    if best_model_name not in zoo:
        raise ValueError(f"[Model] Best model '{best_model_name}' non presente nel model zoo.")

    estimator = zoo[best_model_name]
    if best_params:
        cleaned_params = {
            key.replace("classifier__", "", 1) if key.startswith("classifier__") else key: value
            for key, value in best_params.items()
        }
        estimator.set_params(**cleaned_params)

    pipeline = _build_pipeline_for_model(X, best_model_name, estimator)
    pipeline.fit(X, y_num)
    print(f"[Model] Fit completato: {best_model_name}")
    return pipeline
