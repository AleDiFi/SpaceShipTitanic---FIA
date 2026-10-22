"""Spaceship Titanic: suddivisione dei dati, preprocessing e valutazione finale.

Il set di sviluppo viene utilizzato per la ricerca degli iperparametri tramite
cross-validation. Il set di holdout è valutato dopo la scelta del modello.
L'addestramento finale usa tutte le righe etichettate prima della submission.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.data_analysis import run_full_analysis
from src.model import ModelConfig, build_model_zoo, fine_tune_model, fit_best_model, save_model_results
from src.predict import build_submission, predict_test, save_submission


# =============================================================================
# SEZIONE: Caricamento dati
# =============================================================================
def load_datasets(
    train_path: Path | None = None,
    test_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carica i dataset di train e test.

    Se non vengono passati percorsi espliciti, prova automaticamente:
    - data/raw/train.csv
    - data/raw/test.csv

    Argomenti:
        train_path: Percorso al file di training (CSV).
        test_path: Percorso al file di test (CSV).

    Restituisce:
        (train, test) come DataFrame Pandas.

    Eccezioni:
        FileNotFoundError: Se uno dei due file non esiste.
    """
    base_dir = Path(__file__).resolve().parent
    train_path = train_path or (base_dir / "data" / "raw" / "train.csv")
    test_path = test_path or (base_dir / "data" / "raw" / "test.csv")

    train_path = Path(train_path)
    test_path = Path(test_path)

    if not train_path.exists():
        raise FileNotFoundError(f"File di train non trovato: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"File di test non trovato: {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print(f"[INFO] Dataset caricati: train {train.shape}, test {test.shape}")
    return train, test


# =============================================================================
# SEZIONE: Pipeline principale
# =============================================================================
def main() -> None:
    """Esegue l'intera pipeline: EDA -> FE -> Preprocessing -> Model -> Submission."""
    print("=" * 60)
    print("SPACESHIP TITANIC - Preparazione dati, validazione e holdout")
    print("=" * 60)

    # 1) Caricamento dei dati originali
    print("\n[FASE 1] Caricamento dei dati originali...")
    train, test = load_datasets()

    # VAL-01: suddivisione delle righe originali prima dell'EDA supervisionata e del preprocessing.
    # I dati di test Kaggle non intervengono nell'apprendimento delle statistiche dei fold.
    X = train.drop(columns=["Transported"])
    y = train["Transported"].astype(int)
    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )
    # L'EDA può analizzare il target soltanto sul set di sviluppo.
    # Il test Kaggle è privo di etichette; l'EDA non osserva il target dell'holdout.
    print("\n[FASE 2] Analisi esplorativa del solo set di sviluppo...")
    eda_development = X_dev.copy()
    eda_development["Transported"] = y_dev
    run_full_analysis(eda_development, test)

    cfg = ModelConfig(
        target_col="Transported",
        cv_selection_metric="accuracy",
        tune_refit_metric="accuracy",
    )
    print("[FASE 3] Addestramento e ricerca degli iperparametri sullo sviluppo.")
    base_dir = Path(__file__).resolve().parent
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    available_models = list(build_model_zoo(cfg).keys())
    if not available_models:
        raise ValueError("[Model] Nessun modello disponibile per tuning e selezione.")

    tuning_summary: list[dict[str, object]] = []
    best_model_name = ""
    best_params: dict[str, object] | None = None
    best_score = float("-inf")

    for model_name in available_models:
        print(f"\n[INFO] Tuning modello: {model_name}")
        _, model_best_params, tuning_df = fine_tune_model(
            X_dev,
            y_dev,
            model_name=model_name,
            cfg=cfg,
        )

        top_row = tuning_df.iloc[0]
        model_score = float(top_row[f"mean_test_{cfg.tune_refit_metric}"])
        tuning_summary.append(
            {
                "model": model_name,
                "accuracy_cv": float(top_row["mean_test_accuracy"]),
                "precision_cv": float(top_row["mean_test_precision"]),
                "recall_cv": float(top_row["mean_test_recall"]),
                "f1_cv": float(top_row["mean_test_f1"]),
                "best_params": str(model_best_params),
            }
        )

        save_model_results(tuning_df, outputs_dir / f"tuning_{model_name}.csv")

        if model_score > best_score:
            best_score = model_score
            best_model_name = model_name
            best_params = model_best_params

    results_df = (
        pd.DataFrame(tuning_summary)
        .sort_values(f"{cfg.tune_refit_metric}_cv", ascending=False)
        .reset_index(drop=True)
    )
    print(f"[INFO] Modello migliore da CV e ricerca ({cfg.tune_refit_metric}): {best_model_name}")
    print("[INFO] Primi modelli per accuratezza CV:")
    print(results_df.head(5).to_string(index=False))

    save_model_results(results_df, outputs_dir / "model_results.csv")

    # L'holdout non viene utilizzato per ricerca o selezione del modello.
    print("[FASE 4] Valutazione indipendente del modello selezionato.")
    holdout_model = fit_best_model(
        X_dev, y_dev, best_model_name=best_model_name, cfg=cfg, best_params=best_params,
    )
    holdout_preds = holdout_model.predict(X_holdout)
    holdout_results = pd.DataFrame([{
        "model": best_model_name,
        "accuracy": accuracy_score(y_holdout, holdout_preds),
        "precision": precision_score(y_holdout, holdout_preds, zero_division=0),
        "recall": recall_score(y_holdout, holdout_preds, zero_division=0),
        "f1": f1_score(y_holdout, holdout_preds, zero_division=0),
    }])
    save_model_results(holdout_results, outputs_dir / "holdout_results.csv")
    # Riepilogo descrittivo degli errori: non usarlo per modificare il modello.
    matrix = confusion_matrix(y_holdout, holdout_preds, labels=[0, 1])
    matrix_df = pd.DataFrame(matrix, index=["Actual_False", "Actual_True"],
                             columns=["Pred_False", "Pred_True"])
    matrix_df.to_csv(outputs_dir / "holdout_confusion_matrix.csv")
    print("[VAL-01] Matrice di confusione sull'holdout:\n", matrix_df)
    print("[VAL-01] Risultati della valutazione sull'holdout:\n", holdout_results.to_string(index=False))

    print("[FASE 5] Addestramento finale e creazione della submission.")
    final_model = fit_best_model(
        X, y, best_model_name=best_model_name, cfg=cfg, best_params=best_params,
    )
    test_preds = predict_test(final_model, test)
    submission = build_submission(test["PassengerId"], test_preds)
    save_submission(submission, outputs_dir / "submission.csv")

    # Salvataggio delle caratteristiche prodotte dal preprocessing finale.
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    feature_step = final_model.named_steps["preparazione"]
    feature_step.transform(X).to_csv(processed_dir / "train_processed.csv", index=False)
    feature_step.transform(test).to_csv(processed_dir / "test_processed.csv", index=False)
    y.to_csv(processed_dir / "y_train.csv", index=False)

    print("\n" + "=" * 60)
    print("Pipeline completata con successo!")
    print(f"Salvati: {processed_dir / 'train_processed.csv'}")
    print(f"Salvati: {processed_dir / 'test_processed.csv'}")
    print(f"Salvati: {processed_dir / 'y_train.csv'}")
    print(f"Salvati: {outputs_dir / 'model_results.csv'}")
    print(f"Salvati: {outputs_dir / 'holdout_results.csv'}")
    print(f"Salvati: {outputs_dir / 'holdout_confusion_matrix.csv'}")
    print(f"Salvati: {outputs_dir / 'submission.csv'}")
    print("=" * 60)

    print("\nChiudi tutte le finestre dei grafici per terminare il programma.")
    plt.show()


if __name__ == "__main__":
    main()
