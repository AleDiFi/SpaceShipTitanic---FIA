"""Predizione e creazione della submission per SpaceShip Titanic.

Questo modulo gestisce:
- predizione sul test processato
- conversione nel formato richiesto da Kaggle
- creazione e salvataggio del file di submission


"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.pipeline import Pipeline


# =============================================================================
# SECTION: Helpers
# =============================================================================
def _to_bool_predictions(pred: Iterable[int | bool]) -> pd.Series:
    """Converte predizioni 0/1 o bool nel formato bool richiesto da Kaggle."""
    s = pd.Series(pred)
    if s.dtype == bool:
        return s
    return s.astype(int).astype(bool)


# =============================================================================
# SECTION: Predict
# =============================================================================
def predict_test(model: Pipeline, test_processed: pd.DataFrame) -> pd.Series:
    """Esegue la predizione sul test set processato."""
    print("\n" + "=" * 60)
    print("SPACESHIP TITANIC - Predict")
    print("=" * 60)
    print(f"[Predict] Shape test_processed: {test_processed.shape}")
    print("[Predict] Predizione sul test set...")
    preds = model.predict(test_processed)
    preds_bool = _to_bool_predictions(preds)
    print("[Predict] Predizione completata.")
    print("=" * 60 + "\n")
    return preds_bool


# =============================================================================
# SECTION: Submission
# =============================================================================
def build_submission(
    passenger_ids: pd.Series,
    transported_preds: pd.Series,
) -> pd.DataFrame:
    """Costruisce il DataFrame di submission con le colonne richieste."""
    if len(passenger_ids) != len(transported_preds):
        raise ValueError("[Predict] passenger_ids e predizioni hanno lunghezze diverse.")

    submission = pd.DataFrame(
        {
            "PassengerId": passenger_ids.values,
            "Transported": transported_preds.values,
        }
    )
    print(f"[Predict] Submission costruita: {submission.shape}")
    return submission


# =============================================================================
# SECTION: Persistence
# =============================================================================
def save_submission(submission: pd.DataFrame, output_path: Path) -> Path:
    """Salva la submission su disco."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"[Predict] Submission salvata in: {output_path}")
    return output_path
