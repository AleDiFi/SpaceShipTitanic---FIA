"""Entry point della pipeline SpaceShip Titanic.

Pipeline (ordine invariato):
1) Load raw
2) EDA (su raw)
3) Feature Engineering
4) Preprocessing (post-FE) -> imputazioni, cleaning finale
5) Model selection + fit finale + predict
6) Salvataggio in data/processed e outputs/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data_analysis import run_full_analysis
from src.feature_engineering import run_feature_engineering
from src.model import ModelConfig, evaluate_models, fit_best_model, save_model_results
from src.predict import build_submission, predict_test, save_submission
from src.preprocessing import run_preprocessing


# =============================================================================
# SECTION: Dataset Loading
# =============================================================================
def load_datasets(
	train_path: Path | None = None,
	test_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Carica i dataset di train e test.

	Se non vengono passati percorsi espliciti, prova automaticamente:
	- data/raw/train.csv
	- data/raw/test.csv

	Args:
		train_path: Percorso al file di training (CSV).
		test_path: Percorso al file di test (CSV).

	Returns:
		(train, test) come DataFrame Pandas.

	Raises:
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
# SECTION: Main Pipeline
# =============================================================================
def main() -> None:
	"""Esegue l'intera pipeline: EDA -> FE -> Preprocessing -> Model -> Submission."""
	print("=" * 60)
	print("SPACESHIP TITANIC - Pipeline (EDA -> FE -> Preprocessing)")
	print("=" * 60)

	# 1) Load raw
	print("\n[FASE 1] Caricamento dataset (raw)...")
	train, test = load_datasets()

	# 2) EDA
	print("\n[FASE 2] Exploratory Data Analysis (raw)...")
	run_full_analysis(train, test)

	# 3) Feature Engineering
	print("\n[FASE 3] Feature Engineering...")
	train_fe, test_fe = run_feature_engineering(train, test)

	# 4) Preprocessing post-FE
	print("\n[FASE 4] Preprocessing (post-FE)...")
	train_processed, test_processed, y = run_preprocessing(
		train_fe,
		test_fe,
		target_col="Transported",
		show_plots=False,  # mettere True se si vuole la heatmap dei missing
	)

	# 5) Model selection + predict
	print("\n[FASE 5] Model selection + predizione...")
	cfg = ModelConfig(target_col="Transported")
	results_df, best_model_name = evaluate_models(train_processed, y, cfg=cfg)
	print(f"[INFO] Best model da holdout: {best_model_name}")
	print("[INFO] Top modelli (holdout):")
	print(results_df.head(5).to_string(index=False))

	base_dir = Path(__file__).resolve().parent
	outputs_dir = base_dir / "outputs"
	save_model_results(results_df, outputs_dir / "model_results.csv")

	final_model = fit_best_model(train_processed, y, best_model_name=best_model_name, cfg=cfg)
	test_preds = predict_test(final_model, test_processed)
	submission = build_submission(test["PassengerId"], test_preds)

	save_submission(submission, outputs_dir / "submission.csv")

	# 6) Save processed
	base_dir = Path(__file__).resolve().parent
	processed_dir = base_dir / "data" / "processed"
	processed_dir.mkdir(parents=True, exist_ok=True)

	train_processed.to_csv(processed_dir / "train_processed.csv", index=False)
	test_processed.to_csv(processed_dir / "test_processed.csv", index=False)
	y.to_csv(processed_dir / "y_train.csv", index=False)

	print("\n" + "=" * 60)
	print("Pipeline completata con successo!")
	print(f"Salvati: {processed_dir / 'train_processed.csv'}")
	print(f"Salvati: {processed_dir / 'test_processed.csv'}")
	print(f"Salvati: {processed_dir / 'y_train.csv'}")
	print(f"Salvati: {outputs_dir / 'model_results.csv'}")
	print(f"Salvati: {outputs_dir / 'submission.csv'}")
	print("=" * 60)

	print("\nChiudi tutte le finestre dei grafici per terminare il programma.")
	plt.show()


if __name__ == "__main__":
	main()
