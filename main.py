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
from src.model import ModelConfig, build_model_zoo, fine_tune_model, fit_best_model, save_model_results
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

	# 5) Model selection con CV+tuning + predict
	print("\n[FASE 5] Model selection (CV+tuning) + predizione...")
	cfg = ModelConfig(
		target_col="Transported",
		cv_selection_metric="f1",
		tune_refit_metric="f1",
	)

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
			train_processed,
			y,
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
	print(f"[INFO] Best model da CV+tuning ({cfg.tune_refit_metric}): {best_model_name}")
	print("[INFO] Top modelli (CV+tuning):")
	print(results_df.head(5).to_string(index=False))

	save_model_results(results_df, outputs_dir / "model_results.csv")

	final_model = fit_best_model(
		train_processed,
		y,
		best_model_name=best_model_name,
		cfg=cfg,
		best_params=best_params,
	)
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
