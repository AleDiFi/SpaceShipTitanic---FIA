"""Main di avvio per l'Exploratory Data Analysis, Feature Engineering ...

Questo script centralizza il caricamento dei dati e coordina l'esecuzione
delle varie fasi della pipeline.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from src.data_analysis import run_full_analysis
from src.feature_engineering import run_feature_engineering


def load_datasets(train_path: Path | None = None, test_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Carica i dataset di train e test e li restituisce.

	Prova automaticamente i percorsi data/raw/train.csv e data/raw/test.csv.
	
	Args:
		train_path: Percorso custom per il train set (opzionale)
		test_path: Percorso custom per il test set (opzionale)
		
	Returns:
		Tupla (train_df, test_df)
	"""

	# Definizione del percorso base del progetto (la cartella che contiene main.py)
	BASE_DIR = Path(__file__).resolve().parent
	train_path = BASE_DIR / "data" / "raw" / "train.csv"
	test_path = BASE_DIR / "data" / "raw" / "test.csv"

	# Risoluzione flessibile dei percorsi (fallback su data/raw)
	train_path = Path(train_path)
	test_path = Path(test_path)

	# Se ancora mancanti, generiamo un errore chiaro
	if not train_path.exists():
		raise FileNotFoundError(
			f"File di train non trovato. Percorsi controllati: {train_path}"
		)
	if not test_path.exists():
		raise FileNotFoundError(
			f"File di test non trovato. Percorsi controllati: {test_path}"
		)

	train = pd.read_csv(train_path)
	test = pd.read_csv(test_path)
	
	print(f"[INFO] Dataset caricati: train {train.shape}, test {test.shape}")
	return train, test


def main() -> None:
	"""Punto di ingresso principale: carica i dati ed esegue analisi e feature engineering."""
	print("="*60)
	print("SPACESHIP TITANIC - Pipeline di Analisi e Feature Engineering")
	print("="*60)
	
	# 1. Caricamento dati (centralizzato)
	print("\n[FASE 1] Caricamento dataset...")
	train, test = load_datasets()
	
	# 2. Analisi esplorativa
	print("\n[FASE 2] Exploratory Data Analysis...")
	run_full_analysis(train, test)
	
	# 3. Feature Engineering
	print("\n[FASE 3] Feature Engineering...")
	train_engineered, test_engineered = run_feature_engineering(train, test)
	
	print("\n" + "="*60)
	print("Pipeline completata con successo!")
	print("="*60)
	
	# Mantieni le finestre dei grafici aperte
	print("\nChiudi tutte le finestre dei grafici per terminare il programma.")
	plt.show()

	# Ritorna il dataset train trasformato in data/processed
	BASE_DIR = Path(__file__).resolve().parent
	processed_dir = BASE_DIR / "data" / "processed"
	train_engineered.to_csv(processed_dir / "train_engineered.csv", index=False) # Salva il train trasformato
	test_engineered.to_csv(processed_dir / "test_engineered.csv", index=False) # Salva il test trasformato

if __name__ == "__main__":
	main()

