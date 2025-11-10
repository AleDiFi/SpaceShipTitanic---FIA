"""Main di avvio per l'Exploratory Data Analysis."""

from src.data_analysis import run_full_analysis


def main() -> None:
	# Esegue l'EDA con i percorsi di default definiti in src/data_analysis.py
	run_full_analysis()


if __name__ == "__main__":
	main()

