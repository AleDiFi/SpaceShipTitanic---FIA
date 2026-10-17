# Spaceship Titanic — Fondamenti di Intelligenza Artificiale

Progetto universitario di **classificazione binaria** per la competizione Kaggle *Spaceship Titanic*: prevedere, sulla base delle caratteristiche di ciascun passeggero, il valore booleano `Transported`.

## Ambiente e installazione

Richiede Python e le dipendenze in `requirements.txt`. Dalla cartella principale del repository:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
pip install -r requirements.txt
```

Inserire i file ufficiali della competizione in `data/raw/train.csv` e `data/raw/test.csv`. Il dataset di training deve contenere la colonna `Transported`.

## Pipeline principale

```bash
python main.py
```

La procedura esegue:

1. Caricamento dati ed esplorazione iniziale (`src/data_analysis.py`).
2. Suddivisione stratificata del training Kaggle: **80% sviluppo / 20% holdout indipendente**, seed 42.
3. Per ogni modello, ricerca degli iperparametri con `RandomizedSearchCV` sul solo set di sviluppo. Ogni fold addestra il proprio `FoldSafePreprocessor`, seguito da codifica delle categorie, eventuale scaling e classificatore.
4. Selezione del modello in base alla **Accuracy** media di cross-validation; precision, recall e F1 restano metriche ausiliarie.
5. Una sola valutazione del modello selezionato sull'holdout e successivo addestramento finale su tutti i dati etichettati.
6. Predizione del test Kaggle e creazione di `outputs/submission.csv`.

L'esplorazione iniziale può aprire finestre grafiche: chiuderle per consentire il proseguimento. La ricerca degli iperparametri sui sei modelli può essere impegnativa.

### Output

- `outputs/model_results.csv`: risultati di selezione mediante CV.
- `outputs/holdout_results.csv`: valutazione finale indipendente sul 20% riservato.
- `outputs/tuning_<modello>.csv`: dettagli delle ricerche degli iperparametri.
- `outputs/submission.csv`: colonne `PassengerId` e `Transported`, pronta per il controllo su Kaggle.
- `data/processed/`: feature trasformate dal preprocessing finale e target, per ispezione.

## Controlli automatici

```bash
python -m unittest discover -s tests -v
```

I test coprono la gestione delle spese mancanti (FE-01), gruppi e cognomi (FE-02), indipendenza statistica dei fold (VAL-01), imputazioni categoriche e mediana frazionaria delle cabine (EXP-02). L'esito deve essere verificato nuovamente dopo ogni aggiornamento.

## Esperimento di robustezza EXP-02

```bash
python -m src.robustness_experiment
```

Confronta **XGBoost** e **SVC** con parametri fissati dalla precedente ricerca di sviluppo. Utilizza tre semi (42, 123, 2026) e cinque fold per ciascun seme con due strategie:

- `StratifiedKFold`: preserva approssimativamente le proporzioni delle classi.
- `StratifiedGroupKFold`: evita che membri dello stesso gruppo di viaggio compaiano in training e validazione contemporaneamente.

L'esperimento usa **solo i dati di sviluppo** ed esporta `outputs/exp02_fold_results.csv` e `outputs/exp02_summary.csv`. I fold ripetuti sono correlati: la loro deviazione standard è descrittiva, non un test di significatività.

## Risultati disponibili e limiti

Nell'esecuzione EXP-01, **XGBoost** ha ottenuto Accuracy CV **0,809175** e Accuracy holdout **0,809086**. **SVC** ha ottenuto Accuracy CV **0,808024**. Sono punteggi locali, **non** punteggi della leaderboard Kaggle.

La validazione per gruppi valuta uno scenario diverso rispetto alla validazione stratificata ordinaria. Il preprocessing CV non usa le etichette dell'holdout né le informazioni del test Kaggle durante l'apprendimento delle statistiche. Alcune regole, tra cui imputazione delle spese durante CryoSleep e proxy di parentela basati sui cognomi, sono ipotesi applicative e non relazioni certe.

Le motivazioni di tutte le modifiche e i risultati sperimentali sono registrati in [docs/REVIEW_DECISIONS.md](docs/REVIEW_DECISIONS.md).

## Organizzazione

- `main.py`: pipeline completa e generazione della submission.
- `src/fold_preprocessing.py`: feature engineering e imputazione sicuri rispetto ai fold.
- `src/model.py`: classificatori, pipeline scikit-learn, tuning e training.
- `src/predict.py`: predizioni e creazione CSV Kaggle.
- `src/robustness_experiment.py`: esperimento di stabilità.
- `src/feature_engineering.py`, `src/preprocessing.py`: trasformazioni precedenti mantenute per verifiche e compatibilità; **non** costituiscono il percorso di training di `main.py`.
- `tests/`: test unitari e di regressione.
- `docs/REVIEW_DECISIONS.md`: registro tecnico del processo di revisione.

## Stato di integrazione

Questo branch è una **versione di revisione**. L'integrazione in `main` richiede verifica dei test e approvazione esplicita.
