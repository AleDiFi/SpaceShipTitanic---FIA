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

1. Caricamento dei dataset originali; successivamente si esegue l'EDA (`src/data_analysis.py`) dopo aver riservato l'holdout.
2. Suddivisione stratificata delle righe etichettate: **80% sviluppo / 20% holdout**, seed 42. L'EDA supervisionata viene effettuata **dopo lo split**, usando solo i dati di sviluppo. L'holdout non è necessariamente indipendente per gruppi.
3. Per ogni modello, ricerca degli iperparametri con `RandomizedSearchCV` sul solo set di sviluppo. Ogni fold addestra il proprio `PreprocessorePasseggeri`, seguito da codifica delle categorie, eventuale scaling e classificatore.
4. Selezione del modello in base alla **Accuracy** media di cross-validation; precision, recall e F1 restano metriche ausiliarie.
5. Una sola valutazione del modello selezionato sull'holdout e successivo addestramento finale su tutti i dati etichettati.
6. Predizione del test Kaggle e creazione di `outputs/submission.csv`.

L'esplorazione iniziale può aprire finestre grafiche: chiuderle per consentire il proseguimento. La ricerca degli iperparametri sui sei modelli può essere impegnativa.

### Output

- `outputs/model_results.csv`: risultati di selezione mediante CV.
- `outputs/holdout_results.csv`: valutazione finale sul 20% riservato alla valutazione, senza utilizzo nel tuning.\n- `outputs/holdout_confusion_matrix.csv`: falsi positivi, falsi negativi e predizioni corrette sullo stesso holdout, **solo come analisi descrittiva finale**.
- `outputs/tuning_<modello>.csv`: dettagli delle ricerche degli iperparametri.
- `outputs/submission.csv`: colonne `PassengerId` e `Transported`, pronta per il controllo su Kaggle.
- `data/processed/`: feature trasformate dal preprocessing finale e target, per ispezione.

## Controlli automatici

```bash
python -m unittest discover -s tests -v
```

I test automatizzati usano **la stessa implementazione del preprocessing della pipeline principale** e coprono la gestione delle spese mancanti (FE-01), gruppi e cognomi (FE-02), indipendenza statistica dei fold (VAL-01), imputazioni categoriche e mediana frazionaria delle cabine (EXP-02). L'esito deve essere verificato nuovamente dopo ogni aggiornamento.

## Esperimento di robustezza EXP-02

```bash
python -m src.robustness_experiment
```

Confronta **XGBoost** e **SVC** con parametri fissati dalla precedente ricerca di sviluppo. Utilizza tre semi (42, 123, 2026) e cinque fold per ciascun seme con due strategie:

- `StratifiedKFold`: preserva approssimativamente le proporzioni delle classi.
- `StratifiedGroupKFold`: evita che membri dello stesso gruppo di viaggio compaiano in training e validazione contemporaneamente.

L'esperimento usa **solo i dati di sviluppo** ed esporta `outputs/exp02_fold_results.csv` e `outputs/exp02_summary.csv`. I fold ripetuti sono correlati: la loro deviazione standard è descrittiva, non un test di significatività.

## Audit concettuale FIA e ablazione delle feature

Lo split avviene prima dell'EDA supervisionata e prima di qualsiasi preprocessing appreso. Media/deviazione standard di `StandardScaler`, imputazioni e categorie del `OneHotEncoder` sono determinate dal solo training di ogni fold. `Group` è un identificativo numerico, non una misura continua; `Surname` è una feature ad alta cardinalità. Per verificare la loro utilità senza usare l'holdout è disponibile un esperimento **opzionale**, separato dal programma principale:

```bash
python -m src.feature_ablation
```

Confronta XGBoost e SVC con iperparametri già fissati usando gli stessi cinque fold stratificati sul solo 80% di sviluppo, nella configurazione originale e rimuovendo `Group`, `Surname`, o entrambe. Esporta `outputs/feature_ablation_folds.csv` e `outputs/feature_ablation_summary.csv`. Il codice non cambia automaticamente il modello finale, e la sua utilità non è ancora stata quantificata: **non dichiarare incrementi di Accuracy senza esecuzione**. Lo stesso holdout è stato consultato in EXP-01; non va riutilizzato per scegliere fra nuove feature. Per un confronto veramente nuovo e indipendente servirebbero dati aggiuntivi.

L'EDA precedente era stata effettuata anche sulle etichette di tutto il training Kaggle: la nuova disposizione riduce questo rischio per esecuzioni future, ma non annulla retroattivamente l'osservazione delle etichette dell'holdout già avvenuta nel percorso sperimentale. `StratifiedGroupKFold` resta una verifica distinta di generalizzazione a gruppi non visti, non il separatore dell'holdout ordinario.

## Risultati disponibili e limiti

Nell'esecuzione EXP-01, **XGBoost** ha ottenuto Accuracy CV **0,809175** e Accuracy holdout **0,809086**. **SVC** ha ottenuto Accuracy CV **0,808024**. Sono punteggi locali, **non** punteggi della leaderboard Kaggle.

La validazione per gruppi valuta uno scenario diverso rispetto alla validazione stratificata ordinaria. Il preprocessing CV non usa le etichette dell'holdout né le informazioni del test Kaggle durante l'apprendimento delle statistiche. Alcune regole, tra cui imputazione delle spese durante CryoSleep e proxy di parentela basati sui cognomi, sono ipotesi applicative e non relazioni certe.

Le motivazioni di tutte le modifiche e i risultati sperimentali sono registrati in [docs/REVIEW_DECISIONS.md](docs/REVIEW_DECISIONS.md).

## Organizzazione

- `main.py`: pipeline completa e generazione della submission.
- `src/preprocessing.py`: **unico modulo operativo** per feature engineering e imputazione, appresi esclusivamente dai dati di training di ogni fold.
- `src/model.py`: classificatori, pipeline scikit-learn, tuning e training.
- `src/predict.py`: predizioni e creazione CSV Kaggle.
- `src/robustness_experiment.py`: esperimento di stabilità.\n- `src/feature_ablation.py`: esperimento opzionale di ablazione per gruppi di caratteristiche e indicatori dei dati mancanti.
- `tests/`: test unitari e di regressione.
- `docs/REVIEW_DECISIONS.md`: registro tecnico del processo di revisione.

## Design pattern richiesto dal corso

Il Feature Engineering usa effettivamente lo **Strategy Pattern**: `FeatureTransformer` definisce l'interfaccia astratta, mentre `GroupTransformer`, `SurnameTransformer`, `CabinTransformer`, `AgeGroupTransformer` e `SpendingTransformer` implementano le singole strategie. `FeatureEngineeringPipeline` applica le strategie nell'ordine scelto. Il tutto è contenuto nel **solo** `src/preprocessing.py`, senza duplicati. `PreprocessorePasseggeri` coordina le strategie e impara le statistiche di imputazione esclusivamente dal training fold; mantiene l'interfaccia scikit-learn `fit/transform`. Non è il Factory Pattern: la factory crea oggetti, la strategia incapsula comportamenti sostituibili.

L'EDA in `src/data_analysis.py` è richiamata da `main.py` **prima** della suddivisione e del preprocessing. Le statistiche di fit, invece, non vengono apprese dai fold di validazione.

## Architettura lineare

`main.py` → `src/model.py` → `src/preprocessing.py` (feature engineering + valori mancanti, per fold) → codifica/scaling → classificatore → `src/predict.py`. Non esistono moduli alternativi di feature engineering o preprocessing nella struttura corrente del branch di pulizia.

## Stato

La struttura CLEAN-01 è stata integrata in `main` con test automatici superati. Le modifiche dell'audit FIA sono in un branch di revisione e richiedono ulteriori controlli e approvazione prima del merge.

## Revisione finale delle feature e nomi del codice

Il modulo operativo rimane **`src/preprocessing.py`**. La classe che prepara i passeggeri si chiama **`PreprocessorePasseggeri`**: implementa `fit()` e `transform()` come richiesto da scikit-learn. Il termine storico «fold-safe» descrive il comportamento corretto in validazione, **non** è più parte del nome della classe. Le trasformazioni rimangono organizzate secondo lo **Strategy Pattern** (`FeatureTransformer`, strategie concrete, `FeatureEngineeringPipeline`). I commenti e le spiegazioni nel codice sono in italiano, mentre gli identificativi delle colonne Kaggle, delle API e dei file prodotti mantengono i nomi tecnici.

Lo script `python -m src.feature_ablation` valuta, sul solo sviluppo, la rimozione delle feature derivate dalle spese, delle regioni della cabina, dei conteggi di gruppo e famiglia, delle fasce d'età, nonché l'introduzione di `Spending_missing` e `Cabin_missing`. Questi ultimi indicatori sono **facoltativi e disattivati nella pipeline principale**: prima di utilizzarli occorre un confronto misurato e una decisione esplicita. Nessuna nuova Accuracy o modifica definitiva delle feature viene presunta dai soli test unitari.
