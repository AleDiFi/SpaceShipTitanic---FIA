# SpaceShip Titanic — Spiegazione Codice (File per File)

> Questo documento descrive **come funziona il codice** del progetto: pipeline completa, moduli e sezioni, e una spiegazione dedicata al livello di **Machine Learning**.

---

## 1) Panoramica della Pipeline

Il progetto implementa una pipeline end-to-end per il dataset Kaggle **Spaceship Titanic**:

1. **Load raw**: carica i CSV originali da `data/raw/`.
2. **EDA (su raw)**: report e grafici per capire struttura, missing, distribuzioni.
3. **Feature Engineering**: crea feature derivate (gruppi, cabina, spese, ecc.).
4. **Preprocessing (post-FE)**: imputazioni e pulizia finale sfruttando le feature ingegnerizzate.
5. **Model selection + fit finale + predict**: confronto modelli su holdout, scelta del migliore, training finale su tutto il train.
6. **Salvataggio output**: dataset processati e file di submission.

Il target è `Transported` (classificazione binaria).

---

## 2) main.py (Entry Point)

**File:** `main.py`

### Scopo
Orchestrare tutta la pipeline e salvare gli artefatti (processed CSV, risultati modelli, submission).

### SECTION: Dataset Loading
- **Funzione:** `load_datasets(train_path=None, test_path=None)`
  - Se non passi i path, usa:
    - `data/raw/train.csv`
    - `data/raw/test.csv`
  - Controlla esistenza file e lancia `FileNotFoundError` se manca qualcosa.
  - Ritorna `(train_df, test_df)` come `pandas.DataFrame`.

### SECTION: Main Pipeline
- **Funzione:** `main()`
  1. Carica raw (`train`, `test`).
  2. Esegue EDA con `run_full_analysis(train, test)`.
  3. Feature engineering con `run_feature_engineering(train, test)` → `train_fe`, `test_fe`.
  4. Preprocessing con `run_preprocessing(train_fe, test_fe, target_col="Transported")` → `train_processed`, `test_processed`, `y`.
  5. Model selection:
     - `cfg = ModelConfig(target_col="Transported")`
     - `evaluate_models(train_processed, y, cfg)` → `results_df`, `best_model_name`
     - salvataggio risultati in `outputs/model_results.csv`.
     - fit finale del modello migliore con `fit_best_model(...)`.
  6. Predizione e submission:
     - `predict_test(final_model, test_processed)` → predizioni
     - `build_submission(test["PassengerId"], preds)` → DataFrame submission
     - `save_submission(..., outputs/submission.csv)`.
  7. Salva anche:
     - `data/processed/train_processed.csv`
     - `data/processed/test_processed.csv`
     - `data/processed/y_train.csv`

Nota: `plt.show()` alla fine mantiene aperte le finestre dei grafici finché non le chiudi.

---

## 3) src/data_analysis.py (EDA)

**File:** `src/data_analysis.py`

### Scopo
Fornire utility per **Exploratory Data Analysis**: report testuali e visualizzazioni.

### SECTION: Global Plot Settings
- `sns.set_theme(style="whitegrid")`: imposta un tema grafico globale.

### SECTION: Report Utilities
- `print_dataset_shapes(train, test)`: stampa le shape.
- `preview_head(df, label, n=5)`: stampa le prime `n` righe.
- `report_missing_values(df, label)`: missing per colonna (`df.isnull().sum()`).
- `report_duplicates(df, label)`: duplicati e percentuale (`df.duplicated().sum()`).
- `report_cardinality(df)`: cardinalità per colonna (`df.nunique()`).
- `report_dtypes(df)`: dtype per colonna.

### SECTION: Plots
- `plot_target_distribution(train, target_col="Transported")`: pie chart del target (se presente).
- `plot_age_distribution(train, feature="Age", target_col="Transported")`: istogramma + KDE, con hue sul target.
- `plot_expense_distributions(train, target_col, zoom_ylim=100)`: istogrammi per spese (full + zoom).
- `plot_categorical_features(train, features=None, target_col="Transported")`: countplot per feature categoriche (es. HomePlanet, CryoSleep, Destination, VIP).
- `preview_qualitative_features(train, features=None, n=5)`: preview di feature testuali (PassengerId, Cabin, Name se presenti).

### SECTION: Full EDA
- `run_full_analysis(train, test)`: esegue in sequenza report + plot e conclude con `plt.show()`.

---

## 4) src/feature_engineering.py (Feature Engineering)

**File:** `src/feature_engineering.py`

### Scopo
Creare feature derivate utili al modello e applicarle **in modo coerente** a train e test.

### Pattern usato: Strategy Pattern
- `FeatureTransformer` (classe astratta):
  - `transform(df)`: applica trasformazione e ritorna DataFrame
  - `get_name()`: nome per logging

### SECTION: Transformers

#### 4.1 AgeGroupTransformer
- Crea `Age_group` da `Age` con fasce (0–12, 13–17, 18–25, 26–30, 31–50, 51+).
- Usa `Unknown` quando l’età è mancante.
- Se `Transported` esiste e `show_plots=True`, fa un countplot per fascia con hue sul target.

#### 4.2 ExpenditureTransformer
- Somma le spese in `Expenditure` = somma di:
  - RoomService, FoodCourt, ShoppingMall, Spa, VRDeck
- Crea `No_spending` = 1 se `Expenditure == 0`, altrimenti 0.
- Opzionalmente plot di distribuzione e countplot.

#### 4.3 GroupTransformer
- Estrae `Group` da `PassengerId` (parte prima di `_`).
- Calcola `Group_size` contando occorrenze per gruppo.
- Crea `Solo` = 1 se `Group_size == 1`.
- Opzionalmente plot.

#### 4.4 CabinLocationTransformer
- Gestisce `Cabin` (Deck/Number/Side):
  - sostituisce temporaneamente NaN con `Z/9999/Z` per fare split
  - crea `Cabin_deck`, `Cabin_number`, `Cabin_side`
  - ripristina i NaN originali
  - elimina `Cabin`
- Crea feature binarie `Cabin_region1..7` con soglie su `Cabin_number`.
- Opzionalmente plot.

#### 4.5 FamilySizeTransformer
- Rimpiazza temporaneamente NaN in `Name` con `Unknown Unknown`.
- Estrae `Surname` (ultima parola del nome).
- `Family_size` = frequenza del cognome.
- Ripristina NaN e gestisce outlier (`Family_size > 100`).
- Elimina `Name`.

### SECTION: Pipeline
- `FeatureEngineeringPipeline(transformers)`:
  - `fit_transform(train, test)` applica ogni trasformatore in sequenza su train e test.

### SECTION: Public Entry Point
- `run_feature_engineering(train, test)`:
  - chiede input per abilitare/disabilitare plot (`show_plots` globale)
  - costruisce la pipeline e ritorna `(train_engineered, test_engineered)`.

---

## 5) src/preprocessing.py (Preprocessing post-FE)

**File:** `src/preprocessing.py`

### Scopo
Imputare e pulire i dati **dopo** il feature engineering, usando le nuove feature per fare imputazioni più informative.

### SECTION: Configuration
- `PreprocessConfig`:
  - `target_col`: nome target
  - `show_plots`: abilita heatmap missing
  - `apply_domain_rules`: regole “di dominio” (CryoSleep/No_spending → spese = 0)

### SECTION: Helpers
- `_mode(series)`: moda ignorando NaN, o NaN se la serie è vuota.

### SECTION: Dataset Combination
- `combine_datasets(train, test, target_col)`:
  - separa `y = train[target_col]` e lo normalizza a 0/1
  - crea `X = train.drop(target)`
  - concatena `combined = concat([X, test])` per imputare train+test con le stesse statistiche.

### SECTION: Plots
- `plot_missing_heatmap(df)`: heatmap dei missing solo sulle colonne con NaN.

### SECTION: Imputation Rules

#### 5.1 impute_homeplanet
Ordine regole:
1) per `Group` → valore più frequente nel gruppo
2) per `Cabin_deck` → regole tipo guida (A/B/C/T → Europa, G → Earth)
3) per `Surname` → valore più frequente nel cognome
4) fallback con `Destination + Cabin_deck`

#### 5.2 impute_destination
- Riempie con la moda globale.

#### 5.3 impute_surname
- Se possibile, imputa `Surname` con il cognome più frequente nel `Group` (preferendo gruppi con `Group_size > 1` quando disponibile).

#### 5.4 impute_boolean_cols
- Per `CryoSleep` e `VIP`: fill con moda e cast a bool se possibile.

#### 5.5 impute_age
- Mediana per `Age_group` se presente.
- Fallback: mediana globale.

#### 5.6 impute_spending
- Applica regole di dominio (se abilitate):
  - `CryoSleep == True` → spese mancanti a 0
  - `No_spending == 1` → spese mancanti a 0
- Poi imputa per mediana per gruppi usando (se disponibili) `Age_group`, `HomePlanet`, `VIP`.
- Fallback: mediana globale.
- Ricalcola `Expenditure` e `No_spending` se esistono.

#### 5.7 fill_remaining_missing
- Numeriche → mediana
- Categoriche → moda

### Entry point
- `run_preprocessing(train_engineered, test_engineered, target_col, show_plots)`:
  - combina train+test per imputazioni coerenti
  - applica imputazioni in sequenza
  - split back in `train_processed` e `test_processed`
  - ritorna `(train_processed, test_processed, y)`.

---

## 6) src/model.py (Training + Model Selection)

**File:** `src/model.py`

### Scopo
- Confronto di più modelli su holdout (model selection)
- Fit finale del modello migliore su tutto il train
- Salvataggio risultati

### Configurazione
- `ModelConfig`:
  - `test_size`: dimensione holdout
  - `random_state`: seed
  - `n_jobs`: parallelismo
  - `model_names`: modelli da provare (include `xgboost` se disponibile)

### SECTION: Helpers (Target + Feature Types)
- `_ensure_numeric_target(y)`: normalizza target a 0/1 (gestisce bool e stringhe True/False).
- `_split_feature_types(df)`: separa colonne categoriche vs numeriche.

### SECTION: Preprocessor
- `build_preprocessor(categorical_cols, numeric_cols, scale_numeric, sparse_output)`:
  - categoriche: `OneHotEncoder(handle_unknown="ignore")`
  - numeriche: `StandardScaler` (solo se serve) o `passthrough`
  - combinazione: `ColumnTransformer`

### SECTION: Model Zoo + Rules
- `build_model_zoo(cfg)` crea un dizionario di modelli:
  - Logistic Regression, KNN, SVC, Random Forest, GaussianNB
  - opzionale: XGBoost (`XGBClassifier`) se importabile
- `_model_requires_scaling(model_name)`:
  - `logistic_regression`, `knn`, `svc` → scaling consigliato
- `_model_requires_dense_matrix(model_name)`:
  - `gaussian_nb` → richiede matrice densa (no sparse)
- `_build_pipeline_for_model(X, model_name, estimator)`:
  - costruisce `Pipeline(preprocessor -> classifier)` con parametri adeguati.

### SECTION: Model Evaluation
- `evaluate_models(X, y, cfg=None)`:
  - split stratificato train/valid
  - addestra ogni modello su train
  - valuta su valid
  - metriche: accuracy, precision, recall, f1
  - ritorna classifica `results_df` e `best_model_name`.

### SECTION: Persistence
- `save_model_results(results_df, output_path)` salva CSV con ranking modelli.

### SECTION: Final Fit
- `fit_best_model(X, y, best_model_name, cfg=None)`:
  - ricostruisce la pipeline del modello migliore
  - fa fit su tutto il train
  - ritorna la pipeline pronta per `predict()`.

---

## 7) src/predict.py (Predict + Submission)

**File:** `src/predict.py`

### Scopo
- Predizione sul test processato
- Creazione submission in formato Kaggle
- Salvataggio su disco

### SECTION: Helpers
- `_to_bool_predictions(pred)`:
  - converte predizioni in boolean per rispettare formato Kaggle.

### SECTION: Predict
- `predict_test(model, test_processed)`:
  - chiama `model.predict(test_processed)`
  - converte a bool
  - ritorna le predizioni.

### SECTION: Submission
- `build_submission(passenger_ids, transported_preds)`:
  - verifica lunghezze
  - crea DataFrame con colonne richieste: `PassengerId`, `Transported`.

### SECTION: Persistence
- `save_submission(submission, output_path)`:
  - crea directory se serve
  - salva CSV.

---

## 8) src/utils.py

**File:** `src/utils.py`

Attualmente è vuoto: nessuna utility implementata.

---

## 9) tests/test_sample.py

**File:** `tests/test_sample.py`

È un test “placeholder” che controlla solo `1 + 1 == 2`. Non valida la pipeline ML.

---

## 10) Come funziona “a livello di Machine Learning”

### 10.1 Problema
È una **classificazione binaria**: data una riga (passeggero) con feature $X$, il modello predice $y \in \{0,1\}$ (o `False/True`) per `Transported`.

### 10.2 Pipeline dei dati
- **Feature Engineering** trasforma feature raw in segnali più informativi:
  - discretizzazione età (`Age_group`)
  - aggregazione spese (`Expenditure`) e indicatori (`No_spending`)
  - struttura di viaggio (`Group`, `Group_size`, `Solo`)
  - feature cabina (deck/side/region)
  - cognome e dimensione famiglia (`Surname`, `Family_size`)
- **Preprocessing** imputa missing sfruttando queste feature e concatenando train+test per coerenza statistica.

### 10.3 Trasformazioni prima del modello (preprocessing ML)
In `src/model.py` viene costruito un preprocessor standard per sklearn:
- **Categoriche** → One-Hot Encoding (`OneHotEncoder(handle_unknown="ignore")`)
- **Numeriche** → opzionalmente scaling (`StandardScaler`) per modelli sensibili alla scala (LR/KNN/SVC)

Questo produce una matrice numerica adatta agli algoritmi ML.

### 10.4 Model selection (holdout)
- Si effettua uno split **train/valid** con `train_test_split(..., stratify=y)`.
- Ogni modello viene addestrato sul train e valutato sul valid.
- Metriche:
  - **Accuracy**: percentuale predizioni corrette
  - **Precision**: tra i predetti positivi, quanti sono veri positivi
  - **Recall**: tra i veri positivi, quanti vengono trovati
  - **F1**: media armonica precision/recall

### 10.5 Fit finale e predizione
- Dopo la selezione, il best model viene ri-addestrato su **tutto il train** (`fit_best_model`).
- Si predice sul test processato (`predict_test`).
- Le predizioni vengono convertite in bool e salvate in formato Kaggle (`build_submission`, `save_submission`).

---

## 11) Output generati
- `data/processed/train_processed.csv`
- `data/processed/test_processed.csv`
- `data/processed/y_train.csv`
- `outputs/model_results.csv`
- `outputs/submission.csv`
