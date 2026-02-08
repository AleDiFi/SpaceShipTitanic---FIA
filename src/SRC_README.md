# Spaceship Titanic - Kaggle Challenge

In questo Readme verranno appuntate tutte le scelte tecniche prese al livello di scrittura del codice per il clean coding e per la leggibilità.

---

## Struttura del Progetto

### `main.py`

#### `load_datasets()`

```python
def load_datasets(train_path: Path | None = None, test_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
```

Funzione utilizzata per **caricare i dataset di train e test una sola volta** e passarli poi al codice di data analysis e feature engineering, evitando duplicazione del codice.

#### Definizione dei Percorsi

```python
BASE_DIR = Path(__file__).resolve().parent
train_path = BASE_DIR / "data" / "raw" / "train.csv"
test_path = BASE_DIR / "data" / "raw" / "test.csv"
```

I percorsi vengono definiti in questo modo per garantire la **portabilità e l'affidabilità del codice**, indipendentemente dal sistema operativo su cui viene eseguito (Windows, macOS, Linux) e dalla directory di lavoro corrente. Questa metodologia sfrutta il modulo `pathlib` (e in particolare la classe `Path`).

**Scomposizione della riga:**

```python
BASE_DIR = Path(__file__).resolve().parent
```

- **`Path(__file__)`**:
  - `__file__` è una variabile speciale di Python che contiene il percorso del file Python corrente
  - `Path()` lo converte in un oggetto `Path` di `pathlib`, che permette di utilizzare i metodi orientati agli oggetti per la manipolazione dei percorsi

- **`.resolve()`**:
  - Trasforma il percorso in un percorso **assoluto** (completo, che parte dalla radice del file system, es. `/home/user/...` o `C:\Users\...`)

- **`.parent`**:
  - Restituisce la directory genitore (parent directory) di un percorso
  - Alternativamente con `parents[]`:
    - `parents[0]` è la cartella immediatamente superiore (il genitore del file stesso)
    - `parents[1]` è la cartella due livelli sopra il file corrente (il nonno)

#### `main()`

```python
def main() -> None:
```

Punto di ingresso principale: carica i dati ed esegue analisi, feature engineering, preprocessing...

---

## `data_analysis.py`

### Scelte Implementative di Codice in Data Analysis

#### Type Hinting

Ho scelto di utilizzare il **Type Hinting** (Suggerimenti di Tipo) per migliorare la leggibilità, la manutenibilità e la possibilità di effettuare analisi statica del codice.

**Esempio:**

```python
def plot_age_distribution(train: pd.DataFrame, feature: str = "Age", target_col: str = "Transported") -> None:
```

**Scomposizione dei parametri:**

1. **`train: pd.DataFrame`**
   - **Nome del parametro**: `train`
   - **Suggerimento di Tipo**: `: pd.DataFrame`
     - Indica che ci si aspetta che l'argomento passato per `train` sia un oggetto di tipo `DataFrame` della libreria `pandas` (importata con alias `pd`)
   - È il set di dati che la funzione deve analizzare

2. **`feature: str = "Age"`**
   - **Nome del Parametro**: `feature`
   - **Suggerimento di Tipo**: `: str`
     - Indica che ci si aspetta una stringa (`str`). La stringa rappresenta il nome della colonna nel DataFrame che contiene i dati che si desidera tracciare
   - **Valore di Default**: `= "Age"`
     - Se non viene specificato un valore per `feature` durante la chiamata della funzione, verrà automaticamente utilizzato il valore `"Age"`

3. **`-> None`**
   - **Suggerimento di Tipo per il Valore di Ritorno**: `-> None`
     - Indica che la funzione non restituisce esplicitamente alcun valore (cioè, restituisce `None`)
     - Nel contesto di una funzione di plot, questo è tipico perché il suo scopo è visualizzare un grafico (un effetto collaterale) e non calcolare o restituire un nuovo oggetto dati

#### Creazione Istogramma

La funzione utilizzata è `sns.histplot()`, che crea un istogramma per visualizzare la distribuzione di una variabile numerica.

**Scomposizione dei parametri:**

```python
sns.histplot(data=train, x=feature, hue=target_col if target_col in train.columns else None, binwidth=1, kde=True, palette="pastel")
```

1. **`data=train`**
   - Specifica il DataFrame che contiene i dati da visualizzare
   - Indica a Seaborn di usare il DataFrame chiamato `train` per trovare le colonne specificate successivamente

2. **`x=feature`**
   - Definisce la variabile da visualizzare sull'asse delle ascisse (X)
   - Questa è la colonna numerica di cui vuoi vedere la distribuzione

3. **`hue=target_col if target_col in train.columns else None`**
   - Aggiunge una dimensione di raggruppamento (colore) al grafico
   - Operatore condizionale (ternario) che:
     - Se la colonna specificata in `target_col` esiste nel DataFrame `train`, usa quella colonna per colorare le barre dell'istogramma (dividendo la distribuzione per la categoria target)
     - Altrimenti, imposta `hue=None`, creando un singolo istogramma senza divisioni per colore
   - Garantisce che la funzione non fallisca se la colonna `target_col` non è presente nel set di dati

4. **`binwidth=1`**
   - Imposta la larghezza di ogni barra (bin) dell'istogramma
   - Specificando `binwidth=1`, ogni barra rappresenta un intervallo di ampiezza 1
   - Nel contesto dell'età, significa che le barre raggrupperanno gli individui di 0 anni, 1 anno, 2 anni, ecc.

5. **`kde=True`**
   - Abilita la visualizzazione della **stima della densità del kernel** (Kernel Density Estimate)
   - Disegna una linea morbida e continua sopra le barre dell'istogramma
   - Questa linea è una stima continua della distribuzione dei dati, utile per visualizzare la forma generale della distribuzione e per confrontare meglio le distribuzioni dei diversi gruppi (definiti da `hue`)

---

## `feature_engineering.py`

### Scelte Implementative di Codice in Feature Engineering

#### Lambda Function

In certi casi mi è stato utile implementare delle **lambda function** al fine di migliorare la leggibilità del codice e per renderlo più snello. Le ho usate quando dovevano implementare codice semplice (1 riga) e dove so che avrei usato quel codice una sola volta.

**Esempio** (`feature_engineering.py` riga 49):

```python
df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
```

- `lambda x:` → riceve ogni valore della colonna `PassengerId` come `x`
- `x.split('_')[0]` → divide la stringa sul carattere `_` e prende il primo elemento

---

### Strategy Pattern per Feature Engineering

#### Motivazione della Scelta

Ho implementato il **Strategy Pattern** per strutturare il processo di feature engineering in modo modulare e scalabile. Questa scelta architetturale offre diversi vantaggi:

- ✅ **Separazione delle responsabilità**: Ogni trasformazione è incapsulata in una classe dedicata, rendendo il codice più leggibile e manutenibile
- ✅ **Facilità di testing**: Ogni strategia può essere testata indipendentemente dalle altre
- ✅ **Estensibilità**: È possibile aggiungere nuove trasformazioni senza modificare il codice esistente (Open/Closed Principle)
- ✅ **Riusabilità**: Le strategie possono essere facilmente riutilizzate in contesti diversi o combinate in modi differenti

#### Struttura del Pattern

Il pattern è composto da **tre elementi principali**:

##### 1️⃣ Classe Astratta Base (`FeatureTransformer`)

```python
class FeatureTransformer(ABC):
    """Classe base astratta per le strategie di trasformazione delle feature."""
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applica la trasformazione al DataFrame."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Restituisce il nome descrittivo della trasformazione."""
        pass
```

Questa classe definisce l'**interfaccia comune** che tutte le strategie concrete devono implementare:

- `transform()`: esegue la trasformazione effettiva sul DataFrame
- `get_name()`: fornisce un identificativo leggibile per logging e debugging

##### 2️⃣ Strategie Concrete

Ogni trasformazione specifica eredita da `FeatureTransformer` e implementa la propria logica:

| Classe | Descrizione |
| ------ | ----------- |
| **`AgeGroupTransformer`** | Crea fasce di età discrete dalla feature continua `Age` |
| **`ExpenditureTransformer`** | Calcola le spese totali e identifica i passeggeri senza spese |
| **`GroupTransformer`** | Estrae informazioni sui gruppi di viaggio (dimensione gruppo, passeggeri solitari) |
| **`CabinLocationTransformer`** | Estrae componenti strutturate dalla feature `Cabin` (deck, numero, lato) |

**Esempio di implementazione:**

```python
class AgeGroupTransformer(FeatureTransformer):
    def get_name(self) -> str:
        return "Age Grouping"
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Age_group'] = 'Unknown'
        df.loc[df['Age']<=12, 'Age_group'] = 'Age_0-12'
        df.loc[(df['Age']>12) & (df['Age']<18), 'Age_group'] = 'Age_13-17'
        # ...logica di trasformazione...
        return df
```

##### Pipeline di Esecuzione (`FeatureEngineeringPipeline`)

La classe pipeline coordina l'**applicazione sequenziale** di tutte le strategie:

```python
class FeatureEngineeringPipeline:
    def __init__(self, transformers: list[FeatureTransformer]):
        self.transformers = transformers
    
    def fit_transform(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        for transformer in self.transformers:
            print(f"  -> Applicando: {transformer.get_name()}")
            train = transformer.transform(train)
            test = transformer.transform(test)
        return train, test
```

#### Utilizzo

La funzione `run_feature_engineering()` istanzia la pipeline con le strategie desiderate:

```python
def run_feature_engineering(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pipeline = FeatureEngineeringPipeline([
        AgeGroupTransformer(),
        ExpenditureTransformer(),
        GroupTransformer(),
        CabinLocationTransformer()
    ])
    return pipeline.fit_transform(train, test)
```

#### Vantaggi Pratici

1. **Modificabilità**: Per disabilitare una trasformazione basta rimuoverla dalla lista della pipeline
2. **Ordine esplicito**: L'ordine di applicazione delle trasformazioni è chiaro e facilmente modificabile
3. **Logging consistente**: Il metodo `get_name()` garantisce output di debug uniformi
4. **Compatibilità**: Il pattern mantiene la stessa interfaccia esterna (`run_feature_engineering()`), garantendo retrocompatibilità con il resto del codice

---

## `preprocessing.py`

### Obiettivo e Posizionamento nella Pipeline

Il modulo `preprocessing.py` si occupa del **preprocessing post Feature Engineering**, quindi va eseguito **dopo** `feature_engineering.py`.

L’idea è:

- sfruttare le feature ingegnerizzate (es. `Group`, `Age_group`, `No_spending`, `Cabin_deck`, `Surname`, ecc.) per fare imputazioni più intelligenti;
- evitare divergenze tra train e test facendo imputazioni su un dataset unificato (train senza target + test);
- restituire infine `train_processed`, `test_processed` e `y` (target separato).

Questo approccio riduce il rischio di **data leakage** (il target non entra nelle imputazioni) e garantisce coerenza delle trasformazioni.

---

### Configurazione con `dataclass` (`PreprocessConfig`)

Nel codice viene utilizzata una `dataclass` come contenitore per la configurazione:

```python
@dataclass
class PreprocessConfig:
  target_col: str = "Transported"
  show_plots: bool = False
  apply_domain_rules: bool = True
```

**Perché è utile:**

- raggruppa in un singolo oggetto tutte le opzioni che controllano il comportamento del preprocessing;
- evita parametri “sparsi” in troppe funzioni;
- rende immediato capire cosa è “configurabile”.

**Significato dei parametri principali:**

1. **`target_col`**: nome della colonna target (per default `Transported`).
2. **`show_plots`**: abilita/disabilita le visualizzazioni (es. heatmap dei missing).
3. **`apply_domain_rules`**: se `True`, applica regole “domain-ish” tipiche del challenge (es. passeggero in `CryoSleep` ⇒ spese a 0).

---

### Scelta Implementativa: combinare train e test per imputazioni coerenti

La funzione chiave per questa scelta è `combine_datasets()`:

```python
def combine_datasets(
  train: pd.DataFrame,
  test: pd.DataFrame,
  target_col: str = "Transported",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
```

Restituisce una tupla `(X, y, combined)`:

- **`X`** = train senza la colonna target
- **`y`** = target
- **`combined`** = concatenazione di `X` e `test` (con `ignore_index=True`)

#### Robustezza sulla conversione del target

All’interno della funzione viene gestita anche la conversione del target in formato numerico (0/1) in modo robusto:

- se `y` è booleano (`bool`) ⇒ `astype(int)`
- se `y` è `object` (tipicamente stringhe `"True"`/`"False"`) ⇒ mapping a 0/1 e cast a intero

Questo evita errori nel training e mantiene il pipeline “pulito” anche se il CSV cambia tipo in lettura.

---

### Helper `_mode()`

```python
def _mode(series: pd.Series):
  s = series.dropna()
  if s.empty:
    return np.nan
  return s.mode().iloc[0]
```

È un helper dedicato a calcolare la **moda** ignorando i NaN.

- `dropna()` rimuove i missing
- se la serie è vuota ⇒ ritorna `np.nan`
- altrimenti usa `series.mode()` e prende il primo valore

Serve per imputare variabili categoriche/booleane con una regola semplice e riutilizzabile.

---

### Visualizzazione: `plot_missing_heatmap()`

La funzione crea una heatmap dei valori mancanti, **solo per le colonne che contengono NaN**, per evitare grafici “rumorosi”:

```python
na_cols = df.columns[df.isna().any()].tolist()
```

Poi usa:

```python
sns.heatmap(df[na_cols].isna().T, cmap="summer")
```

- `df[na_cols].isna()` crea una matrice booleana (True se missing)
- `.T` trasposta: colonne come righe, più leggibile
- `cmap="summer"` è solo una scelta estetica

---

### Regole di Imputazione (scelte e ordine)

Il preprocessing applica una serie di funzioni di imputazione, con priorità/ordine esplicito.
L’ordine non è casuale: alcune feature ingegnerizzate (es. `Surname`, `Group`, `Cabin_deck`) migliorano la qualità delle imputazioni successive.

#### 1) `impute_surname()`

Obiettivo: imputare `Surname` sfruttando `Group` (e, se disponibile, `Group_size`).

Idea:

- se esiste `Group_size`, si usa come base solo `Group_size > 1` per evitare di “inventare” cognomi da passeggeri singoli;
- poi per ogni `Group` si trova il cognome più frequente e lo si assegna ai missing di quel gruppo.

#### 2) `impute_homeplanet()` (multi-step)

La funzione imputa `HomePlanet` usando più regole **in sequenza**, applicate solo dove `HomePlanet` è ancora mancante.

Ordine implementato:

1. **Group-based**: se esiste `Group`, usa il pianeta più frequente nel gruppo

2. **Cabin_deck rules**: regole “alla guida”

    ```python
    df.loc[df["HomePlanet"].isna() & df["Cabin_deck"].isin(["A", "B", "C", "T"]), "HomePlanet"] = "Europa"
    df.loc[df["HomePlanet"].isna() & (df["Cabin_deck"] == "G"), "HomePlanet"] = "Earth"
    ```

3. **Surname-based**: usa la moda del cognome (famiglie con lo stesso cognome spesso condividono `HomePlanet`)

4. **Destination fallback** (in combinazione con deck): assegna un valore di fallback basato su `Destination` + `Cabin_deck`

In più, la funzione logga il conteggio dei missing prima/dopo:

```python
print(f"[Preprocessing] HomePlanet missing: {before} -> {after}")
```

#### 3) `impute_destination()`

Imputazione semplice con la moda globale:

```python
df["Destination"] = df["Destination"].fillna(_mode(df["Destination"]))
```

È una scelta pragmatica: `Destination` ha pochi valori e spesso una moda dominante.

#### 4) `impute_boolean_cols()` (CryoSleep/VIP)

Questa funzione:

- imputata con la moda
- tenta di riportare a `bool` quando i valori risultano puliti:

```python
if df[c].dropna().isin([True, False]).all():
  df[c] = df[c].astype(bool)
```

È utile per mantenere tipi coerenti e per facilitare la fase successiva di encoding.

#### 5) `impute_age()`

Imputa `Age` con una strategia a due livelli:

1) se esiste `Age_group`, usa la **mediana per gruppo**
2) fallback: mediana globale

```python
med = df.groupby("Age_group")["Age"].median()
df["Age"] = df["Age"].fillna(df["Age_group"].map(med))
df["Age"] = df["Age"].fillna(df["Age"].median())
```

La mediana è scelta perché robusta agli outlier.

#### 6) `impute_spending()` (RoomService/FoodCourt/ShoppingMall/Spa/VRDeck)

Questa è una parte centrale perché le spese sono fortemente correlate con `CryoSleep` e con la probabilità di `Transported`.

**Feature coinvolte:**

```python
exp_feats = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
```

**Domain rules (se `apply_domain_rules=True`):**

- se `CryoSleep == True` ⇒ spese mancanti impostate a 0
- se `No_spending == 1` ⇒ spese mancanti impostate a 0

Queste regole si applicano **solo sui NaN** tramite `fillna(0)` in selezione:

```python
df.loc[df["CryoSleep"] == True, c] = df.loc[df["CryoSleep"] == True, c].fillna(0)
```

**Imputazione statistica (fallback):**

- calcola la mediana per gruppo su colonne disponibili tra `Age_group`, `HomePlanet`, `VIP`
- se non basta, usa la mediana globale

Infine ricalcola:

- `Expenditure` = somma delle spese
- `No_spending` = indicatore (1 se somma = 0)

Questo garantisce coerenza tra feature derivate e valori imputati.

---

### Fallback finale: `fill_remaining_missing()`

Se resta qualche missing dopo le regole principali:

- colonne numeriche ⇒ mediana
- colonne non numeriche ⇒ moda

È una rete di sicurezza per assicurare che il modello riceva un dataset completo.

---

### Entry Point: `run_preprocessing()`

```python
def run_preprocessing(
  train_engineered: pd.DataFrame,
  test_engineered: pd.DataFrame,
  target_col: str = "Transported",
  show_plots: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
```

Questa funzione è l’**interfaccia unica** del modulo e coordina tutti i passaggi:

1) costruisce `cfg`
2) chiama `combine_datasets()` per ottenere `combined`
3) (opzionale) crea grafici missing
4) applica imputazioni in ordine deterministico
5) applica `fill_remaining_missing()`
6) splitta indietro in `train_processed` e `test_processed`
7) ritorna `(train_processed, test_processed, y)`

---

## `model.py`

### Obiettivo del Modulo

Il file `model.py` contiene tutta la logica per:

- confrontare più modelli su uno split holdout (`evaluate_models`)
- selezionare il best model
- effettuare il fit finale del best model su tutto il train (`fit_best_model`)
- salvare i risultati della selezione (`save_model_results`)

---

### Configurazione con `dataclass` immutabile (`ModelConfig`)

```python
@dataclass(frozen=True)
class ModelConfig:
  target_col: str = "Transported"
  test_size: float = 0.2
  random_state: int = 42
  n_jobs: int = -1
  model_names: tuple[str, ...] = (...)
```

**Note:**

- `frozen=True` rende l’oggetto immutabile: utile per evitare modifiche accidentali alla configurazione.
- `model_names` permette di scegliere quali modelli includere nel confronto senza toccare la logica.

---

### Conversione robusta del target: `_ensure_numeric_target()`

Il target può arrivare come:

- `bool` ⇒ viene convertito a `0/1`
- `object` (stringhe) ⇒ mapping `{"True": 1, "False": 0}`
- altro numerico ⇒ cast a `int`

Questo evita comportamenti inconsistenti tra dataset/letture diverse.

---

### Separazione feature numeriche e categoriche: `_split_feature_types()`

```python
categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numeric_cols = [c for c in df.columns if c not in categorical_cols]
```

Scelta pratica: in questo progetto `bool` viene trattato come categoriale, così viene gestito dal OneHotEncoder (evitando ambiguità e preservando l’interpretazione discreta).

---

### Preprocessor comune: `build_preprocessor()` con `ColumnTransformer`

Il preprocessing per i modelli è centralizzato in una funzione che produce un `ColumnTransformer`:

- **categoriche** ⇒ `OneHotEncoder(handle_unknown="ignore")`
- **numeriche** ⇒ `StandardScaler()` se richiesto, altrimenti `passthrough`

```python
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)
```

**Perché `handle_unknown="ignore"` è importante:**

- in test possono comparire categorie mai viste in train
- ignorarle evita crash in fase di predizione

---

### Model Zoo: `build_model_zoo()`

Il file costruisce un dizionario `nome -> estimator` (tutti modelli scikit-learn, più XGBoost opzionale):

- `LogisticRegression`
- `KNeighborsClassifier`
- `SVC`
- `RandomForestClassifier`
- `GaussianNB`
- `XGBClassifier` (solo se importabile)

La scelta di importare XGBoost in un `try/except` rende il progetto:

- robusto se l’ambiente non lo ha installato
- compatibile con workflow dove si vuole eseguire solo sklearn

Inoltre il codice filtra lo zoo in base a `cfg.model_names` per rispettare la configurazione.

---

### Regole “di compatibilità” per pipeline: scaling e matrice densa

Nel file ci sono due regole esplicite:

```python
def _model_requires_scaling(model_name: str) -> bool:
  return model_name in {"logistic_regression", "knn", "svc"}

def _model_requires_dense_matrix(model_name: str) -> bool:
  return model_name in {"gaussian_nb"}
```

Motivazioni:

- modelli basati su distanza/ottimizzazione (KNN, SVC, Logistic) beneficiano molto dello scaling;
- `GaussianNB` non accetta matrici sparse ⇒ il `OneHotEncoder` deve produrre una matrice densa (`sparse_output=False`).

---

### Pipeline per modello: `_build_pipeline_for_model()`

La pipeline standard è:

1) `preprocessor` (`ColumnTransformer`)
2) `classifier` (estimator scelto)

```python
pipeline = Pipeline(
  steps=[
    ("preprocessor", preprocessor),
    ("classifier", estimator),
  ]
)
```

Questa scelta è fondamentale per:

- evitare data leakage (fit dei trasformatori solo sul train nello split)
- garantire che in predizione si applichino esattamente le stesse trasformazioni

---

### Model Selection su holdout: `evaluate_models()`

```python
def evaluate_models(
  X: pd.DataFrame,
  y: pd.Series,
  cfg: ModelConfig | None = None,
) -> tuple[pd.DataFrame, str]:
```

Aspetti rilevanti:

1. **Split stratificato**

```python
train_test_split(..., stratify=y_num)
```

La stratificazione mantiene proporzioni simili delle classi tra train/valid.

1. **Valutazione multi-metrica**

- accuracy
- precision
- recall
- f1

Il codice calcola le metriche su `X_valid` e salva i risultati in una lista di dizionari.

1. **Selezione del best model**

I risultati vengono convertiti in `DataFrame`, ordinati per `accuracy` e il primo viene scelto come migliore:

```python
results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
best_model_name = str(results_df.loc[0, "model"])
```

La stampa della top-5 rende immediata l’ispezione dei risultati.

---

### Salvataggio risultati: `save_model_results()`

Salva i risultati in CSV e crea automaticamente le cartelle necessarie:

```python
output_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_path, index=False)
```

È una scelta “clean” perché evita errori quando la directory `outputs/` non esiste.

---

### Fit finale: `fit_best_model()`

Questa funzione:

1) ricostruisce lo zoo
2) seleziona l’estimator associato a `best_model_name`
3) costruisce la pipeline corretta (scaling/dense se necessario)
4) fa fit su **tutto** `X` e `y`

Restituisce la pipeline finale (`Pipeline`) pronta per la predizione su test.

---

## `predict.py`

### Obiettivo del Modulo (Predict)

`predict.py` gestisce l’ultima parte del flusso:

- predire sul `test_processed`
- convertire nel formato richiesto da Kaggle
- costruire la submission con le colonne corrette
- salvare la submission in CSV

---

### Helper: `_to_bool_predictions()`

Kaggle richiede `Transported` come booleano (`True`/`False`).

```python
def _to_bool_predictions(pred: Iterable[int | bool]) -> pd.Series:
  s = pd.Series(pred)
  if s.dtype == bool:
    return s
  return s.astype(int).astype(bool)
```

Gestisce due casi:

- se il modello produce già bool ⇒ ritorna direttamente
- se produce 0/1 ⇒ doppio cast `int -> bool` per normalizzare

---

### Predizione: `predict_test()`

```python
def predict_test(model: Pipeline, test_processed: pd.DataFrame) -> pd.Series:
```

Scelte di leggibilità:

- stampa un header di sezione
- logga la shape del test
- produce predizioni e le converte a bool

La funzione restituisce la serie booleana `preds_bool`.

---

### Costruzione submission: `build_submission()`

```python
def build_submission(passenger_ids: pd.Series, transported_preds: pd.Series) -> pd.DataFrame:
```

Prima valida che le lunghezze coincidano:

```python
if len(passenger_ids) != len(transported_preds):
  raise ValueError(...)
```

Poi crea il DataFrame esattamente nel formato Kaggle:

- `PassengerId`
- `Transported`

---

### Salvataggio: `save_submission()`

```python
def save_submission(submission: pd.DataFrame, output_path: Path) -> Path:
```

Anche qui viene creata la directory se non esiste:

```python
output_path.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(output_path, index=False)
```

Restituire `Path` è comodo perché il chiamante può loggare/riusare il percorso in modo consistente.
