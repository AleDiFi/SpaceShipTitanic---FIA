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
|--------|-------------|
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
