from pathlib import Path # Per la gestione dei percorsi dei file
import matplotlib.pyplot as plt # Per la creazione di grafici
import numpy as np # Per operazioni numeriche
import pandas as pd # Per la manipolazione dei dati
import seaborn as sns # Per la visualizzazione dei dati

# Impostazioni di visualizzazione delle figure con Seaborn 
sns.set_theme(style="whitegrid")

# Definizione dei percorsi predefiniti per i dataset
BASE_DIR = Path(__file__).resolve().parents[1] # Percorso base del progetto
# Percorsi predefiniti per i file di train e test nel caso di non suddivsione in raw e processed
DEFAULT_TRAIN_PATH = BASE_DIR / "data" / "train.csv"
DEFAULT_TEST_PATH = BASE_DIR / "data" / "test.csv"

# Funzioni per l'analisi esplorativa dei dati (EDA)
def load_datasets(train_path: Path = DEFAULT_TRAIN_PATH, test_path: Path = DEFAULT_TEST_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carica i dataset di train e test e li restituisce.

    Se i file non sono presenti in data/train.csv e data/test.csv,
    prova automaticamente i percorsi data/raw/train.csv e data/raw/test.csv.
    """

    # Risoluzione flessibile dei percorsi (fallback su data/raw)
    train_path = Path(train_path)
    test_path = Path(test_path)

    # Percorsi alternativi nel caso di suddivisione in raw e processed
    alt_train = BASE_DIR / "data" / "raw" / "train.csv"
    alt_test = BASE_DIR / "data" / "raw" / "test.csv"

    if not train_path.exists() and alt_train.exists():
        print(f"[INFO] train.csv non trovato in {train_path}. Uso percorso alternativo: {alt_train}")
        train_path = alt_train

    if not test_path.exists() and alt_test.exists():
        print(f"[INFO] test.csv non trovato in {test_path}. Uso percorso alternativo: {alt_test}")
        test_path = alt_test

    # Se ancora mancanti, generiamo un errore chiaro
    if not train_path.exists():
        raise FileNotFoundError(
            f"File di train non trovato. Percorsi controllati: {BASE_DIR / 'data' / 'train.csv'} e {alt_train}"
        )
    if not test_path.exists():
        raise FileNotFoundError(
            f"File di test non trovato. Percorsi controllati: {BASE_DIR / 'data' / 'test.csv'} e {alt_test}"
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

# Funzioni di analisi e visualizzazione
def print_dataset_shapes(train: pd.DataFrame, test: pd.DataFrame) -> None:
    print("Train set shape:", train.shape)
    print("Test set shape:", test.shape)

# Preview delle prime n righe di un DataFrame
def preview_head(df: pd.DataFrame, label: str, n: int = 5) -> None:
    print(f"\n{label} preview (prime {n} righe):")
    print(df.head(n))

# Report dei valori mancanti in un DataFrame
def report_missing_values(df: pd.DataFrame, label: str) -> None:
    print(f"\nMissing values in {label} set:\n", df.isnull().sum())

# Report delle righe duplicate in un DataFrame
def report_duplicates(df: pd.DataFrame, label: str) -> None:
    duplicated_rows = df.duplicated().sum() 
    percentage = float(np.round(100 * duplicated_rows / len(df), 1)) if len(df) else 0.0 # Calcolo percentuale di righe duplicate
    print(f"Number of duplicate rows in {label} set: {duplicated_rows} ({percentage}%)")

# Report della cardinalità delle feature in un DataFrame
def report_cardinality(df: pd.DataFrame) -> None:
    print("\nFeature cardinality (valori unici per colonna):\n", df.nunique())

# Report dei tipi di dato delle feature in un DataFrame
def report_dtypes(df: pd.DataFrame) -> None:
    print("\nFeature data types:\n", df.dtypes)

# Visualizzazioni
def plot_target_distribution(train: pd.DataFrame, target_col: str = "Transported") -> None:
    if target_col not in train.columns:
        print(f"Colonna '{target_col}' non presente nel dataset: salto il target distribution plot.")
        return
    
    # Grafico a torta della distribuzione del target
    target_counts = train[target_col].value_counts(dropna=False) # Conta i valori unici, inclusi i NaN (dropna=False serve a mantenere i NaN)
    colors = sns.color_palette("pastel", len(target_counts)) # Palette di colori pastello perchè mi piacciono così

    # Creazione effettiva del grafico a torta
    plt.figure(figsize=(6, 6)) 
    target_counts.plot.pie(
        explode=[0.05] * len(target_counts), # Sposta leggermente le fette per evidenziarle
        autopct="%1.1f%%", # Mostra le percentuali sulle fette
        shadow=True,
        textprops={"fontsize": 14},
        colors=colors,
    ).set_title("Target Distribution", fontsize=18)
    plt.ylabel("") # Rimuove l'etichetta y predefinita
    plt.show()

# Visualizza la distribuzione dell'età
def plot_age_distribution(train: pd.DataFrame, feature: str = "Age", target_col: str = "Transported") -> None:
    if feature not in train.columns:
        print(f"Colonna '{feature}' non presente nel dataset: salto l'istogramma dell'età.")
        return

    # Creazione dell'istogramma
    plt.figure(figsize=(8, 6))
    sns.histplot(data=train, x=feature, hue=target_col if target_col in train.columns else None, binwidth=1, kde=True)
    plt.title("Age Distribution by Transported Status", fontsize=16)
    plt.xlabel("Age (years)", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    if target_col in train.columns:
        plt.legend(title=target_col, fontsize=12)
    plt.show()

# Visualizza le distribuzioni delle spese
def plot_expense_distributions(
    train: pd.DataFrame,
    features: list[str] | None = None,
    target_col: str = "Transported",
    zoom_ylim: int = 100,
) -> None:
    features = features or ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    valid_features = [feature for feature in features if feature in train.columns]

    if not valid_features:
        print("Nessuna delle feature di spesa specificate è presente nel dataset.")
        return

    fig, axes = plt.subplots(len(valid_features), 2, figsize=(12, 4 * len(valid_features)))
    axes = np.atleast_2d(axes)

    # Plot per ogni feature
    for idx, feature in enumerate(valid_features):
        sns.histplot(data=train, x=feature, hue=target_col if target_col in train.columns else None, kde=True, ax=axes[idx, 0])
        axes[idx, 0].set_title(f"{feature} distribution")

        sns.histplot(data=train, x=feature, hue=target_col if target_col in train.columns else None, kde=True, ax=axes[idx, 1])
        axes[idx, 1].set_ylim(0, zoom_ylim)
        axes[idx, 1].set_title(f"{feature} distribution (zoom)")

    fig.tight_layout()
    plt.show()

# Visualizza le feature categoriche
def plot_categorical_features(train: pd.DataFrame, features: list[str] | None = None, target_col: str = "Transported") -> None:
    features = features or ["HomePlanet", "CryoSleep", "Destination", "VIP"]
    valid_features = [feature for feature in features if feature in train.columns]

    if not valid_features:
        print("Nessuna delle feature categoriche specificate è presente nel dataset.")
        return

    # Crea la figura e gli assi
    fig, axes = plt.subplots(len(valid_features), 1, figsize=(10, 4 * len(valid_features)))
    if len(valid_features) == 1:
        axes = [axes]

    for ax, feature in zip(axes, valid_features):
        sns.countplot(data=train, x=feature, hue=target_col if target_col in train.columns else None, ax=ax)
        ax.set_title(feature)
        ax.set_xlabel("")
        ax.set_ylabel("Count")

    fig.tight_layout()
    plt.show()

# Preview delle feature qualitative
# Visualizza le prime n righe delle feature qualitative specificate
def preview_qualitative_features(train: pd.DataFrame, features: list[str] | None = None, n: int = 5) -> None:
    features = features or ["PassengerId", "Cabin", "Name"]
    valid_features = [feature for feature in features if feature in train.columns]

    if not valid_features:
        print("Nessuna delle feature qualitative specificate è presente nel dataset.")
        return

    print(f"\nPreview delle feature qualitative (prime {n} righe):")
    print(train[valid_features].head(n))

# Funzione principale per eseguire l'EDA completa
# Esegue in sequenza tutte le fasi dell'EDA e organizza le visualizzazioni.
def run_full_analysis(train_path: Path = DEFAULT_TRAIN_PATH, test_path: Path = DEFAULT_TEST_PATH) -> None:
    """Esegue in sequenza tutte le fasi dell'EDA e organizza le visualizzazioni."""
    train, test = load_datasets(train_path, test_path)

    print_dataset_shapes(train, test)
    preview_head(train, "Train set")
    preview_head(test, "Test set")

    report_missing_values(train, "train")
    report_missing_values(test, "test")

    report_duplicates(train, "train")
    report_duplicates(test, "test")

    report_cardinality(train)
    report_dtypes(train)

    plot_target_distribution(train)
    plot_age_distribution(train)
    plot_expense_distributions(train)
    plot_categorical_features(train)
    preview_qualitative_features(train)


if __name__ == "__main__":
    run_full_analysis()