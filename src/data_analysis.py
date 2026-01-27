"""Utility di Exploratory Data Analysis (EDA) per SpaceShip Titanic.

Questo modulo contiene funzioni di report testuale e visualizzazione.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# =============================================================================
# SECTION: Global Plot Settings
# =============================================================================
sns.set_theme(style="whitegrid")


# =============================================================================
# SECTION: Report Utilities
# =============================================================================
def print_dataset_shapes(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Stampa la shape di train e test."""
    print("Train set shape:", train.shape)
    print("Test set shape:", test.shape)


def preview_head(df: pd.DataFrame, label: str, n: int = 5) -> None:
    """Stampa le prime `n` righe di un DataFrame.

    Args:
        df: DataFrame da mostrare.
        label: Etichetta descrittiva per l'output testuale.
        n: Numero di righe da stampare.
    """
    print(f"\n{label} preview (prime {n} righe):")
    print(df.head(n))


def report_missing_values(df: pd.DataFrame, label: str) -> None:
    """Stampa il conteggio dei missing values per colonna."""
    print(f"\nMissing values in {label} set:\n", df.isnull().sum())


def report_duplicates(df: pd.DataFrame, label: str) -> None:
    """Stampa il numero (e percentuale) di righe duplicate."""
    duplicated_rows = df.duplicated().sum()
    percentage = float(np.round(100 * duplicated_rows / len(df), 1)) if len(df) else 0.0
    print(f"Number of duplicate rows in {label} set: {duplicated_rows} ({percentage}%)")


def report_cardinality(df: pd.DataFrame) -> None:
    """Stampa la cardinalità (valori unici) per colonna."""
    print("\nFeature cardinality (valori unici per colonna):\n", df.nunique())


def report_dtypes(df: pd.DataFrame) -> None:
    """Stampa i dtype delle colonne."""
    print("\nFeature data types:\n", df.dtypes)


# =============================================================================
# SECTION: Plots
# =============================================================================
def plot_target_distribution(train: pd.DataFrame, target_col: str = "Transported") -> None:
    """Mostra la distribuzione del target con un grafico a torta."""
    if target_col not in train.columns:
        print(
            f"Colonna '{target_col}' non presente nel dataset: "
            "salto il target distribution plot."
        )
        return

    target_counts = train[target_col].value_counts(dropna=False)
    colors = sns.color_palette("pastel", len(target_counts))

    plt.figure(figsize=(6, 6))
    target_counts.plot.pie(
        explode=[0.05] * len(target_counts),
        autopct="%1.1f%%",
        shadow=True,
        textprops={"fontsize": 14},
        colors=colors,
    ).set_title("Target Distribution", fontsize=18)
    plt.ylabel("")
    plt.show(block=False)


def plot_age_distribution(
    train: pd.DataFrame,
    feature: str = "Age",
    target_col: str = "Transported",
) -> None:
    """Mostra la distribuzione dell'età (con hue sul target se presente)."""
    if feature not in train.columns:
        print(f"Colonna '{feature}' non presente nel dataset: salto l'istogramma dell'età.")
        return

    plt.figure(figsize=(8, 6))
    sns.histplot(
        data=train,
        x=feature,
        hue=target_col if target_col in train.columns else None,
        binwidth=1,
        kde=True,
        palette="pastel",
    )
    plt.title("Age Distribution by Transported Status", fontsize=16)
    plt.xlabel("Age (years)", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    if target_col in train.columns:
        plt.legend(title=target_col, fontsize=12)
    plt.show(block=False)


def plot_expense_distributions(
    train: pd.DataFrame,
    target_col: str = "Transported",
    zoom_ylim: int = 100,
) -> None:
    """Mostra la distribuzione delle spese (full + zoom) per ciascuna feature."""
    valid_features = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    if not valid_features:
        print("Nessuna delle feature di spesa specificate è presente nel dataset.")
        return

    fig, axes = plt.subplots(len(valid_features), 2, figsize=(12, 4 * len(valid_features)))
    axes = np.atleast_2d(axes)

    for idx, feature in enumerate(valid_features):
        sns.histplot(
            data=train,
            x=feature,
            hue=target_col if target_col in train.columns else None,
            kde=True,
            ax=axes[idx, 0],
        )
        axes[idx, 0].set_title(f"{feature} distribution")

        sns.histplot(
            data=train,
            x=feature,
            hue=target_col if target_col in train.columns else None,
            kde=True,
            ax=axes[idx, 1],
        )
        axes[idx, 1].set_ylim(0, zoom_ylim)
        axes[idx, 1].set_title(f"{feature} distribution (zoom)")

    fig.tight_layout()
    plt.show(block=False)


def plot_categorical_features(
    train: pd.DataFrame,
    features: list[str] | None = None,
    target_col: str = "Transported",
) -> None:
    """Mostra countplot delle feature categoriche (con hue sul target se presente)."""
    features = features or ["HomePlanet", "CryoSleep", "Destination", "VIP"]
    valid_features = [feature for feature in features if feature in train.columns]

    if not valid_features:
        print("Nessuna delle feature categoriche specificate è presente nel dataset.")
        return

    fig, axes = plt.subplots(len(valid_features), 1, figsize=(10, 4 * len(valid_features)))
    if len(valid_features) == 1:
        axes = [axes]

    for ax, feature in zip(axes, valid_features):
        sns.countplot(
            data=train,
            x=feature,
            hue=target_col if target_col in train.columns else None,
            ax=ax,
        )
        ax.set_title(feature)
        ax.set_xlabel("")
        ax.set_ylabel("Count")

    fig.tight_layout()
    plt.show(block=False)


def preview_qualitative_features(
    train: pd.DataFrame,
    features: list[str] | None = None,
    n: int = 5,
) -> None:
    """Stampa una preview delle feature qualitative selezionate.

    Args:
        train: DataFrame del training set.
        features: Lista di colonne da mostrare (se None usa i default).
        n: Numero di righe da stampare.
    """
    features = features or ["PassengerId", "Cabin", "Name"]
    valid_features = [feature for feature in features if feature in train.columns]

    if not valid_features:
        print("Nessuna delle feature qualitative specificate è presente nel dataset.")
        return

    print(f"\nPreview delle feature qualitative (prime {n} righe):")
    print(train[valid_features].head(n))


# =============================================================================
# SECTION: Full EDA
# =============================================================================
def run_full_analysis(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Esegue in sequenza tutte le fasi dell'EDA e organizza le visualizzazioni.

    Args:
        train: DataFrame del training set.
        test: DataFrame del test set.
    """
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

    # Attende che l'utente chiuda tutte le finestre prima di terminare
    plt.show()