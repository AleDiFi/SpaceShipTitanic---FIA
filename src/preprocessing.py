"""Preprocessing (POST Feature Engineering).

Questo modulo va eseguito DOPO `src/feature_engineering.py`.
Può quindi assumere che alcune colonne "derivate" esistano già, ad esempio:
- Group, Group_size, Solo
- Cabin_deck, Cabin_number, Cabin_side, Cabin_region*
- Surname, Family_size
- Age_group
- Expenditure, No_spending

Obiettivo:
- combinare train+test (senza target) per imputazioni coerenti
- imputare missing values sfruttando le feature ingegnerizzate
- separare di nuovo train/test e ritornare (train_processed, test_processed, y)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# =============================================================================
# SECTION: Configuration
# =============================================================================
@dataclass
class PreprocessConfig:
    """Configurazione del preprocessing (post-FE)."""

    target_col: str = "Transported"
    show_plots: bool = False
    # Se True, applica regole "domain-ish" della guida (es. CryoSleep -> spese = 0)
    apply_domain_rules: bool = True


# =============================================================================
# SECTION: Helpers
# =============================================================================
def _mode(series: pd.Series):
    """Ritorna la moda della serie (ignorando NaN) o NaN se vuota."""
    s = series.dropna()
    if s.empty:
        return np.nan
    return s.mode().iloc[0]


# =============================================================================
# SECTION: Dataset Combination
# =============================================================================
def combine_datasets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "Transported",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Splitta X/y dal train e concatena X e test per imputazioni globali.

    Args:
        train: DataFrame di training (contiene il target).
        test: DataFrame di test.
        target_col: Nome della colonna target.

    Returns:
        (X, y, combined) dove:
        - X è il train senza target
        - y è il target
        - combined è la concatenazione di X e test (index resettato)
    """
    if target_col not in train.columns:
        raise ValueError(f"target_col '{target_col}' non presente nel train.")

    y = train[target_col].copy()
    # Convertiamo a int se booleano
    if y.dtype == bool:
        y = y.astype(int)
    elif y.dtype == object:
        # Kaggle: Transported spesso è bool, ma in caso di stringhe "True"/"False"
        y = y.map({"True": 1, "False": 0}).astype("Int64")

    X = train.drop(columns=[target_col]).copy()
    combined = pd.concat([X, test.copy()], axis=0, ignore_index=True)

    return X, y, combined


# =============================================================================
# SECTION: Plots
# =============================================================================
def plot_missing_heatmap(df: pd.DataFrame, title: str = "Missing values heatmap") -> None:
    """Mostra una heatmap dei valori mancanti (solo colonne con NaN)."""
    na_cols = df.columns[df.isna().any()].tolist()
    if not na_cols:
        print("[Preprocessing] Nessun valore mancante da plottare.")
        return
    plt.figure(figsize=(12, 6))
    sns.heatmap(df[na_cols].isna().T, cmap="summer")
    plt.title(title)
    plt.show(block=False)


# =============================================================================
# SECTION: Imputation Rules
# =============================================================================
def impute_homeplanet(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """Imputa `HomePlanet` usando regole applicate in ordine.

    Ordine (invariato):
    1) Group (se presente) -> valore più frequente nel gruppo
    2) Cabin_deck (se presente) -> regole "alla guida"
    3) Surname (se presente) -> valore più frequente nel cognome
    4) Destination + Cabin_deck -> fallback
    """
    if "HomePlanet" not in df.columns:
        return df

    before = df["HomePlanet"].isna().sum()

    # 1) Group-based
    if "Group" in df.columns:
        grp = df.groupby(["Group", "HomePlanet"]).size().unstack().fillna(0)
        missing_idx = df.index[df["HomePlanet"].isna() & df["Group"].isin(grp.index)]
        if len(missing_idx) > 0:
            df.loc[missing_idx, "HomePlanet"] = df.loc[missing_idx, "Group"].map(
                lambda g: grp.idxmax(axis=1).get(g, np.nan)
            )

    # 2) Cabin_deck rules (come nella guida: A/B/C/T -> Europa, G -> Earth)
    if "Cabin_deck" in df.columns:
        df.loc[
            df["HomePlanet"].isna() & df["Cabin_deck"].isin(["A", "B", "C", "T"]),
            "HomePlanet",
        ] = "Europa"
        df.loc[
            df["HomePlanet"].isna() & (df["Cabin_deck"] == "G"),
            "HomePlanet",
        ] = "Earth"

    # 3) Surname-based
    if "Surname" in df.columns:
        sn = df.groupby(["Surname", "HomePlanet"]).size().unstack().fillna(0)
        missing_idx = df.index[df["HomePlanet"].isna() & df["Surname"].isin(sn.index)]
        if len(missing_idx) > 0:
            df.loc[missing_idx, "HomePlanet"] = df.loc[missing_idx, "Surname"].map(
                lambda s: sn.idxmax(axis=1).get(s, np.nan)
            )

    # 4) Destination fallback
    if "Destination" in df.columns and "Cabin_deck" in df.columns:
        # guida: se Cabin_deck != 'D' -> Earth, se 'D' -> Mars (come nel tuo testo)
        df.loc[
            df["HomePlanet"].isna() & (df["Cabin_deck"] != "D"),
            "HomePlanet",
        ] = "Earth"
        df.loc[
            df["HomePlanet"].isna() & (df["Cabin_deck"] == "D"),
            "HomePlanet",
        ] = "Mars"

    after = df["HomePlanet"].isna().sum()
    print(f"[Preprocessing] HomePlanet missing: {before} -> {after}")
    return df


def impute_destination(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """Imputa `Destination` con la moda globale (se presente e con NaN)."""
    if "Destination" not in df.columns:
        return df
    before = df["Destination"].isna().sum()
    if before == 0:
        return df
    # nella guida spesso TRAPPIST-1e è la moda
    df["Destination"] = df["Destination"].fillna(_mode(df["Destination"]))
    after = df["Destination"].isna().sum()
    print(f"[Preprocessing] Destination missing: {before} -> {after}")
    return df


def impute_surname(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """Imputa `Surname` usando il cognome più frequente nel gruppo (se possibile)."""
    if "Surname" not in df.columns or "Group" not in df.columns:
        return df

    before = df["Surname"].isna().sum()
    if before == 0:
        return df

    # Usa il cognome più frequente nel gruppo (solo per gruppi con size>1 se presente)
    if "Group_size" in df.columns:
        base = df[df["Group_size"] > 1]
    else:
        base = df

    if base.empty:
        return df

    gsn = base.groupby(["Group", "Surname"]).size().unstack().fillna(0)
    missing_idx = df.index[df["Surname"].isna() & df["Group"].isin(gsn.index)]
    if len(missing_idx) > 0:
        df.loc[missing_idx, "Surname"] = df.loc[missing_idx, "Group"].map(
            lambda g: gsn.idxmax(axis=1).get(g, np.nan)
        )

    after = df["Surname"].isna().sum()
    print(f"[Preprocessing] Surname missing: {before} -> {after}")
    return df


def impute_boolean_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Imputa colonne booleane/categoriche usando la moda e prova a fare cast a bool."""
    for c in cols:
        if c not in df.columns:
            continue
        # CryoSleep/VIP spesso sono bool o object; imputiamo con moda e riportiamo a bool dove possibile
        fill = _mode(df[c])
        df[c] = df[c].fillna(fill)
        # Se dopo l'imputazione sono True/False o 0/1, cast pulito
        if df[c].dropna().isin([True, False]).all():
            df[c] = df[c].astype(bool)
    return df


def impute_age(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """Imputa `Age` usando la mediana per `Age_group` (se disponibile) e fallback globale."""
    if "Age" not in df.columns:
        return df

    before = df["Age"].isna().sum()
    if before == 0:
        return df

    # Se esiste Age_group, imputiamo per gruppo
    if "Age_group" in df.columns:
        med = df.groupby("Age_group")["Age"].median()
        df["Age"] = df["Age"].fillna(df["Age_group"].map(med))
    # Fallback globale
    df["Age"] = df["Age"].fillna(df["Age"].median())

    after = df["Age"].isna().sum()
    print(f"[Preprocessing] Age missing: {before} -> {after}")
    return df


def impute_spending(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """Imputa le spese: RoomService/FoodCourt/ShoppingMall/Spa/VRDeck.

    Se `apply_domain_rules` (ordine invariato):
    - CryoSleep == True -> spese = 0 (se NaN)
    - No_spending == 1 -> spese = 0 (se NaN)

    Fallback:
    - mediana per (Age_group, HomePlanet, VIP) quando possibile
    - altrimenti mediana globale
    """
    exp_feats = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    present = [c for c in exp_feats if c in df.columns]
    if not present:
        return df

    # Domain rules
    if cfg.apply_domain_rules:
        if "CryoSleep" in df.columns:
            for c in present:
                df.loc[df["CryoSleep"] == True, c] = df.loc[df["CryoSleep"] == True, c].fillna(0)
        if "No_spending" in df.columns:
            for c in present:
                df.loc[df["No_spending"] == 1, c] = df.loc[df["No_spending"] == 1, c].fillna(0)

    # Grouped median fill (se possibile)
    grouping_cols = [c for c in ["Age_group", "HomePlanet", "VIP"] if c in df.columns]
    if grouping_cols:
        for c in present:
            before = df[c].isna().sum()
            if before == 0:
                continue
            med = df.groupby(grouping_cols)[c].median()
            df[c] = df[c].fillna(
                df[grouping_cols].apply(
                    lambda r: med.get(tuple(r.values), np.nan),
                    axis=1,
                )
            )
            # fallback globale
            df[c] = df[c].fillna(df[c].median())
            after = df[c].isna().sum()
            print(f"[Preprocessing] {c} missing: {before} -> {after}")
    else:
        # fallback semplice
        for c in present:
            before = df[c].isna().sum()
            if before == 0:
                continue
            df[c] = df[c].fillna(df[c].median())
            after = df[c].isna().sum()
            print(f"[Preprocessing] {c} missing: {before} -> {after}")

    # Ricalcola Expenditure/No_spending se esistono (post-imputazione)
    if "Expenditure" in df.columns:
        df["Expenditure"] = df[present].sum(axis=1)
    if "No_spending" in df.columns:
        df["No_spending"] = (df[present].sum(axis=1) == 0).astype(int)

    return df


def fill_remaining_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback finale: numeriche -> median, categoriche -> mode."""
    for col in df.columns:
        if not df[col].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(_mode(df[col]))
    return df


def run_preprocessing(
    train_engineered: pd.DataFrame,
    test_engineered: pd.DataFrame,
    target_col: str = "Transported",
    show_plots: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Entry point del preprocessing (da chiamare DOPO `run_feature_engineering()`)."""
    cfg = PreprocessConfig(target_col=target_col, show_plots=show_plots)

    X, y, combined = combine_datasets(train_engineered, test_engineered, target_col=target_col)

    print("\n" + "=" * 60)
    print("SPACESHIP TITANIC - Preprocessing (POST Feature Engineering)")
    print("=" * 60)
    print(f"[INFO] Combined shape: {combined.shape}")

    if cfg.show_plots:
        plot_missing_heatmap(combined, title="Missing values (post-FE)")

    # Imputazioni guidate (ora che abbiamo feature FE disponibili)
    combined = impute_surname(combined, cfg)
    combined = impute_homeplanet(combined, cfg)
    combined = impute_destination(combined, cfg)
    combined = impute_boolean_cols(combined, cols=["CryoSleep", "VIP"])
    combined = impute_age(combined, cfg)
    combined = impute_spending(combined, cfg)

    # Fallback finale
    combined = fill_remaining_missing(combined)

    # Split back
    n_train = len(X)
    train_processed = combined.iloc[:n_train, :].copy()
    test_processed = combined.iloc[n_train:, :].copy()

    print(f"[INFO] Train processed: {train_processed.shape} | Test processed: {test_processed.shape}")
    print("[Preprocessing] Completato!\n")

    return train_processed, test_processed, y