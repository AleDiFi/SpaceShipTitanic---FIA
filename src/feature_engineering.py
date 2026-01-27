"""Feature Engineering per SpaceShip Titanic.

Questo modulo contiene trasformazioni (Strategy Pattern) applicate a train e test.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# =============================================================================
# SECTION: Base Strategy
# =============================================================================
class FeatureTransformer(ABC):
    """Classe base astratta per le strategie di trasformazione delle feature."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applica la trasformazione al DataFrame.

        Args:
            df: DataFrame da trasformare

        Returns:
            DataFrame trasformato
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Restituisce il nome descrittivo della trasformazione."""
        pass


# =============================================================================
# SECTION: Transformers
# =============================================================================
# Età
# Per la feature "Age" dividiamo in fasce di età; questo aiuta anche a gestire
# valori mancanti, ad esempio per imputazioni relative alle spese.
class AgeGroupTransformer(FeatureTransformer):
    """Trasformatore che crea fasce di età dalla feature Age."""

    def get_name(self) -> str:
        return "Age Grouping"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Inizializza come stringa per evitare warning dtype
        df['Age_group'] = 'Unknown'
        df.loc[df['Age'] <= 12, 'Age_group'] = 'Age_0-12'
        df.loc[(df['Age'] > 12) & (df['Age'] < 18), 'Age_group'] = 'Age_13-17'
        df.loc[(df['Age'] >= 18) & (df['Age'] <= 25), 'Age_group'] = 'Age_18-25'
        df.loc[(df['Age'] > 25) & (df['Age'] <= 30), 'Age_group'] = 'Age_26-30'
        df.loc[(df['Age'] > 30) & (df['Age'] <= 50), 'Age_group'] = 'Age_31-50'
        df.loc[df['Age'] > 50, 'Age_group'] = 'Age_51+'

        # Plotta solo se esiste la colonna Transported (presente solo nel train)
        if 'Transported' in df.columns and show_plots:
            plt.figure(figsize=(10, 4))
            sns.countplot(
                data=df,
                x='Age_group',
                hue='Transported',
                order=[
                    'Age_0-12',
                    'Age_13-17',
                    'Age_18-25',
                    'Age_26-30',
                    'Age_31-50',
                    'Age_51+',
                ],
            )
            plt.title('Age group distribution')
            plt.show(block=False)

        return df


# Spese
# Creiamo una nuova feature che somma tutte le spese e una binaria che indica
# se il passeggero non ha speso nulla.
class ExpenditureTransformer(FeatureTransformer):
    """Trasformatore che calcola le spese totali e identifica passeggeri senza spese."""

    def get_name(self) -> str:
        return "Total Expenditure"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        df['Expenditure'] = df[exp_feats].sum(axis=1)
        df['No_spending'] = (df['Expenditure'] == 0).astype(int)

        # Plot della distribuzione delle spese totali (solo se esiste Transported)
        if 'Transported' in df.columns and show_plots:
            fig = plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            sns.histplot(data=df, x='Expenditure', hue='Transported', bins=200)
            plt.title('Total Expenditure Distribution')
            plt.ylim([0, 200])
            plt.xlim([0, 20000])

            plt.subplot(1, 2, 2)
            sns.countplot(data=df, x='No_spending', hue='Transported')
            plt.title('No Spending Count')
            fig.tight_layout()
            plt.show(block=False)

        return df


# Gruppi di viaggio
# Creiamo nuove feature che estraggono informazioni sui gruppi di viaggio dai PassengerId
class GroupTransformer(FeatureTransformer):
    """Trasformatore che estrae informazioni sui gruppi di viaggio."""

    def get_name(self) -> str:
        return "Group Division"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)

        # Calcoliamo la dimensione del gruppo
        df['Group_size'] = df['Group'].map(lambda x: df['Group'].value_counts()[x])

        # Dato che si può notare un alta correlazione tra gruppi da 1 trasportati
        # più di gruppi grandi, creiamo una nuova feature binaria che traccia questo.
        df['Solo'] = (df['Group_size'] == 1).astype(int)

        # Plot distribution of new features (solo se esiste Transported)
        if 'Transported' in df.columns and show_plots:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            sns.histplot(data=df, x='Group', hue='Transported', binwidth=1)
            plt.title('Group')
            plt.show(block=False)

            plt.figure(figsize=(10, 4))
            sns.countplot(data=df, x='Solo', hue='Transported')
            plt.title('Passenger travelling solo or not')
            plt.ylim([0, 3000])
            plt.show(block=False)

        # Restituisci il DataFrame con le nuove feature
        return df


# Posizione della cabina
# Estraiamo informazioni dalla colonna Cabin: ponte, sezione, lato
class CabinLocationTransformer(FeatureTransformer):
    """Trasformatore che estrae informazioni dalla colonna Cabin."""

    def get_name(self) -> str:
        return "Cabin Location"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Estraiamo il ponte, la sezione e la cabina dalla colonna Cabin

        # Rimpiazziamo i valori NaN con una stringa fittizia (per ora) per poter effettuare lo split
        df['Cabin'] = df['Cabin'].fillna('Z/9999/Z')

        # Nuove feature: Deck, Num, Side
        df['Cabin_deck'] = df['Cabin'].apply(lambda x: x.split('/')[0])
        df['Cabin_number'] = df['Cabin'].apply(lambda x: x.split('/')[1]).astype(int)
        df['Cabin_side'] = df['Cabin'].apply(lambda x: x.split('/')[2])

        # Reinseriamo i NaN originari
        df.loc[df['Cabin_deck'] == 'Z', 'Cabin_deck'] = np.nan
        df.loc[df['Cabin_side'] == 'Z', 'Cabin_side'] = np.nan
        df.loc[df['Cabin_number'] == 9999, 'Cabin_number'] = np.nan

        # Eliminiamo la colonna Cabin originale
        df = df.drop(columns=['Cabin'])

        # Plot delle nuove feature (solo se esiste Transported)
        if 'Transported' in df.columns and show_plots:
            fig = plt.figure(figsize=(10, 4))
            plt.subplot(3, 1, 1)
            sns.countplot(
                data=df,
                x='Cabin_deck',
                hue='Transported',
                order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'],
            )
            plt.title('Cabin Deck Distribution')

            plt.subplot(3, 1, 2)
            sns.histplot(data=df, x='Cabin_number', hue='Transported', binwidth=10)
            plt.vlines(300, ymin=0, ymax=200, color='black')
            plt.vlines(600, ymin=0, ymax=200, color='black')
            plt.vlines(900, ymin=0, ymax=200, color='black')
            plt.vlines(1200, ymin=0, ymax=200, color='black')
            plt.vlines(1500, ymin=0, ymax=200, color='black')
            plt.vlines(1800, ymin=0, ymax=200, color='black')
            plt.title('Cabin Number Distribution')
            plt.xlim([0, 2000])

            plt.subplot(3, 1, 3)
            sns.countplot(data=df, x='Cabin_side', hue='Transported')
            plt.title('Cabin Side Distribution')
            fig.tight_layout()
            plt.show(block=False)

        # Notando che cabin_number è divisa in sezioni di circa 300 numeri, possiamo creare una nuova feature categorica basata su queste sezioni
        df['Cabin_region1'] = (df['Cabin_number'] <= 300).astype(int)
        df['Cabin_region2'] = ((df['Cabin_number'] > 300) & (df['Cabin_number'] <= 600)).astype(int)
        df['Cabin_region3'] = ((df['Cabin_number'] > 600) & (df['Cabin_number'] <= 900)).astype(int)
        df['Cabin_region4'] = ((df['Cabin_number'] > 900) & (df['Cabin_number'] <= 1200)).astype(int)
        df['Cabin_region5'] = ((df['Cabin_number'] > 1200) & (df['Cabin_number'] <= 1500)).astype(int)
        df['Cabin_region6'] = ((df['Cabin_number'] > 1500) & (df['Cabin_number'] <= 1800)).astype(int)
        df['Cabin_region7'] = (df['Cabin_number'] > 1800).astype(int)

        if 'Transported' in df.columns and show_plots:
            plt.figure(figsize=(10, 4))
            sns.countplot(data=df, x='Cabin_region1', hue='Transported')
            plt.title('Cabin Region 1 Distribution')
            plt.ylim([0, 3000])
            plt.show(block=False)

        return df


class FamilySizeTransformer(FeatureTransformer):
    """Trasformatore che calcola la dimensione della famiglia."""

    def get_name(self) -> str:
        return "Family Size"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Rimpiazziamo per ora i valori NaN con dei valori fittizi per poter eseguire lo split
        df['Name'].fillna('Unknown Unknown', inplace=True)

        # Estraiamo il cognome
        df['Surname'] = df['Name'].str.split().str[-1]  # Prendiamo l'ultimo elemento come cognome

        # Calcoliamo la dimensione della famiglia
        df['Family_size'] = df['Surname'].map(lambda x: df['Surname'].value_counts()[x])

        # Reinseriamo i NaN originari
        df.loc[df['Surname'] == 'Unknown', 'Surname'] = np.nan
        df.loc[df['Family_size'] > 100, 'Family_size'] = np.nan  # Consideriamo valori anomali come NaN

        # Drop della colonna Name
        df.drop(columns=['Name'], inplace=True)

        # Plot della distribuzione della dimensione della famiglia (solo se esiste Transported)
        if 'Transported' in df.columns and show_plots:
            plt.figure(figsize=(12, 4))
            sns.countplot(data=df, x='Family_size', hue='Transported')
            plt.title('Family Size Distribution')
        return df


# =============================================================================
# SECTION: Pipeline
# =============================================================================
class FeatureEngineeringPipeline:
    """Pipeline per applicare multiple strategie di trasformazione in sequenza."""

    def __init__(self, transformers: list[FeatureTransformer]):
        """Inizializza la pipeline con una lista di trasformatori.

        Args:
            transformers: Lista di oggetti FeatureTransformer da applicare in sequenza
        """
        self.transformers = transformers

    def fit_transform(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Applica tutte le trasformazioni in sequenza a train e test set.

        Args:
            train: DataFrame di training
            test: DataFrame di test

        Returns:
            Tupla (train_engineered, test_engineered) con le nuove feature
        """
        print("\n[Feature Engineering] Applicazione trasformazioni...")

        # Applichiamo ogni trasformatore in sequenza
        for transformer in self.transformers:
            print(f"  -> Applicando: {transformer.get_name()}")
            train = transformer.transform(train)
            test = transformer.transform(test)

        print("\n[Feature Engineering] Completato!")
        return train, test


# =============================================================================
# SECTION: Public Entry Point
# =============================================================================
def run_feature_engineering(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Esegue tutte le trasformazioni di feature engineering su train e test set.

    Args:
        train: DataFrame di training
        test: DataFrame di test

    Returns:
        Tupla (train_engineered, test_engineered) con le nuove feature
    """
    # Decidi se far mostrare i plot all'interno delle trasformazioni
    global show_plots
    user_input = (
        input(
            "Vuoi mostrare i plot delle trasformazioni di feature engineering? (y/n): "
        )
        .strip()
        .lower()
    )
    if user_input == 'y':
        show_plots = True
    else:
        show_plots = False

    # Definisci la pipeline con le strategie desiderate
    pipeline = FeatureEngineeringPipeline(
        [
            AgeGroupTransformer(),
            ExpenditureTransformer(),
            GroupTransformer(),
            CabinLocationTransformer(),
            FamilySizeTransformer(),
        ]
    )

    return pipeline.fit_transform(train, test)
