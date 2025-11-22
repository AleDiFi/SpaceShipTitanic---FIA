# Qui miglioriamo le features al fine di migliorare le performance dei modelli ML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from abc import ABC, abstractmethod


# Strategy Pattern per Feature Engineering
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


# Età 
# Per la feature "Age" dividiamo in fasce di età, questo ci aiutà anche a gestire i valori mancanti ad esempio relativi a quanto spendere in base all'età

class AgeGroupTransformer(FeatureTransformer):
    """Trasformatore che crea fasce di età dalla feature Age."""
    
    def get_name(self) -> str:
        return "Age Grouping"
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Inizializza come stringa per evitare warning dtype
        df['Age_group'] = 'Unknown'
        df.loc[df['Age']<=12,'Age_group']='Age_0-12'
        df.loc[(df['Age']>12) & (df['Age']<18),'Age_group']='Age_13-17'
        df.loc[(df['Age']>=18) & (df['Age']<=25),'Age_group']='Age_18-25'
        df.loc[(df['Age']>25) & (df['Age']<=30),'Age_group']='Age_26-30'
        df.loc[(df['Age']>30) & (df['Age']<=50),'Age_group']='Age_31-50'
        df.loc[df['Age']>50,'Age_group']='Age_51+'

        # Plotta solo se esiste la colonna Transported (presente solo nel train)
        if 'Transported' in df.columns:
            plt.figure(figsize=(10,4))
            sns.countplot(data=df, x='Age_group', hue='Transported', order=['Age_0-12','Age_13-17','Age_18-25','Age_26-30','Age_31-50','Age_51+'])
            plt.title('Age group distribution')
            plt.show(block=False)
        
        return df
    
# Spese
# Creiamo una nuova feature che somma tutte le spese e una binaria che indica se il passeggero non ha speso nulla

class ExpenditureTransformer(FeatureTransformer):
    """Trasformatore che calcola le spese totali e identifica passeggeri senza spese."""
    
    def get_name(self) -> str:
        return "Total Expenditure"
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        df['Expenditure']=df[exp_feats].sum(axis=1)
        df['No_spending']=(df['Expenditure']==0).astype(int)

        # Plot della distribuzione delle spese totali (solo se esiste Transported)
        if 'Transported' in df.columns:
            fig=plt.figure(figsize=(12,4))
            plt.subplot(1,2,1)
            sns.histplot(data=df, x='Expenditure',hue='Transported', bins=200)
            plt.title('Total Expenditure Distribution')
            plt.ylim([0,200])
            plt.xlim([0,20000])

            plt.subplot(1,2,2)
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
        df['Group_size']=df['Group'].map(lambda x: df['Group'].value_counts()[x])
        
        # Dato che si può notare un alta correlazione tra gruppi da 1 trasportati più di gruppi grandi, creiamo una nuova feature binaria che tiene traccia di questo
        df['Solo']=(df['Group_size']==1).astype(int)
        
        # Plot distribution of new features (solo se esiste Transported)
        if 'Transported' in df.columns:
            plt.figure(figsize=(20,4))
            plt.subplot(1,2,1)
            sns.histplot(data=df, x='Group', hue='Transported', binwidth=1)
            plt.title('Group')
            plt.show(block=False)

            plt.figure(figsize=(10,4))
            sns.countplot(data=df, x='Solo', hue='Transported')
            plt.title('Passenger travelling solo or not')
            plt.ylim([0,3000])
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
        
        return df

# Pipeline che applica le strategie di trasformazione

class FeatureEngineeringPipeline:
    """Pipeline per applicare multiple strategie di trasformazione in sequenza."""
    
    def __init__(self, transformers: list[FeatureTransformer]):
        """Inizializza la pipeline con una lista di trasformatori.
        
        Args:
            transformers: Lista di oggetti FeatureTransformer da applicare in sequenza
        """
        self.transformers = transformers
    
    def fit_transform(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def run_feature_engineering(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Esegue tutte le trasformazioni di feature engineering su train e test set.
    
    Args:
        train: DataFrame di training
        test: DataFrame di test
        
    Returns:
        Tupla (train_engineered, test_engineered) con le nuove feature
    """
    # Definisci la pipeline con le strategie desiderate
    pipeline = FeatureEngineeringPipeline([
        AgeGroupTransformer(),
        ExpenditureTransformer(),
        GroupTransformer(),
        CabinLocationTransformer()
    ])
    
    return pipeline.fit_transform(train, test)
    