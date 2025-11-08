import pandas as pd
import numpy as np



#salviamo i dati sul dataframe
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

#visualizziamo le dimensioni dei dataset
print('Train set shape:', train.shape)
print('Test set shape:', test.shape)
train.head()
test.head()

#controlliamo la presenza di valori nulli
print('Missing values in train set:\n', train.isnull().sum())
print('Missing values in test set:\n', test.isnull().sum())

#controlliamo i duplicati
print(f'Number of duplicate rows in train set: {train.duplicated().sum()}, ({np.round(100 * train.duplicated().sum() / len(train), 1)}%)')
print('')
print(f'Number of duplicate rows in test set: {test.duplicated().sum()}, ({np.round(100 * test.duplicated().sum() / len(test), 1)}%)')

#Cardinalità delle feature
#Sappiamo che ci sono 6 feature continue,4 categoriche e 3 descrittive o qualitative quindi vediamo la cardinalità delle feature
train.nunique()

# Ora controlliamo i tipi di dati delle feature
train.dtypes