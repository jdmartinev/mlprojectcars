import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

import config

class MergeElectricToOther(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy[self.column_name] = X_copy[self.column_name].replace('Electric', 'Other')
        return X_copy


def preparation(base_directory, config):
    #Transforming variables into the formats we need them
    df = _read_data_from_input_csv_files(base_directory)
    train_df, test_df = _split_data(df,test_size=config["test_size"])
    _save_baselines(base_directory, train_df, 'train')
    _save_baselines(base_directory, test_df, 'test')

    #Cambiar el tipo de variables 
    train_df.loc[:,'price'] = train_df['price'].astype('float64')
    test_df.loc[:,'price'] = test_df['price'].astype('float64')
    # Remover columnas con más de un porcentaje determinado de datos faltantes
    missing_percent = train_df.isnull().mean() * 100
    # Filter out columns with no missing values
    missing_percent = missing_percent[missing_percent > 0]
    # Filter out columns with missing values percentage higher than the threshold
    columns_to_drop = missing_percent[missing_percent > config['max_missing_percent']].index
    train_df = train_df.drop(columns=columns_to_drop)
    # Threshold to remove samples having missing values greater than threshold
    max_nulls_allowed = config['max_nulls_allowed']
    # Getting Missing count of each sample            
    nulls_per_row = train_df.isnull().sum(axis=1)
    # Filter out rows with more than the allowed number of null values
    train_df = train_df[nulls_per_row <= max_nulls_allowed]
    # Eliminemos los registros con tamaño del motor menor a 1 litro:
    train_df.drop(index=train_df[train_df["engineSize"] < 1].index, inplace=True)
    # Eliminemos el registro con año mayor que 2020:
    train_df.drop(index=train_df[train_df['year'] > 2020].index, inplace=True)
    # Creemos el pipeline
    pipeline = Pipeline([
        ('merge_electric_to_other', MergeElectricToOther(column_name='fuelType'))
    ])
    # Fit and transform the data
    train_df = pipeline.fit_transform(train_df)
    #Qué deberíamos almacenar?
    prep_config = {}
    prep_config["columns_to_drop"] = columns_to_drop.tolist()
    prep_config["pipeline"] = pipeline
    # Specify the file name
    prep_file = os.path.join(base_directory,'prep','prep_config.pkl')
    # Ensure the directory exists
    os.makedirs(os.path.dirname(prep_file), exist_ok=True)
    with open(prep_file, 'wb') as file:
         pickle.dump(prep_config, file)
    test_df = preparation_transform(base_directory,test_df,config)
    _save_prepared(base_directory,train_df,'train')
    _save_prepared(base_directory,test_df,'test')
    return(train_df, test_df)

def preparation_transform(base_directory, df,  config):
    prep_file = os.path.join(base_directory, 'prep','prep_config.pkl')
    with open(prep_file, 'rb') as file:
        prep_config = pickle.load(file)
    columns_to_drop = prep_config["columns_to_drop"]
    pipeline = prep_config["pipeline"]
    df = df.drop(columns_to_drop)
    # Threshold to remove samples having missing values greater than threshold
    # Getting Missing count of each sample            
    nulls_per_row = df.isnull().sum(axis=1)
    df = df[nulls_per_row <= config['max_nulls_allowed']]
    # Eliminemos los registros con tamaño del motor menor a 1 litro:
    df.drop(index=df[df['engineSize'] < config["min_engineSize"]].index, inplace=True)
    # Eliminemos el registro con año mayor que 2020:
    df.drop(index=df[df['year'] > config["max_year"]].index, inplace=True)
        # Apliquemos el pipeline para combinar las columnas
    df = pipeline.transform(df)
    return(df)
    

    

def _read_data_from_input_csv_files(base_directory):
    """
    This function reads every CSV file available and concatenates
    them into a single dataframe.
    """
    input_directory = Path(base_directory) / "../data"
    files = [file for file in input_directory.glob("*.csv")]
    if len(files) == 0:
        raise ValueError(f"The are no CSV files in {str(input_directory)}/")     
    raw_data = [pd.read_csv(file) for file in files]
    df = pd.concat(raw_data)
    # Shuffle the data
    return df.sample(frac=1, random_state=42)


def _split_data(df,test_size=0.2):
    """
    Splits the data into two sets: train, and test.
    """
    df_train, df_test = train_test_split(df, test_size=0.3)
    return df_train, df_test


def _save_baselines(base_directory, df, partition):
    """
    This function saves the untransformed data to disk so we can use them as baselines later.
    """
    baseline_path = Path(base_directory) / f"{partition}-baseline"
    baseline_path.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df.to_csv(baseline_path / f"{partition}-baseline.csv")

def _save_prepared(base_directory, df, partition):
    """
    This function saves the untransformed data to disk so we can use them as baselines later.
    """
    baseline_path = Path(base_directory) / f"{partition}-prepared"
    baseline_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(baseline_path / f"{partition}-prepared.csv")
  
if __name__ == "__main__":
    # Access the dictionary
    prep_config = config.prep_config
    preparation(base_directory='',config = prep_config)