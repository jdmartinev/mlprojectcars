import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

import os
import wandb
from dotenv import load_dotenv
import typer
import config

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from pathlib import Path

from sklearn.impute import SimpleImputer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    MinMaxScaler,
    OrdinalEncoder,
)
import joblib
import pickle

# Add requirement for wandb core
wandb.require("core")

def train_hyperparameter_tuning(base_path, config):
    df = _read_data_from_input_csv_files(base_path)
    num_cols = ['year','mileage','tax','milesPerGallon','engineSize']
    cat_cols = ['brand_model','transmission','fuelType']
    target = 'price'

    X_train = df[cat_cols+num_cols]
    y_train = df[target]

    #Transformaciones para variables categoricas
    categorical_transformer = Pipeline(steps=[
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ('imputer', SimpleImputer(strategy='median')),
    ])

    numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
    ])

    # Definimos el Pipeline de procesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_cols),  # Numerical columns with mean imputer
            ('cat', categorical_transformer, cat_cols)  # Categorical columns
        ],
        remainder='passthrough'  # This will passthrough any remaining columns
    )

    _save_data_pipeline(base_path,preprocessor)

    param_grid = config['param_grid']
    scorers = config["scorers"]

    base_m = config['base_m']
    print(base_m)
    base_m = base_m(random_state=config["random_state"])

    pipeline = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('regressor', base_m),
    ])

    model = GridSearchCV(pipeline, cv = config['cv'], param_grid=param_grid, scoring=scorers, refit=config['refit'], n_jobs=config['njobs'], return_train_score=config['return_train_score'])
    model.fit(X_train, y_train)

    wandb_config=model.best_params_.copy()
    wandb_config['model_name'] = config['model_name']
    wandb.init(
        # set the wandb project where this run will be logged
        project=config['wandb_project'],
        name = config['run_name'],
        # track hyperparameters and run metadata
        config=wandb_config
    )
    best_index = model.best_index_
    best_results = model.cv_results_
    for metric in scorers.keys():
        metric_mean = best_results[f'mean_test_{metric}'][best_index]
        metric_std = best_results[f'std_test_{metric}'][best_index]

        # Print the metrics to the console
        print(f'Best {metric} mean: {metric_mean}')
        print(f'Best {metric} std: {metric_std}')

    # Log the metrics to wandb
        wandb.log({
            f'best_{metric}_mean': metric_mean,
            f'best_{metric}_std': metric_std,
        })

    wandb.finish()

    _save_model(base_path,model.best_params_)

def train(base_path, config):
    df = _read_data_from_input_csv_files(base_path)
    num_cols = ['year','mileage','tax','milesPerGallon','engineSize']
    cat_cols = ['brand_model','transmission','fuelType']
    target = 'price'

    X_train = df[cat_cols+num_cols]
    y_train = df[target]

    # Load the preprocessor object from the joblib file
    preprocessor_file_path = os.path.join(base_path,'train','preprocessor.joblib')
    preprocessor = joblib.load(preprocessor_file_path)

    best_params_file = os.path.join(base_path,'train','best_params.pkl')
    with open(best_params_file, 'rb') as file:
        best_params = pickle.load(file)
    
    cleaned_params = {key.replace('regressor__', ''): value for key, value in best_params.items()}
    X_train = preprocessor.fit_transform(X_train)
   
    _save_data_pipeline(base_path,preprocessor)
    base_m = config['base_m']
    model = base_m(**cleaned_params)
    model.fit(X_train,y_train)
    model_file_path = os.path.join(base_path,'train','model.joblib')
    joblib.dump(model, model_file_path)

def _save_data_pipeline(base_path, preprocessor):
    joblib_file_path = os.path.join(base_path,'train','preprocessor.joblib')
    os.makedirs(os.path.dirname(joblib_file_path), exist_ok=True)
    joblib.dump(preprocessor, joblib_file_path)

def _save_model(base_path, best_params):
    pkl_file_path = os.path.join(base_path,'train','best_params.pkl')
    os.makedirs(os.path.dirname(pkl_file_path), exist_ok=True)
    with open(pkl_file_path, 'wb') as file:
         pickle.dump(best_params, file)

def _read_data_from_input_csv_files(base_path):
    """
    This function reads every CSV file available and concatenates
    them into a single dataframe.
    """
    input_directory = Path(base_path) / "train-prepared"
    files = [file for file in input_directory.glob("*.csv")]
    if len(files) == 0:
        raise ValueError(f"The are no CSV files in {str(input_directory)}/")     
    raw_data = [pd.read_csv(file) for file in files]
    df = pd.concat(raw_data)
    # Shuffle the data
    return df

def main(
    random_state: int = typer.Option(42),
    max_missing_percent: int = typer.Option(50),
    max_nulls_allowed: int = typer.Option(0),
    min_engine_size: float = typer.Option(1.0),
    max_year: int = typer.Option(2020),
    test_size: float = typer.Option(0.2),
    n_estimators: list[int] = typer.Option([50, 100]),
    max_depth: list[int] = typer.Option([6, 7, 8, 9]),
    min_samples_leaf: list[int] = typer.Option([10, 300, 2000]),
    cv: int = typer.Option(3),
    refit: str = typer.Option('r2'),
    njobs: int = typer.Option(-1),
    return_train_score: bool = typer.Option(True),
    model_name: str = typer.Option('Random Forest'),
    wandb_project: str = typer.Option("cars_gridsearch_cv_project"),
    run_name: str = typer.Option("Example 2"),
    base_m: str = typer.Option("GradientBoosting")
):
    prep_config, train_config = config.configure(
        random_state,
        max_missing_percent,
        max_nulls_allowed,
        min_engine_size,
        max_year,
        test_size,
        n_estimators,
        max_depth,
        min_samples_leaf,
        cv,
        refit,
        njobs,
        return_train_score,
        model_name,
        wandb_project,
        run_name,
        base_m
    )

    base_path = ''
    train_hyperparameter_tuning(base_path, train_config)
    train(base_path, train_config)


if __name__ == "__main__":
    # Access the dictionary
    load_dotenv('../api_keys.env')
    # Get the API key from environment variable
    wandb_api_key = os.getenv('WANDB_API_KEY')
    wandb.login(key=wandb_api_key)
    typer.run(main)


    