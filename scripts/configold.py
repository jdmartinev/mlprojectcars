from dotenv import load_dotenv
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.decomposition import PCA


random_state = 42
prep_config = {}
prep_config['max_missing_percent'] = 50
prep_config['max_nulls_allowed'] = 0
prep_config["min_engineSize"] = 1
prep_config["max_year"] = 2020
prep_config["test_size"] = 0.2



train_config = {}
train_config['param_grid'] = {
    'regressor__n_estimators': [50, 100],
    'regressor__max_depth': list(range(6, 10)),
    'regressor__min_samples_leaf': [10, 300, 2000]
}

train_config['scorers'] = {
    'r2': make_scorer(r2_score),
    'mae': make_scorer(mean_absolute_error),
}

train_config['cv'] = 3
train_config['refit'] = 'r2'
train_config['njobs'] = -1
train_config['return_train_score'] = True
train_config['model_name'] = 'Gradient Boosting'
train_config['wandb_project'] = "cars_gridsearch_cv_project"
train_config['base_m'] = GradientBoostingRegressor
train_config['random_state'] = random_state



