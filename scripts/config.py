import typer
from dotenv import load_dotenv
import os
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

def configure(
    random_state: int = 42,
    max_missing_percent: int = 50,
    max_nulls_allowed: int = 0,
    min_engine_size: float = 1.0,
    max_year: int = 2020,
    test_size: float = 0.2,
    n_estimators: list[int] = [50, 100],
    max_depth: list[int] = [6, 7, 8, 9],
    min_samples_leaf: list[int] = [10, 300, 2000],
    cv: int = 3,
    refit: str = 'r2',
    njobs: int = -1,
    return_train_score: bool = True,
    model_name: str = 'Random Forest',
    wandb_project: str = "cars_gridsearch_cv_project",
    run_name: str = "Example 2",
    base_m: str = "GradientBoosting"
):

    regressor_map = {
        'RandomForest': RandomForestRegressor,
        'GradientBoosting': GradientBoostingRegressor
    }

    if base_m not in regressor_map:
        raise ValueError(f"Unsupported regressor type: {base_m}. Supported types: {list(regressor_map.keys())}")

    model = regressor_map[base_m]


    prep_config = {
        'max_missing_percent': max_missing_percent,
        'max_nulls_allowed': max_nulls_allowed,
        'min_engineSize': min_engine_size,
        'max_year': max_year,
        'test_size': test_size,
    }

    train_config = {
        'param_grid': {
            'regressor__n_estimators': n_estimators,
            'regressor__max_depth': max_depth,
            'regressor__min_samples_leaf': min_samples_leaf,
        },
        'scorers': {
            'r2': make_scorer(r2_score),
            'mae': make_scorer(mean_absolute_error),
        },
        'cv': cv,
        'refit': refit,
        'njobs': njobs,
        'return_train_score': return_train_score,
        'model_name': model_name,
        'wandb_project': wandb_project,
        'run_name': run_name,
        'base_m': model,
        'random_state': random_state,
    }

    return prep_config, train_config

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
    prep_config, train_config = configure(
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
    print("Preparation Configurations: ", prep_config)
    print("Training Configurations: ", train_config)

if __name__ == "__main__":
    typer.run(main)
