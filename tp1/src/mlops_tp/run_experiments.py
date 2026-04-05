"""Run multiple MLflow experiments with different configurations to compare results."""

from mlops_tp.train import train

EXPERIMENTS = [
    {
        "description": "Run 1 - RandomForest baseline (100 arbres, imputation mediane)",
        "params": {
            "model_type": "RandomForest",
            "n_estimators": 100,
            "max_depth": None,
            "num_imputer_strategy": "median",
            "cat_imputer_strategy": "most_frequent",
            "test_ratio": 0.15,
        },
    },
    {
        "description": "Run 2 - RandomForest profond (200 arbres, max_depth=20, imputation moyenne)",
        "params": {
            "model_type": "RandomForest",
            "n_estimators": 200,
            "max_depth": 20,
            "num_imputer_strategy": "mean",
            "cat_imputer_strategy": "most_frequent",
            "test_ratio": 0.15,
        },
    },
    {
        "description": "Run 3 - GradientBoosting (150 arbres, max_depth=4)",
        "params": {
            "model_type": "GradientBoosting",
            "n_estimators": 150,
            "max_depth": 4,
            "num_imputer_strategy": "median",
            "cat_imputer_strategy": "most_frequent",
            "test_ratio": 0.15,
        },
    },
    {
        "description": "Run 4 - RandomForest avec split test plus grand (test_ratio=0.20)",
        "params": {
            "model_type": "RandomForest",
            "n_estimators": 100,
            "max_depth": None,
            "num_imputer_strategy": "median",
            "cat_imputer_strategy": "most_frequent",
            "test_ratio": 0.20,
        },
    },
]


if __name__ == "__main__":
    for exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(exp["description"])
        print('='*60)
        train(**exp["params"])
    print("\nTous les runs sont termines. Lancez 'mlflow ui' pour comparer les resultats.")
