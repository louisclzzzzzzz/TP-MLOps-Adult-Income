"""Training script: loads data, builds a scikit-learn pipeline, trains, evaluates, and saves artifacts."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlops_tp.config import (
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    COLUMN_NAMES,
    FEATURE_SCHEMA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    MODEL_VERSION,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    RUN_INFO_PATH,
    TARGET_COLUMN,
    TEST_RATIO,
    TRAIN_DATA_PATH,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)

MLFLOW_EXPERIMENT_NAME = "adult-income-classification"
MLFLOW_TRACKING_URI = "sqlite:///" + str(Path(__file__).resolve().parent.parent.parent / "mlflow.db")


def load_data() -> pd.DataFrame:
    """Load the adult dataset from CSV."""
    df = pd.read_csv(
        TRAIN_DATA_PATH,
        names=COLUMN_NAMES,
        sep=r",\s*",
        engine="python",
        na_values="?",
    )
    return df


def clean_target(y: pd.Series) -> pd.Series:
    """Normalize target labels (remove trailing dots from test set)."""
    return y.str.strip().str.rstrip(".")


def split_data(df: pd.DataFrame, test_ratio: float = TEST_RATIO):
    """Split data into train / validation / test with stratification."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = clean_target(df[TARGET_COLUMN])

    val_ratio = VALIDATION_RATIO
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(val_ratio + test_ratio),
        random_state=RANDOM_STATE,
        stratify=y,
    )

    val_fraction = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_fraction),
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_pipeline(
    model_type: str = "RandomForest",
    n_estimators: int = 100,
    max_depth: int | None = None,
    num_imputer_strategy: str = "median",
    cat_imputer_strategy: str = "most_frequent",
    C: float = 1.0,
) -> Pipeline:
    """Build a scikit-learn pipeline with preprocessing and a classifier."""
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=num_imputer_strategy)),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=cat_imputer_strategy)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, NUMERICAL_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    if model_type == "RandomForest":
        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif model_type == "GradientBoosting":
        classifier = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth or 3,
            random_state=RANDOM_STATE,
        )
    elif model_type == "LogisticRegression":
        classifier = LogisticRegression(
            C=C,
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    return pipeline


def save_confusion_matrix(y_true, y_pred, output_path: Path):
    """Save confusion matrix figure as PNG."""
    cm = confusion_matrix(y_true, y_pred, labels=[">50K", "<=50K"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[">50K", "<=50K"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (test set)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _pos_class_index(pipeline) -> int:
    """Return the column index of the '>50K' class in predict_proba output."""
    classes = list(pipeline.classes_)
    return classes.index(">50K") if ">50K" in classes else 1


def save_roc_curve(pipeline, X_test, y_test, output_path: Path):
    """Save ROC curve figure as PNG."""
    if not hasattr(pipeline, "predict_proba"):
        return
    idx = _pos_class_index(pipeline)
    y_scores = pipeline.predict_proba(X_test)[:, idx]
    y_bin = (y_test == ">50K").astype(int)
    fpr, tpr, _ = roc_curve(y_bin, y_scores)
    auc = roc_auc_score(y_bin, y_scores)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return auc


def save_artifacts(pipeline: Pipeline, metrics: dict, df: pd.DataFrame):
    """Save model, metrics, feature schema, and run info to disk."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    feature_schema = {}
    for col in NUMERICAL_FEATURES:
        feature_schema[col] = {"type": "float", "example": float(df[col].dropna().iloc[0])}
    for col in CATEGORICAL_FEATURES:
        feature_schema[col] = {
            "type": "string",
            "example": str(df[col].dropna().iloc[0]),
            "values": sorted(df[col].dropna().unique().tolist()),
        }
    with open(FEATURE_SCHEMA_PATH, "w") as f:
        json.dump(feature_schema, f, indent=2)

    run_info = {
        "dataset": "Adult Census Income (UCI)",
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "target": TARGET_COLUMN,
        "split": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": TEST_RATIO,
        },
        "random_state": RANDOM_STATE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(RUN_INFO_PATH, "w") as f:
        json.dump(run_info, f, indent=2)


def train(
    model_type: str = "RandomForest",
    n_estimators: int = 100,
    max_depth: int | None = None,
    num_imputer_strategy: str = "median",
    cat_imputer_strategy: str = "most_frequent",
    test_ratio: float = TEST_RATIO,
    C: float = 1.0,
):
    """Full training pipeline with MLflow tracking."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        # --- Log parameters ---
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("num_imputer_strategy", num_imputer_strategy)
        mlflow.log_param("cat_imputer_strategy", cat_imputer_strategy)
        mlflow.log_param("test_ratio", test_ratio)
        mlflow.log_param("random_state", RANDOM_STATE)
        if model_type == "LogisticRegression":
            mlflow.log_param("C", C)

        print("Loading data...")
        df = load_data()
        print(f"Dataset shape: {df.shape}")

        print(f"Splitting data (train/val/test, test_ratio={test_ratio})...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, test_ratio=test_ratio)
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        print(f"Building pipeline ({model_type})...")
        pipeline = build_pipeline(
            model_type=model_type,
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_imputer_strategy=num_imputer_strategy,
            cat_imputer_strategy=cat_imputer_strategy,
            C=C,
        )

        print("Training model...")
        pipeline.fit(X_train, y_train)

        # Evaluate on validation set
        y_val_pred = pipeline.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, pos_label=">50K")

        # Evaluate on test set
        y_test_pred = pipeline.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, pos_label=">50K")

        print(f"Validation — Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
        print(f"Test       — Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")

        # --- Log metrics ---
        mlflow.log_metric("val_accuracy", round(val_accuracy, 4))
        mlflow.log_metric("val_f1", round(val_f1, 4))
        mlflow.log_metric("test_accuracy", round(test_accuracy, 4))
        mlflow.log_metric("test_f1", round(test_f1, 4))

        # AUC if available
        if hasattr(pipeline, "predict_proba"):
            idx = _pos_class_index(pipeline)
            y_bin = (y_test == ">50K").astype(int)
            y_scores = pipeline.predict_proba(X_test)[:, idx]
            auc = roc_auc_score(y_bin, y_scores)
            mlflow.log_metric("test_roc_auc", round(auc, 4))
            print(f"Test       — ROC AUC: {auc:.4f}")

        # --- Log artifacts ---
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        cm_path = ARTIFACTS_DIR / "confusion_matrix.png"
        save_confusion_matrix(y_test, y_test_pred, cm_path)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        roc_path = ARTIFACTS_DIR / "roc_curve.png"
        save_roc_curve(pipeline, X_test, y_test, roc_path)
        if roc_path.exists():
            mlflow.log_artifact(str(roc_path), artifact_path="plots")

        # --- Log model ---
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        metrics = {
            "model_version": MODEL_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hyperparameters": {
                "model_type": model_type,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": RANDOM_STATE,
            },
            "validation": {
                "accuracy": round(val_accuracy, 4),
                "f1_score": round(val_f1, 4),
            },
            "test": {
                "accuracy": round(test_accuracy, 4),
                "f1_score": round(test_f1, 4),
            },
        }

        print("Saving artifacts to disk...")
        save_artifacts(pipeline, metrics, df)
        print(f"Artifacts saved to {ARTIFACTS_DIR}")

    return pipeline, metrics


if __name__ == "__main__":
    train()
