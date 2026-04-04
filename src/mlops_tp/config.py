"""Configuration constants for the MLOps TP project.

Paths are resolved relative to this file so the project is portable
across local machines, Docker containers, and CI environments.
Critical values can be overridden via environment variables.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"

# Allow overriding the artifacts directory via environment variable.
# This is useful when running in a container where the model is stored
# at a non-standard path, without changing source code.
_default_artifacts = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(_default_artifacts)))

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

TRAIN_DATA_PATH = DATA_DIR / "adult.data"
TEST_DATA_PATH = DATA_DIR / "adult.test"

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

TARGET_COLUMN = "income"

CATEGORICAL_FEATURES = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]

NUMERICAL_FEATURES = [
    "age", "fnlwgt", "education_num", "capital_gain",
    "capital_loss", "hours_per_week",
]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------------------------
# Artifact paths (all derived from ARTIFACTS_DIR)
# ---------------------------------------------------------------------------

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURE_SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"
RUN_INFO_PATH = ARTIFACTS_DIR / "run_info.json"

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

MODEL_VERSION = "0.1.0"
TASK_TYPE = "classification"
