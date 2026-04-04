"""Tests for the training pipeline."""

import json

import pytest

from mlops_tp.config import FEATURE_SCHEMA_PATH, METRICS_PATH, MODEL_PATH, RUN_INFO_PATH
from mlops_tp.train import train


@pytest.fixture(scope="module")
def trained_artifacts():
    """Run training once for all tests in this module."""
    pipeline, metrics = train()
    return pipeline, metrics


def test_training_produces_model(trained_artifacts):
    """Training should generate model.joblib."""
    assert MODEL_PATH.exists(), "model.joblib was not created"
    assert MODEL_PATH.stat().st_size > 0, "model.joblib is empty"


def test_training_produces_metrics(trained_artifacts):
    """Training should generate metrics.json with accuracy and f1_score."""
    assert METRICS_PATH.exists(), "metrics.json was not created"
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    assert "validation" in metrics
    assert "test" in metrics
    assert "accuracy" in metrics["validation"]
    assert "f1_score" in metrics["validation"]
    assert metrics["validation"]["accuracy"] > 0.7, "Validation accuracy is too low"
    assert metrics["test"]["accuracy"] > 0.7, "Test accuracy is too low"


def test_training_produces_feature_schema(trained_artifacts):
    """Training should generate feature_schema.json."""
    assert FEATURE_SCHEMA_PATH.exists(), "feature_schema.json was not created"
    with open(FEATURE_SCHEMA_PATH) as f:
        schema = json.load(f)
    assert "age" in schema
    assert "workclass" in schema


def test_training_produces_run_info(trained_artifacts):
    """Training should generate run_info.json with dataset metadata."""
    assert RUN_INFO_PATH.exists(), "run_info.json was not created"
    with open(RUN_INFO_PATH) as f:
        info = json.load(f)
    assert "dataset" in info
    assert "shape" in info
    assert "target" in info
    assert "split" in info
    assert "random_state" in info
