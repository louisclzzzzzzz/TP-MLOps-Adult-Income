"""Tests for the inference module."""

import pytest

from mlops_tp.inference import load_model, predict


SAMPLE_FEATURES = {
    "age": 39.0,
    "workclass": "State-gov",
    "fnlwgt": 77516.0,
    "education": "Bachelors",
    "education_num": 13.0,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174.0,
    "capital_loss": 0.0,
    "hours_per_week": 40.0,
    "native_country": "United-States",
}


@pytest.fixture(scope="module")
def model():
    """Load the trained model once."""
    return load_model()


def test_predict_returns_known_class(model):
    """predict() should return a prediction that is one of the known classes."""
    result = predict(model, SAMPLE_FEATURES)
    assert result["prediction"] in ("<=50K", ">50K")


def test_predict_returns_probabilities(model):
    """predict_proba values should be between 0 and 1."""
    result = predict(model, SAMPLE_FEATURES)
    assert "proba" in result
    for label, prob in result["proba"].items():
        assert 0.0 <= prob <= 1.0, f"Probability for {label} out of range: {prob}"


def test_predict_probabilities_sum_to_one(model):
    """Probabilities should sum to approximately 1."""
    result = predict(model, SAMPLE_FEATURES)
    total = sum(result["proba"].values())
    assert abs(total - 1.0) < 1e-3, f"Probabilities sum to {total}, expected ~1.0"
