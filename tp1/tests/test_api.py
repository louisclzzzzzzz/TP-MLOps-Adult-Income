"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from mlops_tp.api import app


@pytest.fixture(scope="module")
def client():
    """Create a test client for the API."""
    return TestClient(app)


VALID_PAYLOAD = {
    "features": {
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
}


def test_health(client):
    """GET /health should return 200 with status alive."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_metadata(client):
    """GET /metadata should return 200 with model info."""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert data["task"] == "classification"
    assert "features" in data


def test_predict_success(client):
    """POST /predict with valid data should return 200."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ("<=50K", ">50K")
    assert data["task"] == "classification"
    assert "proba" in data
    assert "model_version" in data
    assert "latency_ms" in data


def test_predict_missing_feature(client):
    """POST /predict with missing features should return 422."""
    incomplete_payload = {
        "features": {
            "age": 39.0,
            "workclass": "State-gov",
        }
    }
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_predict_wrong_type(client):
    """POST /predict with wrong feature type should return 422."""
    bad_payload = {
        "features": {
            "age": "not_a_number",
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
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
