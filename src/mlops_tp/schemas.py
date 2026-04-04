"""Pydantic schemas for the API request/response contract."""

from pydantic import BaseModel


class PredictRequest(BaseModel):
    """POST /predict request body."""
    features: dict

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class PredictResponse(BaseModel):
    """POST /predict response body."""
    prediction: str
    task: str
    proba: dict
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    """GET /health response body."""
    status: str


class MetadataResponse(BaseModel):
    """GET /metadata response body."""
    model_version: str
    task: str
    features: dict
