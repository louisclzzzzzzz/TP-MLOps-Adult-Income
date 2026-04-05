"""FastAPI application exposing the trained model via a REST API."""

import json
import time

from fastapi import FastAPI, HTTPException

from mlops_tp.config import (
    CATEGORICAL_FEATURES,
    FEATURE_SCHEMA_PATH,
    MODEL_VERSION,
    NUMERICAL_FEATURES,
    TASK_TYPE,
)
from mlops_tp.inference import load_model, predict
from mlops_tp.schemas import HealthResponse, MetadataResponse, PredictRequest, PredictResponse

app = FastAPI(
    title="Adult Income Prediction API",
    description="API REST pour prédire si le revenu d'un individu dépasse 50K$/an.",
    version=MODEL_VERSION,
)

# Load model once at startup
model = load_model()

# Load feature schema once at startup
with open(FEATURE_SCHEMA_PATH) as f:
    feature_schema = json.load(f)


@app.get("/health", response_model=HealthResponse)
def health():
    """Check if the API is alive and ready."""
    return HealthResponse(status="alive")


@app.get("/metadata", response_model=MetadataResponse)
def metadata():
    """Return model version, task type, and expected features."""
    return MetadataResponse(
        model_version=MODEL_VERSION,
        task=TASK_TYPE,
        features=feature_schema,
    )


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    """Receive features and return the model prediction."""
    features = request.features
    expected_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    # Validate that all required features are present
    missing = [f for f in expected_features if f not in features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing features: {missing}",
        )

    # Validate numerical feature types
    for feat in NUMERICAL_FEATURES:
        val = features[feat]
        if not isinstance(val, (int, float)):
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{feat}' must be a number, got {type(val).__name__}.",
            )

    # Validate categorical feature types
    for feat in CATEGORICAL_FEATURES:
        val = features[feat]
        if not isinstance(val, str):
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{feat}' must be a string, got {type(val).__name__}.",
            )

    start = time.perf_counter()
    result = predict(model, features)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return PredictResponse(
        prediction=result["prediction"],
        task=TASK_TYPE,
        proba=result["proba"],
        model_version=MODEL_VERSION,
        latency_ms=latency_ms,
    )
