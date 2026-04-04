"""Inference module: loads the trained model and makes predictions."""

import joblib
import pandas as pd

from mlops_tp.config import CATEGORICAL_FEATURES, MODEL_PATH, NUMERICAL_FEATURES


def load_model():
    """Load the trained pipeline from disk."""
    return joblib.load(MODEL_PATH)


def predict(model, features: dict) -> dict:
    """Run a prediction on a single sample.

    Args:
        model: The trained scikit-learn pipeline.
        features: A dict with feature names as keys.

    Returns:
        A dict with prediction and probabilities.
    """
    all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    df = pd.DataFrame([features], columns=all_features)

    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    class_labels = model.classes_

    proba_dict = {str(label): round(float(prob), 4) for label, prob in zip(class_labels, probabilities)}

    return {
        "prediction": str(prediction),
        "proba": proba_dict,
    }
