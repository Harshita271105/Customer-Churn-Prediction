from pathlib import Path

import joblib

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "churn_model.joblib"
)


def test_trained_model_exists():
    assert MODEL_PATH.exists()


def test_model_can_be_loaded():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_has_prediction_method():
    model = joblib.load(MODEL_PATH)
    assert callable(getattr(model, "predict", None))