import json
from pathlib import Path

import joblib
import pandas as pd

from src.predict import CHURN_THRESHOLD, predict_churn


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "churn_model.joblib"
)


def sample_customer_data():
    return pd.DataFrame({
        "gender": ["Female", "Male"],
        "SeniorCitizen": [0, 0],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "tenure": [12, 48],
        "PhoneService": ["Yes", "Yes"],
        "MultipleLines": ["No", "Yes"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["Yes", "No"],
        "OnlineBackup": ["No", "Yes"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["Yes", "No"],
        "StreamingTV": ["No", "Yes"],
        "StreamingMovies": ["No", "Yes"],
        "Contract": ["One year", "Month-to-month"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Credit card (automatic)",
        ],
        "MonthlyCharges": [55.0, 90.0],
        "TotalCharges": [660.0, 4320.0],
    })


def test_trained_model_exists():
    assert MODEL_PATH.exists()


def test_model_can_be_loaded():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_has_prediction_method():
    model = joblib.load(MODEL_PATH)
    assert callable(getattr(model, "predict", None))
    assert callable(getattr(model, "predict_proba", None))


def test_prediction_output_columns():
    results = predict_churn(sample_customer_data())

    assert len(results) == 2
    assert "PredictedChurn" in results.columns
    assert "ChurnProbability" in results.columns
    assert "DecisionThreshold" in results.columns


def test_prediction_probabilities_are_valid():
    results = predict_churn(sample_customer_data())

    assert results["ChurnProbability"].between(0, 1).all()
    assert results["PredictedChurn"].isin([0, 1]).all()


def test_predictions_use_configured_threshold():
    results = predict_churn(sample_customer_data())

    assert CHURN_THRESHOLD == 0.60
    assert (results["DecisionThreshold"] == CHURN_THRESHOLD).all()

    expected = (
        results["ChurnProbability"] >= CHURN_THRESHOLD
    ).astype(int)

    assert (results["PredictedChurn"] == expected).all()

def test_feature_importance_report_is_valid():
    report_path = (
        Path(__file__).resolve().parents[1]
        / "reports"
        / "feature_importance.json"
    )

    assert report_path.exists()

    with open(report_path, encoding="utf-8") as file:
        report = json.load(file)

    assert isinstance(report, list)
    assert len(report) > 0

    required_columns = {
        "feature",
        "importance_mean",
        "importance_std",
    }

    assert required_columns.issubset(report[0].keys())

    assert all(
        isinstance(item["feature"], str) and item["feature"]
        for item in report
    )