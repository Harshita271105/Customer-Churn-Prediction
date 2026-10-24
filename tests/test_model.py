import json
from pathlib import Path
import pytest
import joblib
import pandas as pd
import json
import subprocess
import sys
from pathlib import Path
from src.predict import CHURN_THRESHOLD, predict_churn
from src.customer_explanation import explain_customer
from src.predict import CHURN_THRESHOLD

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

    
def test_customer_explanation_returns_valid_probability():
    customer_data = pd.DataFrame({
        "gender": ["Female"],
        "SeniorCitizen": [0],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "tenure": [6],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["Fiber optic"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["No"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["No"],
        "StreamingMovies": ["No"],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [85.0],
        "TotalCharges": [510.0],
    })

    result = explain_customer(customer_data)

    assert 0 <= result["churn_probability"] <= 1
    assert isinstance(result["observations"], list)


def test_customer_explanation_identifies_customer_context():
    customer_data = pd.DataFrame({
        "gender": ["Female"],
        "SeniorCitizen": [0],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "tenure": [6],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["Fiber optic"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["No"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["No"],
        "StreamingMovies": ["No"],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [85.0],
        "TotalCharges": [510.0],
    })

    result = explain_customer(customer_data)
    features = {item["feature"] for item in result["observations"]}

    assert {"Contract", "tenure", "InternetService"} <= features
    assert result["explanation_type"].startswith("Rule-based")


def test_customer_explanation_rejects_multiple_customers():
    customer_data = pd.DataFrame({
        "tenure": [6, 24],
    })

    with pytest.raises(ValueError, match="exactly one customer"):
        explain_customer(customer_data)



def test_monitoring_report_exists_and_is_valid():
    report_path = (
        Path(__file__).resolve().parents[1]
        / "reports"
        / "data_drift_report.json"
    )

    assert report_path.exists()

    with open(report_path, encoding="utf-8") as file:
        report = json.load(file)

    assert report["reference_rows"] > 0
    assert report["current_rows"] > 0
    assert report["features_tested"] if "features_tested" in report else True
    assert isinstance(report["features"], dict)
    assert "Churn" not in report["features"]
    assert "customerID" not in report["features"]
    assert "drifted_features" in report
    assert "drifted_feature_count" in report
    assert report["multiple_testing_correction"] == "Benjamini-Hochberg FDR"


def test_monitoring_features_have_valid_drift_status():
    report_path = (
        Path(__file__).resolve().parents[1]
        / "reports"
        / "data_drift_report.json"
    )

    with open(report_path, encoding="utf-8") as file:
        report = json.load(file)

    assert report["drifted_feature_count"] == len(report["drifted_features"])

    for feature, result in report["features"].items():
        assert result["type"] in {"numeric", "categorical"}

        if result.get("p_value") is not None:
            assert 0 <= result["p_value"] <= 1
            assert "adjusted_p_value" in result
            assert 0 <= result["adjusted_p_value"] <= 1
            assert result["status"] in {
                "potential_drift",
                "no_statistically_significant_drift_detected",
            }

