from pathlib import Path

import joblib

from pathlib import Path

import joblib
import pandas as pd

from src.predict import predict_churn

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

    import pandas as pd

from src.predict import predict_churn


def test_prediction_output_columns():
    customer_data = pd.DataFrame({
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

    results = predict_churn(customer_data)

    assert len(results) == 2
    assert "PredictedChurn" in results.columns
    assert "ChurnProbability" in results.columns
    assert results["ChurnProbability"].between(0, 1).all()