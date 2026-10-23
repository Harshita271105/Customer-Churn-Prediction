
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"


def explain_customer(customer: pd.DataFrame, model=None) -> dict:
    """
    Return transparent, rule-based context for one customer's prediction.

    These observations are not causal explanations of the model output.
    """
    if customer.empty or len(customer) != 1:
        raise ValueError("Provide exactly one customer record.")

    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)

    probability = float(model.predict_proba(customer)[0, 1])

    observations = []
    row = customer.iloc[0]

    if row["Contract"] == "Month-to-month":
        observations.append({
            "feature": "Contract",
            "value": row["Contract"],
            "context": "This segment had a higher observed churn rate.",
        })

    if row["tenure"] <= 12:
        observations.append({
            "feature": "tenure",
            "value": row["tenure"],
            "context": "Customers with shorter tenure had a higher observed churn rate.",
        })

    if row["InternetService"] == "Fiber optic":
        observations.append({
            "feature": "InternetService",
            "value": row["InternetService"],
            "context": "This segment had a higher observed churn rate.",
        })

    if row["PaymentMethod"] == "Electronic check":
        observations.append({
            "feature": "PaymentMethod",
            "value": row["PaymentMethod"],
            "context": "This segment had a higher observed churn rate.",
        })

    if row["MonthlyCharges"] >= 70:
        observations.append({
            "feature": "MonthlyCharges",
            "value": row["MonthlyCharges"],
            "context": "Review whether the customer's plan and charges meet their needs.",
        })

    return {
        "churn_probability": probability,
        "observations": observations,
        "explanation_type": "Rule-based context; not causal or model-attribution analysis",
    }
