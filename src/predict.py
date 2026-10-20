
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_churn_cleaned.csv"
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def load_customer_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Remove identifiers and target to prevent data leakage.
    return df.drop(
        columns=["customerID", "Churn"],
        errors="ignore",
    )


def predict_churn(customer_data):
    model = load_model()

    predictions = model.predict(customer_data)
    probabilities = model.predict_proba(customer_data)[:, 1]

    results = customer_data.copy()
    results["PredictedChurn"] = predictions
    results["ChurnProbability"] = probabilities

    return results


def main():
    customer_data = load_customer_data()
    results = predict_churn(customer_data)

    print("\n--- CUSTOMER CHURN PREDICTIONS ---")
    print(results.head(10).to_string(index=False))

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "churn_predictions.csv"
    )

    results.to_csv(output_path, index=False)

    print(f"\nTotal customers scored: {len(results)}")
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
