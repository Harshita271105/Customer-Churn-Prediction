
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "telco_churn_cleaned.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
REPORT_PATH = ROOT / "reports" / "model_validation.json"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)

    print("\n--- 5-FOLD CROSS-VALIDATION ---")

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring={
            "roc_auc": "roc_auc",
            "f1": "f1",
            "recall": "recall",
        },
        n_jobs=-1,
    )

    cv_summary = {}

    for metric in ["roc_auc", "f1", "recall"]:
        scores = cv_results[f"test_{metric}"]
        cv_summary[metric] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
        }

        print(
            f"{metric}: {scores.mean():.4f} "
            f"+/- {scores.std():.4f}"
        )

    print("\n--- TEST SET THRESHOLD ANALYSIS ---")

    # The saved pipeline was trained on the training split.
    # Cross-validation above evaluates its generalization on training folds.
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]

    threshold_results = {}

    for threshold in [0.30, 0.40, 0.50, 0.60]:
        predictions = (probabilities >= threshold).astype(int)

        metrics = {
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
        }

        threshold_results[str(threshold)] = metrics

        print(f"\nThreshold: {threshold:.2f}")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

    report = {
        "model": "churn_model.joblib",
        "cross_validation": {
            "folds": 5,
            "training_rows": len(X_train),
            "metrics": cv_summary,
        },
        "test_set": {
            "rows": len(X_test),
            "roc_auc": float(
                roc_auc_score(y_test, probabilities)
            ),
            "threshold_analysis": threshold_results,
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"\nValidation report saved: {REPORT_PATH}")
    print("\nModel validation completed.")


if __name__ == "__main__":
    main()
