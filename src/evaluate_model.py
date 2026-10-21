
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "telco_churn_cleaned.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
REPORT_PATH = ROOT / "reports" / "model_validation.json"

THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions
        ).tolist(),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Unexpected or missing target labels.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    print("\n--- TRAINING-SET CROSS-VALIDATION ---")

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
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
        print(f"{metric}: {scores.mean():.4f} +/- {scores.std():.4f}")

    print("\n--- OUT-OF-FOLD THRESHOLD SELECTION ---")

    oof_probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    threshold_results = {
        str(threshold): calculate_metrics(
            y_train, oof_probabilities, threshold
        )
        for threshold in THRESHOLDS
    }

    best_threshold = max(
        THRESHOLDS,
        key=lambda threshold: threshold_results[str(threshold)]["f1_score"],
    )

    print(f"Selected threshold by training OOF F1: {best_threshold:.2f}")

    print("\n--- FINAL HELD-OUT TEST EVALUATION ---")

    test_probabilities = model.predict_proba(X_test)[:, 1]

    test_roc_auc = float(roc_auc_score(y_test, test_probabilities))

    test_metrics = calculate_metrics(
        y_test, test_probabilities, best_threshold
    )

    for name, value in test_metrics.items():
        if name != "confusion_matrix":
            print(f"{name}: {value:.4f}")

    print("Confusion matrix:")
    print(test_metrics["confusion_matrix"])

    report = {
        "model": "churn_model.joblib",
        "threshold_selection_metric": "f1_score",
        "selected_threshold": float(best_threshold),
        "cross_validation": {
            "folds": 5,
            "training_rows": len(X_train),
            "metrics": cv_summary,
        },
        "threshold_selection": {
            "method": "5-fold out-of-fold predictions on training data",
            "thresholds": threshold_results,
        },
        "test_set": {
            "rows": len(X_test),
            "roc_auc": test_roc_auc,
            "metrics_at_selected_threshold": test_metrics,
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"\nValidation report saved: {REPORT_PATH}")
    print("\nModel validation completed.")


if __name__ == "__main__":
    main()
