
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
)
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
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


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
            y_true, predictions, labels=[0, 1]
        ).tolist(),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH)

    if df.empty or df["Churn"].isna().any():
        raise ValueError("Dataset is empty or contains missing targets.")

    if set(df["Churn"].unique()) != {"Yes", "No"}:
        raise ValueError("Unexpected churn target labels.")

    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print("\n--- TRAINING-SET OUT-OF-FOLD PREDICTIONS ---")

    oof_probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    threshold_results = {
        str(threshold): calculate_metrics(
            y_train, oof_probabilities, threshold
        )
        for threshold in THRESHOLDS
    }

    best_threshold = max(
        THRESHOLDS,
        key=lambda threshold: (
            threshold_results[str(threshold)]["f1_score"],
            threshold,
        ),
    )

    print(f"Selected threshold using training OOF F1: {best_threshold:.2f}")

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
        "threshold_selection": {
            "method": "5-fold out-of-fold predictions on training data",
            "thresholds": threshold_results,
        },
        "test_set": {
            "rows": len(X_test),
            "roc_auc": test_roc_auc,
            "metrics_at_selected_threshold": test_metrics,
        },
        "configuration": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "cross_validation_folds": CV_FOLDS,
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"\nValidation report saved: {REPORT_PATH}")
    print("\nModel validation completed.")


if __name__ == "__main__":
    main()
