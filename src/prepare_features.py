
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "telco_churn_cleaned.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    if df.empty or df["Churn"].isna().any():
        raise ValueError("Dataset is empty or contains missing targets.")

    if set(df["Churn"].unique()) != {"Yes", "No"}:
        raise ValueError("Unexpected churn target labels.")

    if df["customerID"].isna().any() or df["customerID"].duplicated().any():
        raise ValueError("Customer IDs are missing or duplicated.")

    y = df["Churn"].map({"No": 0, "Yes": 1})
    X = df.drop(columns=["customerID", "Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}
    trained_pipelines = {}

    print("\n--- DATA SPLIT ---")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print("Stratified split completed.")

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }

        results[name] = metrics
        trained_pipelines[name] = pipeline

        print(f"\n--- {name.upper()} ---")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        print("\nClassification report:")
        print(classification_report(
            y_test,
            predictions,
            target_names=["No Churn", "Churn"],
            zero_division=0,
        ))

        print("Confusion matrix:")
        print(confusion_matrix(y_test, predictions))

    best_name = max(
        results,
        key=lambda name: results[name]["roc_auc"],
    )

    best_pipeline = trained_pipelines[best_name]

    joblib.dump(best_pipeline, MODEL_DIR / "churn_model.joblib")

    report = {
        "best_model": best_name,
        "selection_metric": "roc_auc",
        "test_size": 0.20,
        "random_state": 42,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "models": results,
    }

    with open(REPORT_DIR / "model_metrics.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"\nBest model by ROC-AUC: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'churn_model.joblib'}")
    print(f"Saved metrics: {REPORT_DIR / 'model_metrics.json'}")
    print("\nModel training and evaluation completed.")


if __name__ == "__main__":
    main()
