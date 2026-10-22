
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

from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
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

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


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
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
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
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "recall": "recall",
    }

    cv_results = {}
    pipelines = {}

    print("\n--- TRAINING / TEST SPLIT ---")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Cross-validation folds: {CV_FOLDS}")

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )

        cv_results[name] = {
            metric: {
                "mean": float(scores[f"test_{metric}"].mean()),
                "std": float(scores[f"test_{metric}"].std()),
            }
            for metric in scoring
        }

        pipelines[name] = pipeline

        print(f"\n--- {name.upper()} CROSS-VALIDATION ---")
        for metric, values in cv_results[name].items():
            print(
                f"{metric}: {values['mean']:.4f} "
                f"+/- {values['std']:.4f}"
            )

    best_name = max(
        cv_results,
        key=lambda name: cv_results[name]["roc_auc"]["mean"],
    )

    print(f"\nSelected using training CV ROC-AUC: {best_name}")

    # Fit the selected pipeline on all training records.
    best_pipeline = pipelines[best_name]
    best_pipeline.fit(X_train, y_train)

    # Evaluate on the held-out test set only after model selection.
    probabilities = best_pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    test_metrics = {
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

    print("\n--- FINAL HELD-OUT TEST RESULTS (THRESHOLD 0.50) ---")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No Churn", "Churn"],
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    model_path = MODEL_DIR / "churn_model.joblib"
    joblib.dump(best_pipeline, model_path)

    report = {
        "best_model": best_name,
        "selection_metric": "mean cross-validation ROC-AUC",
        "cross_validation_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "cross_validation": cv_results,
        "final_test_threshold": 0.50,
        "final_test_metrics": test_metrics,
    }

    report_path = REPORT_DIR / "model_metrics.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {report_path}")
    print("\nModel selection and final test evaluation completed.")


if __name__ == "__main__":
    main()
