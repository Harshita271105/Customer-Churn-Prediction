
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_cleaned.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "feature_importance.json"

TARGET = "Churn"
RANDOM_STATE = 42


def main():
    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)

    X = data.drop(columns=[TARGET, "customerID"], errors="ignore")
    y = data[TARGET].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    importance.to_json(
        OUTPUT_PATH,
        orient="records",
        indent=2,
    )

    print("\nTop 10 features influencing ROC-AUC:")
    print(importance.head(10).to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
