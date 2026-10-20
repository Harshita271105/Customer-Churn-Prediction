
from pathlib import Path

import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "telco_churn_cleaned.csv"
)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("\n--- CLEANED DATASET VALIDATION ---")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("Duplicate rows:", df.duplicated().sum())
    print("Missing values:", df.isnull().sum().sum())
    print("Unique customer IDs:", df["customerID"].nunique())

    print("\n--- TOTALCHARGES VALIDATION ---")
    print("Data type:", df["TotalCharges"].dtype)
    print("Missing values:", df["TotalCharges"].isna().sum())
    print("Zero TotalCharges:", (df["TotalCharges"] == 0).sum())

    print("\n--- CHURN DISTRIBUTION ---")
    print(df["Churn"].value_counts().to_string())

    assert df.shape == (7043, 21)
    assert df["customerID"].is_unique
    assert df.isnull().sum().sum() == 0
    assert df["TotalCharges"].dtype.kind in "fi"
    assert df["Churn"].isin(["Yes", "No"]).all()

    print("\nAll validation assertions passed.")


if __name__ == "__main__":
    main()
