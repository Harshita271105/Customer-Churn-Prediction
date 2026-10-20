
from pathlib import Path

import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("\n--- DATASET OVERVIEW ---")
    print(f"File: {DATA_PATH.name}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n--- COLUMN NAMES AND DATA TYPES ---")
    print(df.dtypes.to_string())

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum().to_string())

    print("\n--- DUPLICATE ROWS ---")
    print(df.duplicated().sum())

    print("\n--- CHURN DISTRIBUTION ---")
    print(df["Churn"].value_counts(dropna=False).to_string())

    print("\n--- CUSTOMER ID VALIDATION ---")
    print(f"Unique IDs: {df['customerID'].nunique()}")
    print(f"Missing IDs: {df['customerID'].isnull().sum()}")

    print("\n--- TOTALCHARGES VALIDATION ---")
    total_charges = df["TotalCharges"].astype(str).str.strip()
    charges_numeric = pd.to_numeric(total_charges, errors="coerce")
    invalid_charges = charges_numeric.isna()

    print(f"Blank strings: {(total_charges == '').sum()}")
    print(f"Non-numeric values: {invalid_charges.sum()}")

    print("\n--- AFFECTED CUSTOMER RECORDS ---")
    affected_rows = df.loc[
        invalid_charges,
        [
            "customerID",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Churn",
        ],
    ]
    print(affected_rows.to_string(index=False))

    print("\n--- CATEGORICAL COLUMN SUMMARY ---")
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:
        blank_count = (
            df[column].astype(str).str.strip() == ""
        ).sum()

        print(
            f"{column}: {df[column].nunique()} unique values, "
            f"{blank_count} blank strings"
        )


if __name__ == "__main__":
    main()
