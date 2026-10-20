
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

    print("\n--- COLUMN NAMES ---")
    print(df.columns.tolist())

    print("\n--- DATA TYPES ---")
    print(df.dtypes.to_string())

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum().to_string())

    print("\n--- DUPLICATE ROWS ---")
    print(df.duplicated().sum())

    print("\n--- FIRST 5 ROWS ---")
    print(df.head().to_string())

    print("\n--- POSSIBLE CHURN TARGET ---")
    if "Churn" in df.columns:
        print(df["Churn"].value_counts(dropna=False).to_string())
    else:
        print("No column named 'Churn' found.")

    print("\n--- CUSTOMER ID CHECK ---")
    if "customerID" in df.columns:
        print(f"Unique customer IDs: {df['customerID'].nunique()}")
        print(f"Missing customer IDs: {df['customerID'].isnull().sum()}")
    else:
        print("No column named 'customerID' found.")

    print("\n--- DATA QUALITY CHECKS ---")

    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nTotal rows and columns:")
    print(df.shape)

    print("\n--- TOTALCHARGES INSPECTION ---")
    if "TotalCharges" in df.columns:
        total_charges = df["TotalCharges"].astype(str).str.strip()

        print("Data type:", df["TotalCharges"].dtype)
        print("Blank strings:", (total_charges == "").sum())
        print("Whitespace-only strings:", total_charges.eq("").sum())
    else:
        print("TotalCharges column not found.")



    print("\n--- MISSING AND INVALID VALUE CHECK ---")

    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())

    print("\nTotalCharges conversion check:")
    charges_numeric = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(),
        errors="coerce"
    )

    invalid_charges = charges_numeric.isna()

    print("Values that cannot be converted to numbers:", invalid_charges.sum())
    print("Affected customer IDs:")
    print(df.loc[invalid_charges, "customerID"].to_string(index=False))


if __name__ == "__main__":
    main()
