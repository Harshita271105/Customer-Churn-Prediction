
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "telco_churn_cleaned.csv"


def main():
    df = pd.read_csv(RAW_PATH)

    print("Original dataset shape:", df.shape)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(),
        errors="coerce"
    )

    # Validate missing values before cleaning
    missing_mask = df["TotalCharges"].isna()

    if missing_mask.any():
        if not (df.loc[missing_mask, "tenure"] == 0).all():
            raise ValueError(
                "Missing TotalCharges found for customers with nonzero tenure."
            )

        # Business assumption: zero tenure implies no accumulated charges
        df.loc[missing_mask, "TotalCharges"] = 0.0

    # Validate the cleaned dataset
    assert df["TotalCharges"].isna().sum() == 0
    assert df["customerID"].is_unique
    assert df["Churn"].isin(["Yes", "No"]).all()
    assert len(df) == 7043

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Cleaned dataset shape:", df.shape)
    print("Missing TotalCharges handled:", missing_mask.sum())
    print("Remaining missing values:", df.isnull().sum().sum())
    print("Cleaned dataset saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
