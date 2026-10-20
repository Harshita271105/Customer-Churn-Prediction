
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "telco_churn_cleaned.csv"
)

REPORTS_DIR = BASE_DIR / "reports"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    sns.set_theme(style="whitegrid")

    # 1. Overall churn distribution
    churn_counts = df["Churn"].value_counts()

    print("\n--- CHURN DISTRIBUTION ---")
    print(churn_counts.to_string())

    churn_rate = (df["Churn"] == "Yes").mean() * 100
    print(f"\nOverall churn rate: {churn_rate:.2f}%")

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Churn", hue="Churn", legend=False)
    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "churn_distribution.png", dpi=300)
    plt.close()

    # 2. Churn by contract type
    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"],
        normalize="index"
    ) * 100

    print("\n--- CHURN RATE BY CONTRACT TYPE (%) ---")
    print(contract_churn.round(2).to_string())

    plt.figure(figsize=(9, 5))
    sns.countplot(
        data=df,
        x="Contract",
        hue="Churn",
        order=["Month-to-month", "One year", "Two year"]
    )
    plt.title("Customer Churn by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "churn_by_contract.png", dpi=300)
    plt.close()

    # 3. Monthly charges by churn status
    print("\n--- MONTHLY CHARGES BY CHURN STATUS ---")
    print(
        df.groupby("Churn")["MonthlyCharges"]
        .agg(["mean", "median"])
        .round(2)
        .to_string()
    )

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn", legend=False)
    plt.title("Monthly Charges by Churn Status")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "monthly_charges_by_churn.png", dpi=300)
    plt.close()

    print("\nEDA charts saved in:", REPORTS_DIR)
    print("Exploratory analysis completed.")


    # 4. Churn rate by customer tenure
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 72],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49-60 months",
            "61-72 months",
        ],
    )

    tenure_churn = (
        df.groupby("TenureGroup", observed=False)["Churn"]
        .apply(lambda values: (values == "Yes").mean() * 100)
    )

    print("\n--- CHURN RATE BY TENURE GROUP (%) ---")
    print(tenure_churn.round(2).to_string())

    plt.figure(figsize=(9, 5))
    sns.barplot(
        x=tenure_churn.index.astype(str),
        y=tenure_churn.values,
    )
    plt.title("Customer Churn Rate by Tenure Group")
    plt.xlabel("Customer Tenure")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(
        REPORTS_DIR / "churn_by_tenure.png",
        dpi=300,
    )
    plt.close()

    print("\nTenure analysis chart saved.")


    # 5. Churn rate by internet service
    internet_churn = (
        df.groupby("InternetService")["Churn"]
        .apply(lambda values: (values == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )

    print("\n--- CHURN RATE BY INTERNET SERVICE (%) ---")
    print(internet_churn.round(2).to_string())

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=internet_churn.index,
        y=internet_churn.values,
    )
    plt.title("Customer Churn Rate by Internet Service")
    plt.xlabel("Internet Service")
    plt.ylabel("Churn Rate (%)")
    plt.tight_layout()
    plt.savefig(
        REPORTS_DIR / "churn_by_internet_service.png",
        dpi=300,
    )
    plt.close()

    print("Internet service analysis chart saved.")


    # 6. Churn rate by payment method
    payment_churn = (
        df.groupby("PaymentMethod")["Churn"]
        .apply(lambda values: (values == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )

    print("\n--- CHURN RATE BY PAYMENT METHOD (%) ---")
    print(payment_churn.round(2).to_string())

    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=payment_churn.index,
        y=payment_churn.values,
    )
    plt.title("Customer Churn Rate by Payment Method")
    plt.xlabel("Payment Method")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(
        REPORTS_DIR / "churn_by_payment_method.png",
        dpi=300,
    )
    plt.close()

    print("Payment method analysis chart saved.")

if __name__ == "__main__":
    main()
