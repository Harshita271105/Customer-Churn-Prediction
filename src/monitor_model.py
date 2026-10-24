
from pathlib import Path
import json

import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_churn_cleaned.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "data_drift_report.json"
)

REFERENCE_ROWS = 1000
RANDOM_STATE = 42
SIGNIFICANCE_LEVEL = 0.05


def calculate_feature_drift(reference, current, column):
    """Compare one feature's distributions between two samples."""

    reference_values = reference[column].dropna()
    current_values = current[column].dropna()

    if reference_values.empty or current_values.empty:
        return {
            "type": (
                "numeric"
                if pd.api.types.is_numeric_dtype(reference[column])
                else "categorical"
            ),
            "p_value": None,
            "drift_detected": False,
            "status": "insufficient_data",
        }

    if pd.api.types.is_numeric_dtype(reference[column]):
        statistic, p_value = ks_2samp(
            reference_values,
            current_values,
        )

        return {
            "type": "numeric",
            "reference_mean": float(reference_values.mean()),
            "current_mean": float(current_values.mean()),
            "mean_difference": float(
                current_values.mean() - reference_values.mean()
            ),
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
        }

    reference_values = reference_values.astype(str)
    current_values = current_values.astype(str)

    categories = sorted(
        set(reference_values.unique())
        | set(current_values.unique())
    )

    reference_counts = reference_values.value_counts().reindex(
        categories, fill_value=0
    )
    current_counts = current_values.value_counts().reindex(
        categories, fill_value=0
    )

    contingency = pd.DataFrame({
        "reference": reference_counts,
        "current": current_counts,
    })

    # A feature with only one category cannot support this test.
    if len(categories) < 2:
        return {
            "type": "categorical",
            "p_value": None,
            "drift_detected": False,
            "status": "insufficient_categories",
        }

    statistic, p_value, _, _ = chi2_contingency(
        contingency.to_numpy()
    )

    return {
        "type": "categorical",
        "reference_distribution": {
            category: float(reference_counts[category] / len(reference_values))
            for category in categories
        },
        "current_distribution": {
            category: float(current_counts[category] / len(current_values))
            for category in categories
        },
        "chi_square_statistic": float(statistic),
        "p_value": float(p_value),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)

    if len(data) <= REFERENCE_ROWS:
        raise ValueError(
            "Not enough rows to create reference and comparison samples."
        )

    # Demonstration only: both samples come from the same historical dataset.
    reference = data.sample(
        n=REFERENCE_ROWS,
        random_state=RANDOM_STATE,
    )

    remaining = data.drop(index=reference.index)

    current = remaining.sample(
        n=min(REFERENCE_ROWS, len(remaining)),
        random_state=RANDOM_STATE,
    )

    features = [
        column
        for column in data.columns
        if column not in ["customerID", "Churn"]
    ]

    drift_results = {}

    for column in features:
        drift_results[column] = calculate_feature_drift(
            reference,
            current,
            column,
        )

    # Correct p-values across tested features to reduce false positives.
    tested_features = [
        column
        for column, result in drift_results.items()
        if result.get("p_value") is not None
    ]

    if tested_features:
        raw_p_values = [
            drift_results[column]["p_value"]
            for column in tested_features
        ]

        rejected, adjusted_p_values, _, _ = multipletests(
            raw_p_values,
            alpha=SIGNIFICANCE_LEVEL,
            method="fdr_bh",
        )

        for column, adjusted_p, is_drift in zip(
            tested_features,
            adjusted_p_values,
            rejected,
        ):
            drift_results[column]["adjusted_p_value"] = float(adjusted_p)
            drift_results[column]["drift_detected"] = bool(is_drift)
            drift_results[column]["status"] = (
                "potential_drift"
                if is_drift
                else "no_statistically_significant_drift_detected"
            )

    drifted_features = [
        column
        for column, result in drift_results.items()
        if result.get("drift_detected", False)
    ]

    report = {
        "monitoring_type": "statistical distribution comparison",
        "data_source": "two samples from the same cleaned historical dataset",
        "reference_rows": len(reference),
        "current_rows": len(current),
        "significance_level": SIGNIFICANCE_LEVEL,
        "multiple_testing_correction": "Benjamini-Hochberg FDR",
        "drifted_feature_count": len(drifted_features),
        "drifted_features": drifted_features,
        "warning": (
            "This is a demonstration, not evidence of production drift. "
            "Statistical significance does not establish practical importance "
            "or prove that model performance has degraded."
        ),
        "features": drift_results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n--- DATA DRIFT MONITORING ---")
    print(f"Reference rows: {len(reference)}")
    print(f"Comparison rows: {len(current)}")
    print(f"Features tested: {len(tested_features)}")
    print(f"Potentially drifted features: {len(drifted_features)}")

    if drifted_features:
        print("Flagged features:", ", ".join(drifted_features))
    else:
        print("No statistically significant drift detected.")

    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
