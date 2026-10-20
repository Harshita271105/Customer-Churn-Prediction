# ChurnGuard — Initial Data Validation Report

## Dataset

- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Source:** [IBM Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows:** 7,043
- **Columns:** 21

## Verified Findings

- The dataset loaded successfully using pandas.
- No duplicate rows were detected.
- The `Churn` target contains two observed classes: `No` and `Yes`.
- Churn counts: 5,174 `No` and 1,869 `Yes`.
- `TotalCharges` was read as a string rather than a numeric field.
- Eleven `TotalCharges` entries are blank and cannot be converted to numbers.
- The blank `TotalCharges` entries require further investigation before preprocessing.
- The original CSV remains unchanged.

## Next Actions

1. Validate column types and suspicious values.
2. Investigate the blank `TotalCharges` records.
3. Confirm dataset attribution and licensing requirements.
4. Begin exploratory data analysis after data validation is complete.

## Limitations

These findings represent the initial inspection only. They do not establish model performance, causal relationships, or the effectiveness of retention interventions.

## Additional Validation Findings

- The `customerID` column contains 7,043 unique identifiers.
- The `Churn` target contains two observed classes: `No` and `Yes`.
- The inspected categorical service and billing columns contain no blank strings.
- `TotalCharges` contains 11 blank strings that cannot be converted to numeric values.
- The affected records require investigation before a preprocessing decision is made.
- No changes have been made to the original Kaggle CSV.

