# Exploratory Data Analysis (EDA) Findings

## Dataset Overview

- **Project:** ChurnGuard — Customer Churn Prediction
- **Dataset:** Telco Customer Churn dataset from Kaggle
- **Total customer records:** 7,043
- **Total features:** 21
- **Target variable:** `Churn`

## Key Findings

### 1. Overall Customer Churn

- Customers who churned: 1,869
- Customers who stayed: 5,174
- Overall churn rate: 26.54%

Approximately one in four customers in the dataset discontinued their service.

### 2. Contract Type

- Month-to-month contracts: 42.71% churn rate
- One-year contracts: 11.27% churn rate
- Two-year contracts: 2.83% churn rate

Customers with month-to-month contracts had substantially higher churn rates than customers with longer contracts.

### 3. Customer Tenure

- 0–12 months: 47.44% churn rate
- 13–24 months: 28.71%
- 25–48 months: 20.39%
- 49–60 months: 14.42%
- 61–72 months: 6.61%

Churn decreased as customer tenure increased, making early customer retention an important area to investigate.

### 4. Internet Service

- Fiber optic: 41.89% churn rate
- DSL: 18.96%
- No internet service: 7.40%

Fiber-optic customers had the highest churn rate among these internet-service categories.

### 5. Payment Method

- Electronic check: 45.29% churn rate
- Mailed check: 19.11%
- Bank transfer (automatic): 16.71%
- Credit card (automatic): 15.24%

Customers using electronic checks had the highest churn rate among the listed payment methods.

### 6. Monthly Charges

- Average monthly charges for customers who stayed: 61.27
- Average monthly charges for customers who churned: 74.44
- Median monthly charges for customers who stayed: 64.43
- Median monthly charges for customers who churned: 79.65

Customers who churned had higher average and median monthly charges in this dataset.

## Business Implications

1. Investigate retention strategies for month-to-month customers.
2. Prioritize early-tenure customer engagement.
3. Investigate pricing, service quality, and support experiences among fiber-optic customers.
4. Review the customer experience associated with electronic-check payments.
5. Evaluate whether higher monthly charges are associated with customer dissatisfaction.

## Important Note

These findings describe associations in the dataset. They do not establish that any individual feature causes customer churn. The model's predictions are estimates and should be evaluated using appropriate validation metrics.
