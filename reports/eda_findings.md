# ChurnGuard — Exploratory Data Analysis Findings

## 1. Dataset Overview

- **Dataset:** IBM Telco Customer Churn
- **Total customers:** 7,043
- **Total features:** 21
- **Customers who churned:** 1,869
- **Customers retained:** 5,174
- **Overall churn rate:** 26.54%

## 2. Key Findings

### Contract Type

| Contract | Churn Rate |
|---|---:|
| Month-to-month | 42.71% |
| One year | 11.27% |
| Two year | 2.83% |

Customers with month-to-month contracts have the highest churn rate.

### Customer Tenure

| Tenure Group | Churn Rate |
|---|---:|
| 0–12 months | 47.44% |
| 13–24 months | 28.71% |
| 25–48 months | 20.39% |
| 49–60 months | 14.42% |
| 61–72 months | 6.61% |

Churn decreases as customer tenure increases.

### Internet Service

| Internet Service | Churn Rate |
|---|---:|
| Fiber optic | 41.89% |
| DSL | 18.96% |
| No internet service | 7.40% |

Fiber-optic customers have the highest churn rate among the three internet-service categories.

### Payment Method

| Payment Method | Churn Rate |
|---|---:|
| Electronic check | 45.29% |
| Mailed check | 19.11% |
| Bank transfer (automatic) | 16.71% |
| Credit card (automatic) | 15.24% |

Electronic-check customers have the highest churn rate.

### Monthly Charges

| Churn Status | Mean Monthly Charges | Median Monthly Charges |
|---|---:|---:|
| Retained | 61.27 | 64.43 |
| Churned | 74.44 | 79.65 |

Customers who churned had higher average and median monthly charges in this dataset.

## 3. Business Implications

- Prioritize retention efforts for new customers.
- Investigate the reasons for higher churn among month-to-month customers.
- Examine pricing, service quality, and support issues affecting fiber-optic customers.
- Evaluate whether automatic payment adoption is associated with improved retention.
- Investigate whether higher monthly charges contribute to churn.

## 4. Limitations

These findings describe associations in the historical dataset. They do not establish that any individual factor directly causes churn. Further statistical analysis and predictive modeling are required.

## 5. Generated Visualizations

The following charts are saved in this project's `reports/` directory:

- `churn_distribution.png`
- `churn_by_contract.png`
- `churn_by_tenure.png`
- `monthly_charges_by_churn.png`
- `churn_by_internet_service.png`
- `churn_by_payment_method.png`