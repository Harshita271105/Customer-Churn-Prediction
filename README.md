# 🛡️ ChurnGuard — Customer Churn Prediction & Retention Intelligence

ChurnGuard is a machine learning application that predicts customer churn risk and explores customer-retention patterns using the Telco Customer Churn dataset.

The project combines data cleaning, data validation, exploratory data analysis (EDA), machine learning pipelines, model evaluation, permutation-based feature importance, and an interactive Streamlit dashboard.

## 🚀 Live Demo

Try the deployed application:

**[Launch ChurnGuard](https://customer-churn-prediction-kpbazmnzgxsg4gkjbh6tbe.streamlit.app/)**

## ✨ Features

- **Customer Analytics:** View customer counts, churn rate, and retention statistics.
- **Churn Prediction:** Estimate a customer's churn probability from service, contract, billing, and demographic details.
- **Exploratory Data Analysis:** Explore churn patterns by contract type, tenure, internet service, payment method, and monthly charges.
- **Model Explainability:** Visualize permutation feature importance to understand which input features contribute most to predictive performance.
- **Machine Learning Pipeline:** Apply consistent preprocessing to numerical and categorical features using scikit-learn pipelines.
- **Model Evaluation:** Review accuracy, precision, recall, F1-score, ROC-AUC, and confusion-matrix results.
- **Threshold-Based Classification:** Use a selected decision threshold of 0.60 to classify predicted churn probabilities.
- **Automated Testing:** Run tests to verify model loading and prediction functionality.
- **Interactive Dashboard:** Explore analytics, predictions, and insights through a Streamlit web application.
- **Individual Customer Explanations:** Display rule-based risk indicators for an individual customer alongside global feature importance.
- **Model Explainability:** Analyze global feature importance using permutation importance evaluated with ROC-AUC.
- **Cross-Validated Model Selection:** Compares Logistic Regression and Random Forest using stratified cross-validation and ROC-AUC.
- **Customer-Level Risk Context:** Presents rule-based customer risk indicators alongside predicted churn probability.
- **Feature Importance:** Uses permutation importance to identify features associated with overall model performance.
- **Statistical Data-Drift Monitoring:** Compares reference and comparison samples using Kolmogorov–Smirnov and chi-square tests, with Benjamini–Hochberg correction.
- **Automated Testing:** Includes pytest coverage for prediction outputs, customer explanations, feature-importance reports, and monitoring reports.

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Data processing and machine learning |
| pandas and NumPy | Data manipulation and numerical operations |
| scikit-learn | Preprocessing, model training, evaluation, and permutation importance |
| Matplotlib and Seaborn | Data visualization |
| Streamlit | Interactive web dashboard |
| joblib | Saving and loading the trained model |
| pytest | Automated testing |
| Git and GitHub | Version control and project hosting |

## 📊 Dataset

This project uses the **Telco Customer Churn** dataset sourced from Kaggle.

- **Customer records:** 7,043
- **Original columns:** 21
- **Churned customers:** 1,869
- **Retained customers:** 5,174
- **Observed churn rate:** 26.54%

These statistics describe the dataset used in this project, not the current customer base of a live telecommunications company.

**Dataset source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Review the dataset's license and attribution requirements before redistributing the raw data.

## 🖥️ Application Screenshots

### Analytics Dashboard

![ChurnGuard analytics dashboard](screenshots/dashboard.png)

### Churn Prediction

![ChurnGuard prediction results](screenshots/prediction-1.png)

![ChurnGuard prediction results](screenshots/prediction-2.png)

![ChurnGuard prediction results](screenshots/prediction-3.png)

### Model Explainability — Feature Importance

Add a screenshot of the new permutation feature-importance chart to the `screenshots/` directory and update the filename below to match the image you save.

![ChurnGuard permutation feature importance](screenshots/feature-importance.png)

## 🔍 Exploratory Data Analysis Highlights

The exploratory analysis identified the following observed churn rates:

| Customer segment | Observed churn rate |
|---|---:|
| Month-to-month contracts | 42.71% |
| One-year contracts | 11.27% |
| Two-year contracts | 2.83% |
| Tenure of 0–12 months | 47.44% |
| Tenure of 61–72 months | 6.61% |
| Fiber-optic internet service | 41.89% |
| Electronic-check payments | 45.29% |

These are observational associations, not proof that a particular contract, service, or payment method causes churn.

## 🤖 Model Evaluation

The project evaluates classification performance using a stratified train-test split and additional cross-validation.

### Recorded baseline model metrics

The following values come from `reports/model_metrics.json`:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 73.81% | 77.08% |
| Precision | 50.43% | 56.06% |
| Recall | 78.34% | 63.10% |
| F1-score | 61.36% | 59.37% |
| ROC-AUC | 84.16% | 82.20% |

Logistic Regression was selected by ROC-AUC in the recorded baseline comparison. Random Forest achieved higher accuracy, while Logistic Regression achieved higher recall and ROC-AUC.

### Decision threshold validation

The validation workflow selects a classification threshold using out-of-fold predictions on the training portion of the data. The recorded selected threshold is **0.60**.

At this threshold, the recorded held-out test results were:

| Metric | Result |
|---|---:|
| Accuracy | 76.22% |
| Precision | 53.99% |
| Recall | 70.59% |
| F1-score | 61.18% |

Recorded confusion matrix:

| | Predicted: No Churn | Predicted: Churn |
|---|---:|---:|
| Actual: No Churn | 810 | 225 |
| Actual: Churn | 110 | 264 |

The threshold changes the trade-off between identifying likely churners and generating false-positive alerts. It should be selected according to the business costs of missed churners and unnecessary retention interventions.

**Important evaluation note:** The baseline model-selection workflow compares candidate models using test-set ROC-AUC. Consequently, the test set has influenced model selection, and the results should not be treated as a completely untouched final estimate of generalization performance. A stronger future evaluation would select models and thresholds using training data or cross-validation, then evaluate the final pipeline once on a separate test set.

Detailed reports are available in:

- `reports/model_metrics.json`
- `reports/model_validation.json`
- `reports/eda_findings.md`

## 📈 Model Explainability — Permutation Feature Importance

ChurnGuard includes a permutation feature-importance analysis based on ROC-AUC.

The method measures how much predictive performance decreases when a feature is randomly shuffled while the other features remain unchanged. A larger decrease indicates that the model relies more heavily on that feature for its predictions.

The current analysis highlights:

1. **Tenure**
2. **InternetService**
3. **MonthlyCharges**
4. **Contract**

Other features, including TotalCharges, StreamingMovies, OnlineSecurity, and PaymentMethod, also appear in the recorded results.

These rankings describe predictive importance for this fitted model and evaluation sample. They do not establish causation, show the direction of an individual prediction's effect, or guarantee that the same rankings will hold for another dataset.

The script is located at `src/feature_importance.py`, and its output is saved to `reports/feature_importance.json`.

## 📁 Project Structure

```text
Customer-Churn-Prediction/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── churn_model.joblib
├── reports/
│   ├── eda_findings.md
│   ├── feature_importance.json
│   ├── model_metrics.json
│   ├── model_validation.json
│   └── *.png
├── screenshots/
│   ├── dashboard.png
│   ├── prediction-1.png
│   ├── prediction-2.png
│   ├── prediction-3.png
│   └── feature-importance.png
├── src/
│   ├── inspect_data.py
│   ├── clean_data.py
│   ├── validate_cleaned_data.py
│   ├── eda.py
│   ├── prepare_features.py
│   ├── evaluate_model.py
│   ├── feature_importance.py
│   └── predict.py
├── tests/
│   └── test_model.py
├── app.py
├── requirements.txt
├── pytest.ini
└── README.md
```

*This is a representative structure; it omits some generated files and folders.*

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Harshita271105/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

### 5. Run automated tests

```bash
python -m pytest tests/ -v
```

### 6. Recalculate feature importance

Ensure the cleaned dataset and trained model are available, then run:

```bash
python src/feature_importance.py
```

The results will be saved to `reports/feature_importance.json`.

## ⚠️ Limitations

- Predictions depend on the training data and model quality and may not generalize to every customer population.
- Predicted churn probabilities are estimates, not guarantees.
- Feature importance indicates predictive usefulness, not causation.
- The selected classification threshold reflects a model-validation procedure and may not be optimal for every business.
- Retention suggestions should be evaluated against actual business outcomes.
- Probability calibration, independent external validation, and ongoing monitoring would be needed before production use.

## 🔮 Future Improvements

- Strengthen model selection and final evaluation using a fully independent test set.
- Investigate probability calibration and business-cost-based threshold selection.
- Add individual-prediction explanations using SHAP or another suitable explainability method.
- Add model monitoring and data-drift detection.
- Evaluate retention interventions using observed customer outcomes.
- Improve deployment monitoring and add automated checks for the live application.

## 👩‍💻 Author

**Harshita Verma**

- **GitHub:** [Harshita271105](https://github.com/Harshita271105)
- **Project repository:** [Customer-Churn-Prediction](https://github.com/Harshita271105/Customer-Churn-Prediction)
