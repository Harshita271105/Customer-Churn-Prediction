# 🛡️ ChurnGuard — Customer Churn Prediction & Retention Intelligence

ChurnGuard is a machine learning application that predicts customer churn risk and explores customer-retention patterns using the Telco Customer Churn dataset.

It combines data validation, exploratory data analysis, machine learning pipelines, model evaluation, and an interactive Streamlit dashboard.

## Live Demo

Try ChurnGuard here: https://customer-churn-prediction-kpbazmnzgxsg4gkjbh6tbe.streamlit.app/
## ✨ Features

- **Customer Analytics:** View customer counts, churn rate, and retention statistics.
- **Churn Prediction:** Estimate churn probability from customer, service, contract, and billing details.
- **Exploratory Data Analysis:** Explore churn patterns by contract, tenure, internet service, payment method, and monthly charges.
- **Machine Learning Pipeline:** Preprocess numerical and categorical features consistently with the trained model.
- **Model Evaluation:** Compare classification metrics and inspect model validation reports.
- **Automated Tests:** Verify that the saved model loads and the prediction interface works.

## 🧰 Tech Stack

- Python
- pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn
- Streamlit
- joblib
- pytest

## 📊 Dataset

This project uses the Telco Customer Churn dataset sourced from Kaggle.

- **Customer records:** 7,043
- **Original columns:** 21
- **Churned customers:** 1,869
- **Retained customers:** 5,174
- **Observed churn rate:** 26.54%

These figures describe the dataset used in this project, not a live telecommunications company's current customer base.

Dataset source: [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Review the dataset's license and attribution requirements before redistributing the raw data.

## 🖥️ Application Screenshots

### Analytics Dashboard

![ChurnGuard analytics dashboard](screenshots/dashboard.png)

### Churn Prediction

![ChurnGuard prediction results](screenshots/prediction-1.png)
![ChurnGuard prediction results](screenshots/prediction-2.png)
![ChurnGuard prediction results](screenshots/prediction-3.png)

## 📁 Project Structure

Customer-Churn-Prediction/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── churn_model.joblib
├── reports/
│   ├── eda_findings.md
│   ├── model_metrics.json
│   └── model_validation.json
├── screenshots/
│   ├── dashboard.png
│   └── prediction.png
├── src/
│   ├── inspect_data.py
│   ├── clean_data.py
│   ├── validate_cleaned_data.py
│   ├── eda.py
│   ├── prepare_features.py
│   ├── evaluate_model.py
│   └── predict.py
├── tests/
│   └── test_model.py
├── app.py
├── requirements.txt
└── README.md



## 🚀 Run Locally

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

## 📈 Analysis Highlights

The initial exploratory analysis found the following churn rates:

| Segment | Observed churn rate |
|---|---:|
| Month-to-month contracts | 42.71% |
| One-year contracts | 11.27% |
| Two-year contracts | 2.83% |
| Customer tenure of 0–12 months | 47.44% |
| Customer tenure of 61–72 months | 6.61% |
| Fiber-optic internet service | 41.89% |
| Electronic-check payments | 45.29% |

These are observational associations. They do not establish that a particular service, payment method, or contract causes churn.

## 🧪 Model Evaluation

The project compares machine learning classifiers using a held-out test set and additional validation.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

See the generated reports in `reports/` for the actual recorded results. Model quality should be judged using those results, not assumed from the existence of a trained model.

## ⚠️ Limitations

- Predictions depend on the training data and may not generalize to every customer population.
- A churn probability is an estimate, not a guarantee.
- Retention suggestions are general prompts and should be evaluated against real business outcomes.
- Further calibration, monitoring, and external validation would be needed before production use.

## 🔮 Future Improvements

- Add SHAP-based model explanations.
- Improve probability calibration and threshold selection.
- Add model monitoring and drift detection.
- Evaluate retention interventions using real outcomes.
- Deploy the application publicly.

## 👩‍💻 Author

**Harshita Verma**

GitHub: [Harshita271105](https://github.com/Harshita271105)
