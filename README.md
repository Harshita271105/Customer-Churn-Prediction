# ChurnGuard — Customer Churn Prediction and Retention Intelligence System

## Overview

ChurnGuard is a machine learning project designed to identify customers who may be at risk of leaving a service. It aims to support data-driven customer retention decisions through predictive modelling, model evaluation, and an interactive application.

**Project status:** In development.

## Objectives

- Explore and validate a public customer churn dataset.
- Perform exploratory data analysis (EDA).
- Build a reproducible machine learning pipeline.
- Compare models using appropriate evaluation metrics.
- Interpret model predictions and identify relevant patterns.
- Develop an interactive Streamlit application.
- Add automated tests and maintain project documentation.

## Technology Stack

- Python
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- Streamlit and Plotly
- pytest and joblib
- Git and GitHub

## Dataset

The project plans to use the IBM Telco Customer Churn dataset.

Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Dataset validation, attribution requirements, and data-quality findings will be documented after the dataset has been obtained and inspected.

## Project Structure

```text
Customer-Churn-Prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── models/
├── tests/
├── reports/
├── .gitignore
└── README.md
```

## Current Progress

- [x] Local project repository initialized.
- [x] Virtual environment configured.
- [x] Initial dependencies installed.
- [x] Project folder structure created.
- [x] `.gitignore` added.
- [ ] Dataset acquisition and validation.
- [ ] Exploratory data analysis.
- [ ] Model training and evaluation.
- [ ] Prediction pipeline and Streamlit application.
- [ ] Automated testing and deployment.

## Limitations

ChurnGuard is intended to estimate churn risk, not guarantee that a customer will leave or that a retention intervention will succeed. Model performance and business usefulness will be assessed using actual experimental results.

## License

A project license will be selected after reviewing the dataset's applicable licensing and attribution requirements.
