
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"

st.set_page_config(
    page_title="ChurnGuard | Customer Retention",
    page_icon="📊",
    layout="wide",
)

st.title("ChurnGuard")
st.subheader("Customer Churn Prediction & Retention Intelligence")
st.write("Estimate customer churn risk using your trained machine learning model.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = load_model()

st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines", ["No", "Yes", "No phone service"]
)
internet = st.sidebar.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)
online_security = st.sidebar.selectbox(
    "Online Security", ["Yes", "No", "No internet service"]
)
online_backup = st.sidebar.selectbox(
    "Online Backup", ["Yes", "No", "No internet service"]
)
device_protection = st.sidebar.selectbox(
    "Device Protection", ["Yes", "No", "No internet service"]
)
tech_support = st.sidebar.selectbox(
    "Tech Support", ["Yes", "No", "No internet service"]
)
streaming_tv = st.sidebar.selectbox(
    "Streaming TV", ["Yes", "No", "No internet service"]
)
streaming_movies = st.sidebar.selectbox(
    "Streaming Movies", ["Yes", "No", "No internet service"]
)
contract = st.sidebar.selectbox(
    "Contract", ["Month-to-month", "One year", "Two year"]
)
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)
monthly_charges = st.sidebar.number_input(
    "Monthly Charges", min_value=0.0, value=70.0
)
total_charges = st.sidebar.number_input(
    "Total Charges", min_value=0.0, value=840.0
)

if st.button("Predict Churn Risk", type="primary"):
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }])

    try:
        probability = float(model.predict_proba(customer)[0, 1])
        prediction = model.predict(customer)[0]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Churn Risk", f"{probability:.1%}")

        with col2:
            st.metric(
                "Prediction",
                "Likely to Churn" if prediction == 1 else "Likely to Stay",
            )

        st.progress(probability)

        if probability >= 0.5:
            st.warning("High predicted churn risk. Consider reviewing retention options.")
        else:
            st.success("Lower predicted churn risk.")

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
