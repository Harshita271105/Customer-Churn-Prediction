
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "telco_churn_cleaned.csv"
)
REPORTS_PATH = PROJECT_ROOT / "reports"
CHURN_THRESHOLD = 0.60
# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="ChurnGuard | Retention Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    .stApp {
        background: #0b1120;
        color: #e5e7eb;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #273449;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f9fafb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .hero {
        background: linear-gradient(120deg, #172554, #0f766e);
        border: 1px solid #285e70;
        border-radius: 20px;
        padding: 30px 32px;
        margin-bottom: 25px;
    }

    .hero-tag {
        color: #99f6e4;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 38px;
        font-weight: 800;
        margin: 8px 0;
    }

    .hero p {
        color: #d1fae5;
        font-size: 15px;
        margin-bottom: 0;
    }

    .section-label {
        color: #5eead4;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 12px;
        margin-bottom: 5px;
    }

    .section-heading {
        color: #f9fafb;
        font-size: 25px;
        font-weight: 750;
        margin-bottom: 16px;
    }

    .metric-card {
        background: linear-gradient(145deg, #172033, #111827);
        border: 1px solid #29364b;
        border-radius: 15px;
        padding: 20px;
        min-height: 125px;
    }

    .metric-title {
        color: #9ca3af;
        font-size: 13px;
        font-weight: 500;
    }

    .metric-value {
        color: #f9fafb;
        font-size: 30px;
        font-weight: 800;
        margin-top: 9px;
    }

    .metric-accent {
        color: #5eead4;
    }

    .panel {
        background: #111827;
        border: 1px solid #29364b;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 14px;
    }

    .panel-title {
        color: #f9fafb;
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .muted {
        color: #9ca3af;
        font-size: 13px;
    }

    .risk-number {
        font-size: 44px;
        font-weight: 800;
        color: #f9fafb;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #0d9488, #14b8a6);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        min-height: 48px;
        transition: 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #0f766e;
        border: none;
        color: white;
    }

    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #29364b;
        padding: 15px;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #9ca3af;
    }

    div[data-testid="stMetricValue"] {
        color: #f9fafb;
    }

    div[data-baseweb="tab-list"] {
        gap: 8px;
    }

    button[data-baseweb="tab"] {
        border-radius: 8px;
    }

    hr {
        border-color: #29364b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Expected location: "
        "models/churn_model.joblib"
    )
    st.stop()

model = load_model()

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)


df = None

if DATA_PATH.exists():
    try:
        df = load_dataset(str(DATA_PATH))
    except Exception as exc:
        st.warning(f"Could not load the cleaned dataset: {exc}")

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-tag">Customer Intelligence Platform</div>
        <h1>🛡️ ChurnGuard</h1>
        <p>
            Predict customer churn. Discover retention opportunities.
            Make data-informed decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------
overview_tab, prediction_tab, insights_tab, about_tab = st.tabs(
    [
        "📊 Overview",
        "🎯 Churn Prediction",
        "🔎 EDA & Insights",
        "ℹ️ About",
    ]
)

# --------------------------------------------------
# OVERVIEW TAB
# --------------------------------------------------
with overview_tab:
    st.markdown(
        '<div class="section-label">Business Overview</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-heading">Customer Analytics</div>',
        unsafe_allow_html=True,
    )

    if df is not None and "Churn" in df.columns:
        total_customers = len(df)
        churned_customers = int((df["Churn"] == "Yes").sum())
        retained_customers = total_customers - churned_customers
        churn_rate = (
            churned_customers / total_customers * 100
            if total_customers else 0
        )

        metrics = [
            ("TOTAL CUSTOMERS", f"{total_customers:,}", "All records"),
            ("CHURNED CUSTOMERS", f"{churned_customers:,}", "Churn = Yes"),
            ("RETAINED CUSTOMERS", f"{retained_customers:,}", "Churn = No"),
            ("OVERALL CHURN RATE", f"{churn_rate:.2f}%", "Dataset baseline"),
        ]

        columns = st.columns(4)

        for column, (title, value, subtitle) in zip(columns, metrics):
            with column:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">{title}</div>
                        <div class="metric-value">{value}</div>
                        <div class="muted">{subtitle}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.write("")
        left, right = st.columns([1.15, 0.85], gap="large")

        with left:
            st.markdown(
                '<div class="panel-title">Customer Retention Snapshot</div>',
                unsafe_allow_html=True,
            )

            retention_data = pd.DataFrame(
                {
                    "Customer Status": ["Retained", "Churned"],
                    "Customers": [
                        retained_customers,
                        churned_customers,
                    ],
                }
            )

            fig, ax = plt.subplots(figsize=(7, 3.5))
            fig.patch.set_facecolor("#111827")
            ax.set_facecolor("#111827")

            bars = ax.bar(
                retention_data["Customer Status"],
                retention_data["Customers"],
                color=["#14b8a6", "#fb7185"],
                width=0.52,
            )

            ax.set_ylabel("Number of customers", color="#d1d5db")
            ax.tick_params(colors="#d1d5db")
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#374151")
            ax.grid(axis="y", alpha=0.12, color="white")
            ax.set_axisbelow(True)

            for bar in bars:
                ax.annotate(
                    f"{int(bar.get_height()):,}",
                    (
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                    ),
                    ha="center",
                    va="bottom",
                    color="#f9fafb",
                    fontsize=10,
                    xytext=(0, 5),
                    textcoords="offset points",
                )

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with right:
            st.markdown(
                '<div class="panel-title">Retention at a Glance</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="panel">
                    <div class="muted">Customers retained</div>
                    <div class="metric-value metric-accent">
                        {retained_customers / total_customers * 100:.1f}%
                    </div>
                    <div class="muted">
                        Share of customers who stayed
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="panel">
                    <div class="muted">Customers who churned</div>
                    <div class="metric-value" style="color:#fb7185;">
                        {churn_rate:.1f}%
                    </div>
                    <div class="muted">
                        Historical churn rate in this dataset
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                "These metrics describe the dataset; they are not "
                "forecasts of future customer behavior."
            )
    else:
        st.warning(
            "Cleaned dataset unavailable. Check "
            "data/processed/telco_churn_cleaned.csv."
        )

    st.divider()

    st.markdown(
        '<div class="section-heading">Exploratory Data Analysis</div>',
        unsafe_allow_html=True,
    )

    chart_names = [
        ("churn_distribution.png", "Overall Churn Distribution"),
        ("churn_by_contract.png", "Churn by Contract Type"),
        ("churn_by_tenure.png", "Churn by Customer Tenure"),
        ("churn_by_internet_service.png", "Churn by Internet Service"),
        ("churn_by_payment_method.png", "Churn by Payment Method"),
        ("monthly_charges_by_churn.png", "Monthly Charges by Churn"),
    ]

    available_charts = [
        (filename, title)
        for filename, title in chart_names
        if (REPORTS_PATH / filename).exists()
    ]

    if available_charts:
        for start in range(0, len(available_charts), 2):
            columns = st.columns(2, gap="large")

            for column, (filename, title) in zip(
                columns,
                available_charts[start:start + 2],
            ):
                with column:
                    st.markdown(
                        f'<div class="panel-title">{title}</div>',
                        unsafe_allow_html=True,
                    )
                    st.image(
                        str(REPORTS_PATH / filename),
                        use_container_width=True,
                    )
    else:
        st.info("Run src/eda.py to generate the EDA charts.")

# --------------------------------------------------
# PREDICTION TAB
# --------------------------------------------------
with prediction_tab:
    st.markdown(
        '<div class="section-label">Machine Learning</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-heading">Customer Churn Risk Assessment</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "Enter a customer's subscription and billing details to "
        "estimate their churn probability."
    )

    with st.form("customer_prediction_form"):
        st.markdown("#### 01 · Customer Profile")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Has Partner", ["Yes", "No"])

        with col2:
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone Service", ["Yes", "No"])

        with col3:
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"],
            )
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
            )

        st.divider()
        st.markdown("#### 02 · Internet & Services")

        col1, col2, col3 = st.columns(3)

        with col1:
            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"],
            )
            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"],
            )
            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"],
            )

        with col2:
            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"],
            )
            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"],
            )

        with col3:
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"],
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"],
            )

        st.divider()
        st.markdown("#### 03 · Billing Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        with col2:
            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.0,
                step=5.0,
            )

        with col3:
            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=840.0,
                step=50.0,
            )

        submitted = st.form_submit_button(
            "Calculate Churn Risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        customer = pd.DataFrame(
            [
                {
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
                }
            ]
        )

        try:
            probability = float(model.predict_proba(customer)[0, 1])
            prediction = int(probability >= CHURN_THRESHOLD)

            st.divider()
            st.markdown(
                '<div class="section-heading">Prediction Results</div>',
                unsafe_allow_html=True,
            )
            col1, col2 = st.columns([1, 1], gap="large")

            with col1:
                st.markdown(
                    f"""
                    <div class="panel">
                        <div class="muted">PREDICTED CHURN PROBABILITY</div>
                        <div class="risk-number">{probability:.1%}</div>
                        <div class="muted">
                            Estimated probability from the trained model
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(probability)

            with col2:
                if probability >= CHURN_THRESHOLD:
                    st.error("### Higher predicted churn risk")
                    st.write(
                        "Review this customer's service experience "
                        "and consider appropriate retention options."
                    )
                else:
                    st.success("### Lower predicted churn risk")
                    st.write(
                        "The model estimates a lower likelihood of churn. "
                        "Continue monitoring customer satisfaction."
                    )

                st.metric(
                    "Predicted Class",
                    "Likely to Churn"
                    if prediction == 1
                    else "Likely to Stay",
                )

            # Suggestions are based on observed dataset associations.
            st.markdown("#### Retention Opportunities")
            suggestions = []

            if contract == "Month-to-month":
                suggestions.append(
                    "Review whether a suitable longer-term plan "
                    "would benefit this customer."
                )

            if tenure <= 12:
                suggestions.append(
                    "Consider an early-tenure onboarding or "
                    "customer-success check-in."
                )

            if internet == "Fiber optic":
                suggestions.append(
                    "Review internet-service satisfaction, reliability, "
                    "and value for money."
                )

            if payment == "Electronic check":
                suggestions.append(
                    "Review payment experience and communicate "
                    "available payment options."
                )

            if monthly_charges >= 70:
                suggestions.append(
                    "Review the customer's plan, usage, and available "
                    "pricing options."
                )

            if not suggestions:
                suggestions.append(
                    "Continue regular satisfaction checks and "
                    "monitor future customer activity."
                )

            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")

            st.caption(
                f"Classification threshold: {CHURN_THRESHOLD:.0%}. "
                "The probability is a model estimate, not a guarantee. "
                "The threshold was selected using training-set cross-validation "
                "to maximize F1-score. Suggestions are general prompts, "
                "not causal explanations of the prediction."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# --------------------------------------------------
# EDA & INSIGHTS TAB
# --------------------------------------------------
with insights_tab:
    st.markdown(
        '<div class="section-label">Exploratory Analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-heading">Understand Customer Behavior</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Explore the visualizations generated from the cleaned Kaggle "
        "dataset. These describe observed patterns rather than "
        "establishing causes of churn."
    )

    if df is not None and "Churn" in df.columns:
        if "Contract" in df.columns:
            contract_churn = (
                df.groupby("Contract", observed=False)["Churn"]
                .apply(lambda values: (values == "Yes").mean() * 100)
                .sort_values(ascending=False)
            )

            st.markdown("#### Churn Rate by Contract")
            st.bar_chart(
                contract_churn.rename("Churn Rate (%)")
            )

        if "tenure" in df.columns:
            tenure_df = df[["tenure", "Churn"]].copy()
            tenure_df["Tenure Group"] = pd.cut(
                tenure_df["tenure"],
                bins=[-1, 12, 24, 48, 60, 72],
                labels=[
                    "0–12 months",
                    "13–24 months",
                    "25–48 months",
                    "49–60 months",
                    "61–72 months",
                ],
            )

            tenure_churn = (
                tenure_df.groupby(
                    "Tenure Group",
                    observed=False,
                )["Churn"]
                .apply(lambda values: (values == "Yes").mean() * 100)
            )

            st.markdown("#### Churn Rate by Tenure")
            st.bar_chart(tenure_churn.rename("Churn Rate (%)"))

        if "InternetService" in df.columns:
            internet_churn = (
                df.groupby("InternetService", observed=False)["Churn"]
                .apply(lambda values: (values == "Yes").mean() * 100)
                .sort_values(ascending=False)
            )

            st.markdown("#### Churn Rate by Internet Service")
            st.bar_chart(
                internet_churn.rename("Churn Rate (%)")
            )
    else:
        st.info(
            "The cleaned dataset is required for interactive "
            "segment-level analysis."
        )

# --------------------------------------------------
# ABOUT TAB
# --------------------------------------------------
with about_tab:
    st.markdown(
        '<div class="section-heading">About ChurnGuard</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "ChurnGuard is a customer churn prediction and analytics "
        "project built with Python, pandas, scikit-learn, and Streamlit."
    )

    st.markdown("#### Project Capabilities")
    st.markdown(
        """
        - Customer data inspection and cleaning
        - Exploratory data analysis
        - Machine learning preprocessing and prediction
        - Customer churn probability estimation
        - Interactive retention analytics
        """
    )

    st.markdown("#### Dataset")
    st.write(
        "The project uses a Telco Customer Churn dataset sourced "
        "from Kaggle. Its metrics describe the supplied dataset and "
        "may not represent other customer populations."
    )

    st.markdown("#### Important Limitations")
    st.write(
        "Predictions depend on the training data and model quality. "
        "They should support, not replace, informed business decisions. "
        "No prediction guarantees that a customer will or will not churn."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "ChurnGuard | Customer Retention Intelligence | "
    "Machine learning estimates should be validated before "
    "production business use."
)
