import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="💳",
    layout="wide"
)

model = joblib.load("credit_risk_model.pkl")
feature_names = model.feature_names_in_

st.title("💳 Microcredit Credit Risk Dashboard")
st.markdown("Machine Learning Powered **Loan Default Prediction System**")

# -----------------------
# SIDEBAR INPUT
# -----------------------

st.sidebar.header("Loan Application")

loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)

tenor = st.sidebar.number_input("Tenor (months)", min_value=1)

sector = st.sidebar.selectbox(
    "Sector",
    [
        "Boda Boda",
        "Consumer",
        "Corporate",
        "Express Motor",
        "Micro",
        "Micro Chap chap",
        "Mobile Money",
        "SME",
        "TEST"
    ]
)

payment_frequency = st.sidebar.selectbox(
    "Payment Frequency",
    ["Weekly","Monthly"]
)

run_model = st.sidebar.button("Run Risk Assessment")

# -----------------------
# MODEL PREDICTION
# -----------------------

if run_model:

    input_data = pd.DataFrame(columns=feature_names)
    input_data.loc[0] = 0

    if "disbursed_amount" in input_data.columns:
        input_data["disbursed_amount"] = loan_amount

    if "tenor" in input_data.columns:
        input_data["tenor"] = tenor

    sector_col = f"sector_{sector}"
    if sector_col in input_data.columns:
        input_data[sector_col] = 1

    if payment_frequency == "Weekly":
        if "payment_frequency_Weekly" in input_data.columns:
            input_data["payment_frequency_Weekly"] = 1

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    risk_percent = int(probability * 100)

    credit_score = int((1 - probability) * 850)

    # -----------------------
    # METRIC CARDS
    # -----------------------

    col1, col2, col3 = st.columns(3)

    col1.metric("Loan Amount", f"{loan_amount:,.0f}")
    col2.metric("Tenor", f"{tenor} months")
    col3.metric("Credit Score", credit_score)

    st.divider()

    # -----------------------
    # RISK GAUGE
    # -----------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Default Risk Gauge")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_percent,
            title={'text': "Default Probability %"},
            gauge={
                'axis': {'range': [0,100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0,30], 'color': "green"},
                    {'range': [30,60], 'color': "yellow"},
                    {'range': [60,100], 'color': "red"}
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    # -----------------------
    # PROBABILITY BAR CHART
    # -----------------------

    with col2:

        st.subheader("Default Probability Breakdown")

        prob_data = pd.DataFrame({
            "Outcome":["No Default","Default"],
            "Probability":[1-probability, probability]
        })

        st.bar_chart(prob_data.set_index("Outcome"))

    # -----------------------
    # INTEREST CALCULATION
    # -----------------------

    sector_rates = {
        "Boda Boda":0.36,
        "Consumer":0.40,
        "Corporate":0.25,
        "Express Motor":0.30,
        "Micro":0.42,
        "Micro Chap chap":0.45,
        "Mobile Money":0.35,
        "SME":0.32,
        "TEST":0.30
    }

    rate = sector_rates.get(sector,0.35)

    estimated_interest = loan_amount * rate * (tenor/12)

    st.divider()

    col1, col2 = st.columns(2)

    col1.metric("Sector APR", f"{rate*100:.1f}%")
    col2.metric("Estimated Interest", f"{estimated_interest:,.0f}")

    # -----------------------
    # LOAN DECISION
    # -----------------------

    st.subheader("Loan Decision")

    if risk_percent < 30:
        st.success("Loan Approved – Low Risk Borrower")

    elif risk_percent < 60:
        st.warning("Manual Review Required")

    else:
        st.error("Loan Rejected – High Risk")