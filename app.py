import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import numpy as np

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="UnifiedShield – UPI Fraud Detection",
    page_icon="🛡",
    layout="centered"
)

# -----------------------------
# Load model
# -----------------------------
with open("unified_shield_project.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Load dataset ONLY for categories
# -----------------------------
data = pd.read_csv("Copy of Sample_DATA.csv")

# -----------------------------
# UI
# -----------------------------
st.title("🛡 UnifiedShield")
st.caption("ML-Based UPI Fraud Detection System")

st.divider()

# -----------------------------
# Inputs
# -----------------------------
transaction_date = st.date_input("📅 Transaction Date", datetime.today())
amount = st.number_input("💰 Transaction Amount", min_value=1.0, step=1.0)

transaction_type = st.selectbox(
    "🔄 Transaction Type",
    sorted(data["Transaction_Type"].dropna().unique())
)

payment_gateway = st.selectbox(
    "🏦 Payment Gateway",
    sorted(data["Payment_Gateway"].dropna().unique())
)

transaction_state = st.selectbox(
    "📍 Transaction State",
    sorted(data["Transaction_State"].dropna().unique())
)

merchant_category = st.selectbox(
    "🏪 Merchant Category",
    sorted(data["Merchant_Category"].dropna().unique())
)

# -----------------------------
# Feature engineering
# -----------------------------
year = transaction_date.year
month = transaction_date.month

input_df = pd.DataFrame({
    "amount": [amount],
    "Year": [year],
    "Month": [month],
    "Transaction_Type": [transaction_type],
    "Payment_Gateway": [payment_gateway],
    "Transaction_State": [transaction_state],
    "Merchant_Category": [merchant_category]
})

# -----------------------------
# ONE-HOT ENCODE (MATCH TRAINING)
# -----------------------------
input_encoded = pd.get_dummies(
    input_df,
    columns=[
        "Transaction_Type",
        "Payment_Gateway",
        "Transaction_State",
        "Merchant_Category"
    ],
    drop_first=True
)

# Align columns with model
model_features = model.get_booster().feature_names
input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

# -----------------------------
# Prediction
# -----------------------------
st.divider()

if st.button("🔍 Predict Fraud Risk"):
    proba = model.predict_proba(input_encoded)[0][1]
    
    # Convert to percentage
    risk_percentage = proba * 100
    
    # Calibrated threshold (adjusted for percentage)
    if risk_percentage >= 75:
        st.error(f"⚠️ High Fraud Risk\n\nRisk Score: **{risk_percentage:.2f}%**")
    elif risk_percentage >= 45:
        st.warning(f"⚠️ Medium Risk Transaction\n\nRisk Score: **{risk_percentage:.2f}%**")
    else:
        st.success(f"✅ Low Risk Transaction\n\nRisk Score: **{risk_percentage:.2f}%**")

st.caption("© UnifiedShield | Final-Year ML Project")