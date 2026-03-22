import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")

@st.cache_resource
def load_artifacts():
    for f in ("churn_model.pkl", "scaler.pkl"):
        if not os.path.exists(f):
            st.error(f"❗ '{f}' not found. Please run the notebook first.")
            st.stop()

    model_data  = joblib.load("churn_model.pkl")
    scaler_data = joblib.load("scaler.pkl")

    # Support both old format (bare object) and new format (dict with feature_cols)
    if isinstance(model_data, dict):
        model        = model_data["model"]
        feature_cols = model_data["feature_cols"]
    else:
        model        = model_data
        feature_cols = None

    if isinstance(scaler_data, dict):
        scaler = scaler_data["scaler"]
        if feature_cols is None:
            feature_cols = scaler_data["feature_cols"]
    else:
        scaler = scaler_data

    # Last resort fallback if neither pkl had feature_cols
    if feature_cols is None:
        feature_cols = [
            'SeniorCitizen','Partner','Dependents','tenure','PhoneService',
            'PaperlessBilling','MonthlyCharges','TotalCharges','gender_Male',
            'MultipleLines_No phone service','MultipleLines_Yes',
            'InternetService_Fiber optic','InternetService_No',
            'OnlineSecurity_No internet service','OnlineSecurity_Yes',
            'OnlineBackup_No internet service','OnlineBackup_Yes',
            'DeviceProtection_No internet service','DeviceProtection_Yes',
            'TechSupport_No internet service','TechSupport_Yes',
            'StreamingTV_No internet service','StreamingTV_Yes',
            'StreamingMovies_No internet service','StreamingMovies_Yes',
            'Contract_One year','Contract_Two year',
            'PaymentMethod_Credit card (automatic)',
            'PaymentMethod_Electronic check','PaymentMethod_Mailed check',
        ]

    return model, scaler, feature_cols

model, scaler, FEATURE_COLS = load_artifacts()

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("📊 Customer Churn Prediction App")
st.write("Fill in the customer details below and click **Predict**.")
st.divider()

with st.form("churn_form"):
    st.subheader("👤 Customer Information")
    col1, col2 = st.columns(2)

    with col1:
        gender         = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner        = st.selectbox("Partner", ["No", "Yes"])
        dependents     = st.selectbox("Dependents", ["No", "Yes"])
        tenure         = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        phone_service  = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    with col2:
        internet_service  = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security   = st.selectbox("Online Security",   ["No", "Yes", "No internet service"])
        online_backup     = st.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support      = st.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
        streaming_tv      = st.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
        streaming_movies  = st.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])

    st.subheader("💳 Billing Information")
    col3, col4 = st.columns(2)

    with col3:
        contract          = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        monthly_charges   = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=65.0)

    with col4:
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=780.0)

    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

# ── Predict ────────────────────────────────────────────────────────────────────
if submitted:
    raw = pd.DataFrame([{
        "SeniorCitizen":    1 if senior_citizen    == "Yes" else 0,
        "Partner":          1 if partner           == "Yes" else 0,
        "Dependents":       1 if dependents        == "Yes" else 0,
        "tenure":           int(tenure),
        "PhoneService":     1 if phone_service     == "Yes" else 0,
        "PaperlessBilling": 1 if paperless_billing == "Yes" else 0,
        "MonthlyCharges":   float(monthly_charges),
        "TotalCharges":     float(total_charges),
        "gender":           gender,
        "MultipleLines":    multiple_lines,
        "InternetService":  internet_service,
        "OnlineSecurity":   online_security,
        "OnlineBackup":     online_backup,
        "DeviceProtection": device_protection,
        "TechSupport":      tech_support,
        "StreamingTV":      streaming_tv,
        "StreamingMovies":  streaming_movies,
        "Contract":         contract,
        "PaymentMethod":    payment_method,
    }])

    encoded = pd.get_dummies(raw, drop_first=True)
    aligned = encoded.reindex(columns=FEATURE_COLS, fill_value=0)
    scaled  = scaler.transform(aligned)

    prediction  = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0]

    st.divider()
    if prediction == 1:
        st.error("❌ **This customer is likely to Churn**")
    else:
        st.success("✅ **This customer is likely to Stay**")

    c1, c2 = st.columns(2)
    c1.metric("Churn Probability",     f"{probability[1]*100:.1f}%")
    c2.metric("Retention Probability", f"{probability[0]*100:.1f}%")
    st.progress(float(probability[1]), text=f"Churn Risk: {probability[1]*100:.1f}%")
