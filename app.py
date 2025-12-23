import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Telco Churn Risk Dashboard", layout="wide")
st.title("📊 Telco Churn Risk Dashboard")
st.markdown("**Random Forest (F1: 63.2%) – PRODUCTION READY**")

model = joblib.load("jay/models/best_model.pkl")

# Sidebar inputs
st.sidebar.header("👤 Customer Profile")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.slider("Monthly Charges ($)", 18.0, 118.8, 70.0)
total = st.sidebar.slider("Total Charges ($)", 0.0, 8684.8, 1000.0)

st.sidebar.markdown("---")

contract = st.sidebar.selectbox(
    "Contract type",
    ["Month-to-month", "One year", "Two year"]
)

payment = st.sidebar.selectbox(
    "Payment method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

paperless = st.sidebar.selectbox(
    "Paperless billing",
    ["Yes", "No"]
)

internet = st.sidebar.selectbox(
    "Internet service",
    ["DSL", "Fiber optic", "No"]
)

st.header("🔮 Predict Churn Risk")

if st.button("🎯 PREDICT CHURN", type="primary"):
    with st.spinner("Analyzing customer risk..."):
        # 1) Take exact 37-feature structure
        X_template = pd.read_csv("jay/data/X_test.csv").iloc[[0]]

        # 2) Override numeric features
        X_template["tenure"] = tenure
        X_template["MonthlyCharges"] = monthly
        X_template["TotalCharges"] = total
        X_template["avg_monthly_charges"] = total / max(tenure + 1, 1)

        # 3) Contract dummies
        for col in X_template.columns:
            if col.startswith("Contract_"):
                X_template[col] = 0
        if contract == "Month-to-month" and "Contract_Month-to-month" in X_template.columns:
            X_template["Contract_Month-to-month"] = 1
        elif contract == "One year" and "Contract_One year" in X_template.columns:
            X_template["Contract_One year"] = 1
        elif contract == "Two year" and "Contract_Two year" in X_template.columns:
            X_template["Contract_Two year"] = 1

        # 4) Payment method dummies
        for col in X_template.columns:
            if col.startswith("PaymentMethod_"):
                X_template[col] = 0
        mapping_pay = {
            "Electronic check": "PaymentMethod_Electronic check",
            "Mailed check": "PaymentMethod_Mailed check",
            "Bank transfer (automatic)": "PaymentMethod_Bank transfer (automatic)",
            "Credit card (automatic)": "PaymentMethod_Credit card (automatic)",
        }
        pay_col = mapping_pay.get(payment)
        if pay_col in X_template.columns:
            X_template[pay_col] = 1

        # 5) PaperlessBilling
        if "PaperlessBilling" in X_template.columns:
            X_template["PaperlessBilling"] = 1 if paperless == "Yes" else 0

        # 6) InternetService dummies
        for col in X_template.columns:
            if col.startswith("InternetService_"):
                X_template[col] = 0
        mapping_int = {
            "DSL": "InternetService_DSL",
            "Fiber optic": "InternetService_Fiber optic",
            "No": "InternetService_No",
        }
        int_col = mapping_int.get(internet)
        if int_col in X_template.columns:
            X_template[int_col] = 1

        # 7) Predict
        prob = model.predict_proba(X_template)[0, 1]
        threshold = 0.35
        pred = int(prob >= threshold)

        col1, col2 = st.columns(2)
        col1.metric("Churn Probability", f"{prob:.1%}")
        col2.metric("Risk", "🚨 HIGH" if pred == 1 else "✅ LOW")

        st.markdown("### 📈 Recommendation")
        if pred == 1:
            st.error("High churn risk – offer retention discounts / benefits.")
        else:
            st.success("Customer likely to stay – maintain current plan.")

st.markdown("---")
st.caption("Uses the same 37 encoded features as your training data (X_train/X_test).")
