import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from inference.freight_predict import pred_freight_cost
from inference.flagging_invoice import prediction_flagging

st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",

    page_icon="📦",
    layout="wide"
)

st.markdown("""
# Vendor Invoice Intelligence Portal

### AI-Driven Freight Cost Prediction & Invoice Risk Flagging

---

This analytics portal uses machine learning to improve **invoice processing, cost forecasting, and financial control**.

#### Key Capabilities

- **Freight Cost Prediction**  
  Estimate expected freight costs based on invoice quantity and dollar value.

- **Invoice Risk Flagging**  
  Identify potentially abnormal invoices that require manual review.

- **Financial Control**  
  Reduce cost leakage, improve approval accuracy, and minimize manual workload.

---
""")

st.divider()

st.sidebar.title(" Model Selection")
selected_model = st.sidebar.radio(
"Choose Prediction Module",[

"Freight Cost Prediction", "Invoice Manual Approval Flag"])

st. sidebar.markdown("""
---
**Business Impact**
- Improved cost forecasting\n
-Reduced invoice fraud & anomalies\n
-Faster finance operations""")

if selected_model == "Freight Cost Prediction":
    st.subheader(" Freight Cost Prediction")
    st.markdown("""
    **0bjective:**
    Predict freight cost for a vendor invoice using **Quantity** and **Invoice Dollars** 
    to support budgeting, forecasting, and vendor negotiations.""")

    with st.form("freight_form"):
        col1, col2 = st.columns(2)
        with col1:
            quantity = st.number_input(
            "Quantity",

            min_value=1.0,

            value=1200.0)

        with col2:
            dollars= st.number_input(
                "Invoice Dollars",

                min_value=1.0,

                value=18500.0)
        submit_freight = st.form_submit_button(" Predict Freight Cost")
    if submit_freight:
        input_data = {


            "Dollars": [dollars],
            "Quantity": [quantity],}
        prediction=pred_freight_cost(input_data)['freight_predicted']
        st.success("Prediction completed successfully.")


        st.metric(label="l Estimated Freight Cost", value=f"RS:{prediction[0]:,.2f}")

else:
    st.subheader("Invoice Manual Approval Prediction")
    st.markdown("""
    **Objective: **
    Predict whether a vendor invoice should be **flagged for manual approval**
     based on abnormal cost, freight, or delivery patterns.""")

    with st.form("invoice flag form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            invoice_quantity = st.number_input(

                "Invoice Quantity",

                min_value=1.0,

                value=50.0)
            freight = st.number_input(
                "Freight Cost",
                min_value=0.0,

                value=1.73
            )
        with col2:
            invoice_dollars= st.number_input(

                "Invoice dollars",

                min_value=1.0,

                value=352.95)
            total_item_quantity = st.number_input(
                "Total item quanitity",
                min_value=0.0,

                value=162.0
            )

        with col3:
           total_item_dollars = st.number_input(

                "total item dollars",

                min_value=1.0,

                value=2476.0)
        submit_flag=st.form_submit_button('evaluate invoice risk!')

    if submit_flag:
        input_data={
            "invoice_quantity": [invoice_quantity],

            "invoice_dollars":[invoice_dollars],
            "Freight": [freight],

            "total_item_quantity":[total_item_quantity],
            "total_item_dollars":[total_item_dollars]
        }

        flag_prediction = prediction_flagging(input_data)['flagging_pred']

        is_flagged = bool(flag_prediction[0])

        if is_flagged:

            st.error(" Invoice requires **MANUAL APPROVAL**")
        else:
            st.success(" Invoice is **SAFE for Auto-Approval*")