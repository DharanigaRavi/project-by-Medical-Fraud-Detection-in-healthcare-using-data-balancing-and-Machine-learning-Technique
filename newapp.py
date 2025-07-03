import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load trained model
model = joblib.load('fraud_detection_xgb_model.pkl')

# Define categorical mappings (based on LabelEncoder from training)
CATEGORY_MAPPINGS = {
    'gender': ['Female', 'Male'],
    'procedure_type': sorted(['Routine Check', 'Blood Test', 'X-Ray', 'Vaccination',
                             'MRI Scan', 'Surgery', 'Specialist Consult']),
    'diagnosis_code': sorted(['J45', 'I10', 'E11', 'M54', 'C34', 'I21', 'E66']),
    'claim_month': ['Jan', 'Mar', 'Jun', 'Dec']
}

# Feature order expected by the model
FEATURE_ORDER = [
    'patient_age', 'gender', 'provider_id', 'hospital_id', 'claim_month',
    'procedure_type', 'diagnosis_code', 'claim_amount', 'num_procedures',
    'days_admitted', 'previous_claims', 'billing_discrepancy', 'anomaly_score',
    'month_fraud_risk', 'risk_score'
]

def encode_category(feature_name, value):
    """Encode categorical features using predefined mappings"""
    return CATEGORY_MAPPINGS[feature_name].index(value)

def calculate_risk_score(inputs):
    """Calculate risk score based on business rules"""
    risk = (
        0.3 * inputs['anomaly_score'] +
        0.25 * inputs['billing_discrepancy'] +
        0.2 * (inputs['num_procedures']/10) +
        0.15 * (inputs['claim_amount']/100000) +
        0.1 * inputs['month_fraud_risk']
    )
    return min(max(risk, 0), 1)  # Ensure between 0-1

def main():
    st.title("Medical Insurance Fraud Detection")
    st.markdown("### Assess claim legitimacy using AI-powered fraud detection")

    with st.form("claim_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Information")
            inputs = {
                'patient_age': st.slider("Age", 18, 100, 45),
                'gender': st.selectbox("Gender", CATEGORY_MAPPINGS['gender']),
                'provider_id': st.number_input("Provider ID", 1, 50, 10),
                'hospital_id': st.selectbox("Hospital ID", list(range(1, 20)) + [99]),
            }
            
        with col2:
            st.subheader("Claim Details")
            inputs.update({
                'procedure_type': st.selectbox("Procedure Type", CATEGORY_MAPPINGS['procedure_type']),
                'diagnosis_code': st.selectbox("Diagnosis Code", CATEGORY_MAPPINGS['diagnosis_code']),
                'claim_amount': st.number_input("Claim Amount ($)", 1000, 50000, 5000),
                'num_procedures': st.slider("Number of Procedures", 1, 10, 2),
                'days_admitted': st.slider("Hospital Stay (Days)", 0, 30, 1),
            })
            
        st.subheader("Additional Information")
        inputs.update({
            'previous_claims': st.slider("Previous Claims (1 year)", 0, 10, 1),
            'billing_discrepancy': st.slider("Billing Discrepancy", 0.0, 1.0, 0.1),
            'anomaly_score': st.slider("Anomaly Score", 0.0, 1.0, 0.1),
            'claim_month': st.selectbox("Claim Month", CATEGORY_MAPPINGS['claim_month']),
        })
        
        # Calculate derived features
        inputs['month_fraud_risk'] = {'Jan': 0.25, 'Mar': 0.22, 'Jun': 0.18, 'Dec': 0.35}[inputs['claim_month']]
        inputs['risk_score'] = calculate_risk_score(inputs)
        
        submitted = st.form_submit_button("Analyze Claim")
    
    if submitted:
        # Encode categorical features
        encoded_inputs = {}
        for key, value in inputs.items():
            if key in CATEGORY_MAPPINGS:
                encoded_inputs[key] = encode_category(key, value)
            else:
                encoded_inputs[key] = value
                
        # Create dataframe with correct feature order
        claim_data = pd.DataFrame([encoded_inputs])[FEATURE_ORDER]
        
        # Make prediction
        proba = model.predict_proba(claim_data)[0][1]
        prediction = model.predict(claim_data)[0]
        
        # Display results
        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Fraud Probability", f"{proba*100:.1f}%")
            st.progress(proba)
            
        with col2:
            status = "Fraud Detected ⚠️" if prediction == 1 else "Legitimate Claim ✅"
            st.markdown(f"## {status}")
            
        # Show risk factors
        st.subheader("Key Risk Indicators")
        risk_factors = []
        if inputs['hospital_id'] == 99:
            risk_factors.append("⚠️ High-risk hospital (ID 99)")
        if inputs['risk_score'] > 0.7:
            risk_factors.append(f"⚠️ Extreme risk score ({inputs['risk_score']:.2f})")
        if inputs['anomaly_score'] > 0.8:
            risk_factors.append(f"⚠️ High anomaly score ({inputs['anomaly_score']:.2f})")
            
        if risk_factors:
            for factor in risk_factors:
                st.error(factor)
        else:
            st.success("No significant risk factors detected")
            
        # Show detailed feature impacts
        with st.expander("Advanced Details"):
            st.write("Feature Values:", claim_data)
            feature_importance = pd.DataFrame({
                'Feature': FEATURE_ORDER,
                'Value': claim_data.values[0]
            })
            st.write("Encoded Feature Values:", feature_importance)

if __name__ == "__main__":
    main()