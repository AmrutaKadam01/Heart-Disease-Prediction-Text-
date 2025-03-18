import streamlit as st
import numpy as np
import joblib

# Load trained models and scaler
rf_model = joblib.load("random_forest_model.pkl")
lr_model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

# Function to make predictions using both models
def predict_disease(features):
    features_array = np.array([features])
    features_scaled = scaler.transform(features_array)  # Apply scaling

    # Predictions from both models
    rf_prediction = rf_model.predict(features_scaled)[0]
    lr_prediction = lr_model.predict(features_scaled)[0]

    # Convert predictions into readable format
    rf_result = "Heart Disease Present" if rf_prediction == 1 else "No Heart Disease"
    lr_result = "Heart Disease Present" if lr_prediction == 1 else "No Heart Disease"

    return rf_result, lr_result

# Streamlit App UI
st.title("Heart Disease Prediction")
st.write("Enter the details below to predict the risk of heart disease.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=50)
sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type (CP)", options=[0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=600, value=200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dL", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2], format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)
exang = st.radio("Exercise-Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x])

# Collect input features
input_features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Ensure input types are correct
input_features = list(map(float, input_features))  # Use float for proper scaling

# Predict button
if st.button("Predict"):
    rf_result, lr_result = predict_disease(input_features)
    
    st.subheader("Prediction Results:")
    st.success(f"🌲 Random Forest: {rf_result}")
    st.info(f"📊 Logistic Regression: {lr_result}")
