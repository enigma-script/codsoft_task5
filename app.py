import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# Load data
data = pd.read_csv("creditcard.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_res, y_res)

# UI
st.title("💳 Credit Card Fraud Detection App")

st.write("Enter transaction values (simulated input)")

# Create dummy inputs (since features are V1..V28)
user_input = []
for i in range(X.shape[1]):
    value = st.number_input(f"Feature V{i+1}", value=0.0)
    user_input.append(value)

if st.button("Predict"):
    input_scaled = scaler.transform([user_input])
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")
