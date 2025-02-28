import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.classification import load_model, predict_model
from pycaret.regression import load_model as load_regressor

st.title("Interactive PyCaret Model Deployment ")
# Upload a PyCaret model file
uploaded_model = st.file_uploader("Upload your PyCaret model (.pkl)", type=["pkl"])

if uploaded_model:
    # Save and load the uploaded model
    with open("temp_model.pkl", "wb") as f:
        f.write(uploaded_model.getbuffer())

    model = load_model("temp_model")
    st.success("Model uploaded and loaded successfully!")

    # Dynamically ask for feature inputs
    st.subheader("Enter Feature Values")

    # Extract feature names from the PyCaret model
    feature_names = list(model.feature_names_in_)  # Get expected features
    user_inputs = {}

    for feature in feature_names:
        user_inputs[feature] = st.text_input(f"Enter {feature}", "")

    # Convert input to DataFrame
    if st.button("Predict"):
        # Convert input values to correct data types
        input_df = pd.DataFrame([user_inputs])
        
        # Predict using PyCaret
        predictions = predict_model(model, data=input_df)
        st.success(f"Prediction: {predictions['prediction_label'][0]}")
