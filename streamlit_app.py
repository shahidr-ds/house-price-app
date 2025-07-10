import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load model and scaler ---
with open("models/xgboost_house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --- Get feature names used by scaler and model ---
scaler_features = scaler.feature_names_in_
model_features = model.get_booster().feature_names

# --- Streamlit UI ---
st.title("🏠 House Price Prediction App")
st.sidebar.header("Enter Property Details")

# --- User input ---
area = st.sidebar.number_input("Area (in marla)", min_value=1.0, max_value=100.0, value=5.0)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2)

location = st.sidebar.selectbox("Location", [
    "DHA Defence", "E-11", "F-7", "F-8", "G-13", "G-15"
])

property_type = st.sidebar.selectbox("Property Type", [
    "Flat", "House", "Lower Portion", "Penthouse", "Room", "Upper Portion"
])

if st.sidebar.button("Predict Price"):

    # --- Step 1: Initialize input_data with scaler features ---
    input_data = dict.fromkeys(scaler_features, 0)

    # --- Step 2: Set numeric fields ---
    if "area" in input_data:
        input_data["area"] = area
    if "bedrooms" in input_data:
        input_data["bedrooms"] = bedrooms
    if "bathrooms" in input_data:
        input_data["bathrooms"] = bathrooms

    # --- Step 3: Set one-hot encoded fields ---
    location_col = f"location_{location}"
    prop_col = f"property_type_{property_type}"

    if location_col in input_data:
        input_data[location_col] = 1
    if prop_col in input_data:
        input_data[prop_col] = 1

    # --- Step 4: Create DataFrame for scaler ---
    input_df = pd.DataFrame([input_data])[scaler_features]  # exact match

    # --- Step 5: Scale the input ---
    input_scaled = scaler.transform(input_df)

    # --- Step 6: Create DataFrame for model input ---
    input_scaled_df = pd.DataFrame(input_scaled, columns=scaler_f
