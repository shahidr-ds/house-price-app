import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load model and scaler ---
with open("models/xgboost_house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --- Get feature names used during model training ---
trained_feature_names = model.get_booster().feature_names

# --- Streamlit UI ---
st.title("🏠 House Price Prediction App")
st.sidebar.header("Enter Property Details")

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

    # --- Create base input with all zero columns ---
    input_data = dict.fromkeys(trained_feature_names, 0)

    # --- Set numeric features ---
    input_data["area"] = area
    input_data["bedrooms"] = bedrooms
    input_data["bathrooms"] = bathrooms

    # --- Set one-hot encoded fields ---
    location_col = f"location_{location}"
    prop_col = f"property_type_{property_type}"

    if location_col in input_data:
        input_data[location_col] = 1
    if prop_col in input_data:
        input_data[prop_col] = 1

    # --- Convert to DataFrame ---
    input_df = pd.DataFrame([input_data])

    # --- Reorder columns to match training ---
    input_df = input_df[trained_feature_names]

    # --- Scale input data ---
    input_scaled = scaler.transform(input_df)

    # --- Convert back to DataFrame with correct column names ---
    input_scaled_df = pd.DataFrame(input_scaled, columns=trained_feature_names)

    # --- Predict log price ---
    log_price = model.predict(input_scaled_df, validate_features=True)[0]

    # --- Convert from log(price) to actual price ---
    predicted_price = np.exp(log_price)

    # --- Display result ---
    st.subheader("🏷️ Predicted House Price")
    st.success(f"Estimated Price: **PKR {predicted_price:,.0f}**")
