import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler
with open("models/xgboost_house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Get expected features
feature_names = scaler.feature_names_in_

# Streamlit UI
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

    # Create full input row with zeros
    input_data = dict.fromkeys(feature_names, 0)
    input_data['area'] = area
    input_data['bedrooms'] = bedrooms
    input_data['bathrooms'] = bathrooms

    # Set one-hot encoded fields
    location_col = f"location_{location}"
    property_col = f"property_type_{property_type}"

    if location_col in input_data:
        input_data[location_col] = 1
    if property_col in input_data:
        input_data[property_col] = 1

    # Create input DataFrame
    input_df = pd.DataFrame([input_data])[feature_names]

    # Scale input and convert to float32 numpy array
    input_scaled = scaler.transform(input_df)
    input_scaled = np.array(input_scaled, dtype=np.float32)

    # Predict
    log_price = model.predict(input_scaled, validate_features=False)[0]
    predicted_price = np.exp(log_price)

    # Show result
    st.subheader("🏷️ Predicted House Price")
    st.success(f"Estimated Price: **PKR {predicted_price:,.0f}**")
