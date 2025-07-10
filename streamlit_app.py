import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("models/xgboost_house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the scaler
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define all possible feature columns (same as used in training)
expected_columns = [
    'area', 'bedrooms', 'bathrooms', 'location_DHA Defence', 'location_E-11',
    'location_F-7', 'location_F-8', 'location_G-13', 'location_G-15',
    'property_type_Flat', 'property_type_House', 'property_type_Lower Portion',
    'property_type_Penthouse', 'property_type_Room', 'property_type_Upper Portion'
]

st.title("🏠 House Price Prediction App")

# Sidebar inputs
st.sidebar.header("Enter property details:")

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

    # Initialize a zero row
    input_data = dict.fromkeys(expected_columns, 0)
    input_data['area'] = area
    input_data['bedrooms'] = bedrooms
    input_data['bathrooms'] = bathrooms

    # Set the one-hot encoded values
    location_col = f"location_{location}"
    prop_col = f"property_type_{property_type}"

    if location_col in input_data:
        input_data[location_col] = 1

    if prop_col in input_data:
        input_data[prop_col] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Scale the data
    input_scaled = scaler.transform(input_df)

    # Predict log price
    log_price = model.predict(input_scaled, validate_features=False)[0]

    # Convert log price to actual price
    predicted_price = np.exp(log_price)

    st.subheader("🏷️ Predicted House Price")
    st.success(f"Estimated Price: **PKR {predicted_price:,.0f}**")
