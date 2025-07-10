import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os

# 🧠 Load model and scaler
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/xgboost_house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# 🛠 App settings
st.set_page_config(page_title="Pakistan House Price Predictor")
st.title("🏠 House Price Prediction - Islamabad")

# 📥 User Inputs
property_type = st.selectbox("Property Type", [
    'Flat', 'House', 'Lower Portion', 'Penthouse', 'Room', 'Upper Portion'
])
location = st.selectbox("Location", [
    'DHA Defence', 'E-11', 'F-7', 'F-8', 'G-13', 'G-15', 'Other', 'Soan Garden'
])
area = st.number_input("Total Area (sq ft)", min_value=100, max_value=20000, value=1000)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

# 🚀 Predict button
if st.button("Predict Price"):

    # ⚙️ Feature Engineering
    features = {}

    # One-hot: Property Type
    for pt in ['Flat', 'House', 'Lower Portion', 'Penthouse', 'Room', 'Upper Portion']:
        features[f'property_type_{pt}'] = 1 if property_type == pt else 0

    # One-hot: Location
    for loc in ['DHA Defence', 'E-11', 'F-7', 'F-8', 'G-13', 'G-15', 'Other', 'Soan Garden']:
        features[f'location_{loc}'] = 1 if location == loc else 0

    # Numeric Features
    total_rooms = bedrooms + bathrooms
    features['Total_Area_log'] = np.log1p(area)
    features['distance_to_center'] = 5
    features['bed_bath_ratio'] = bedrooms / bathrooms
    features['total_rooms'] = total_rooms
    features['season_winter'] = 0
    features['is_weekend'] = 0
    features['area_per_bedroom'] = area / bedrooms
    features['price_per_room'] = area / total_rooms
    features['location_cluster'] = 0

    # Create input DataFrame
    input_df = pd.DataFrame([features])

    # 🛡 Ensure all expected columns are present
    try:
        expected_columns = scaler.feature_names_in_
    except AttributeError:
        st.error("Scaler missing `feature_names_in_`. Retrain with DataFrame instead of NumPy.")
        st.stop()

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    # 🔄 Scale the input
    input_scaled_array = scaler.transform(input_df)
    input_scaled = pd.DataFrame(input_scaled_array, columns=input_df.columns)

    # 🔮 Predict log-price, then convert back
    log_price = model.predict(input_scaled, validate_features=False)[0]
    predicted_price = np.expm1(log_price)

    # 💡 Display results
    st.subheader("🔍 Model Input Preview")
    st.dataframe(input_df)

    st.success(f"🏷️ Estimated House Price: PKR {predicted_price:,.0f}")
