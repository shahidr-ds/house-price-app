import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load trained model and target scaler
model = pickle.load(open("models/xgboost_house_price_model.pkl", "rb"))
target_scaler = pickle.load(open("models/target_scaler.pkl", "rb"))

# Page settings
st.set_page_config(page_title="Pakistan House Price Predictor")
st.title("🏠 House Price Prediction - Islamabad")

# Input fields
property_type = st.selectbox("Property Type", ['Flat', 'House', 'Lower Portion', 'Room', 'Upper Portion'])
location = st.selectbox("Location", ['DHA Defence', 'E-11', 'F-11', 'F-7', 'F-8', 'G-13', 'Other'])
area = st.number_input("Total Area (sq ft)", min_value=100, max_value=20000, value=1000)
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=2)
bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)

# Predict on button click
if st.button("Predict Price"):
    # Build feature dictionary
    features = {}

    # One-hot encode property type
    for pt in ['Flat', 'House', 'Lower Portion', 'Room', 'Upper Portion']:
        features[f'property_type_{pt}'] = 1 if property_type == pt else 0

    # One-hot encode location
    for loc in ['DHA Defence', 'E-11', 'F-11', 'F-7', 'F-8', 'G-13', 'Other']:
        features[f'location_{loc}'] = 1 if location == loc else 0

    # Numerical features
    features['Total_Area_log'] = np.log1p(area)
    features['bed_bath_ratio'] = bedrooms / bathrooms if bathrooms > 0 else 0
    features['total_rooms'] = bedrooms + bathrooms
    features['price_per_room'] = area / (bedrooms + bathrooms) if (bedrooms + bathrooms) > 0 else 0

    # Dummy columns expected by model
    features['season_summer'] = 0
    features['season_winter'] = 0
    features['location_cluster'] = 0

    # Define column order as used during model training
    ordered_features = [
        'property_type_Flat', 'property_type_House', 'property_type_Lower Portion',
        'property_type_Room', 'property_type_Upper Portion', 'location_DHA Defence',
        'location_E-11', 'location_F-11', 'location_F-7', 'location_F-8',
        'location_G-13', 'location_Other', 'Total_Area_log', 'bed_bath_ratio',
        'total_rooms', 'season_summer', 'season_winter', 'price_per_room',
        'location_cluster'
    ]

    # Convert to DataFrame in correct order
    input_df = pd.DataFrame([[features[feat] for feat in ordered_features]], columns=ordered_features)

    # Predict in scaled log space
    scaled_log_price = model.predict(input_df)[0]

    # Unscale and reverse log1p to get actual price
    log_price = target_scaler.inverse_transform([[scaled_log_price]])[0][0]
    predicted_price = np.expm1(log_price)

    # Show result
    st.success(f"🏷️ Estimated House Price: PKR {predicted_price:,.0f}")
