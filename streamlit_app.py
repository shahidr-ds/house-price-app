import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load trained model (trained on log1p(price))
with open("xgboost_house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit app config
st.set_page_config(page_title="Pakistan House Price Predictor")
st.title("🏠 House Price Prediction - Islamabad")

# User Inputs
property_type = st.selectbox("Property Type", [
    'Flat', 'House', 'Lower Portion', 'Penthouse', 'Room', 'Upper Portion'
])

location = st.selectbox("Location", [
    'DHA Defence', 'E-11', 'F-7', 'F-8', 'G-13', 'G-15', 'Soan Garden', 'Other'
])

area = st.number_input("Total Area (sq ft)", min_value=100, max_value=20000, value=1000)
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=2)
bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)

# Optional user inputs (defaulted here)
distance_to_center = st.slider("Distance to City Center (km)", 0, 30, 5)
is_weekend = st.checkbox("Is it Weekend?", value=False)

# Predict button
if st.button("Predict Price"):

    # Feature engineering
    features = {}

    for pt in ['Flat', 'House', 'Lower Portion', 'Penthouse', 'Room', 'Upper Portion']:
        features[f'property_type_{pt}'] = 1 if property_type == pt else 0

    for loc in ['DHA Defence', 'E-11', 'F-7', 'F-8', 'G-13', 'G-15', 'Soan Garden', 'Other']:
        features[f'location_{loc}'] = 1 if location == loc else 0

    features['Total_Area_log'] = np.log1p(area)
    features['bed_bath_ratio'] = bedrooms / bathrooms if bathrooms else 0
    features['total_rooms'] = bedrooms + bathrooms
    features['price_per_room'] = area / (bedrooms + bathrooms) if (bedrooms + bathrooms) else 0
    features['area_per_bedroom'] = area / bedrooms if bedrooms else 0
    features['distance_to_center'] = distance_to_center
    features['is_weekend'] = 1 if is_weekend else 0
    features['season_winter'] = 0  # default
    features['location_cluster'] = 0  # default

    # Ensure column order matches the model
    ordered_features = [
        'property_type_Flat', 'property_type_House', 'property_type_Lower Portion',
        'property_type_Penthouse', 'property_type_Room', 'property_type_Upper Portion',
        'location_DHA Defence', 'location_E-11', 'location_F-7', 'location_F-8',
        'location_G-13', 'location_G-15', 'location_Other', 'location_Soan Garden',
        'is_weekend', 'Total_Area_log', 'distance_to_center', 'bed_bath_ratio',
        'total_rooms', 'season_winter', 'area_per_bedroom', 'price_per_room',
        'location_cluster'
    ]

    input_df = pd.DataFrame([[features[feat] for feat in ordered_features]], columns=ordered_features)

    # Predict and convert back from log1p
    log_price = model.predict(input_df)[0]
    predicted_price = np.expm1(log_price)

    st.success(f"🏷️ Estimated House Price: PKR {predicted_price:,.0f}")
