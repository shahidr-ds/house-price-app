import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load trained XGBoost model (trained on raw price)
with open("xgboost_house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# App settings
st.set_page_config(page_title="Pakistan House Price Predictor")
st.title("🏠 House Price Prediction - Islamabad")

# User inputs
property_type = st.selectbox("Property Type", [
    'Flat', 'House', 'Lower Portion', 'Penthouse', 'Room', 'Upper Portion'
])
location = st.selectbox("Location", [
    'DHA Defence', 'E-11', 'F-7', 'F-8', 'G-13', 'G-15', 'Other', 'Soan Garden'
])
area = st.number_input("Total Area (sq ft)", min_value=100, max_value=20000, value=1000)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

# Predict button
if st.button("Predict Price"):

    # Feature engineering
    features = {}

    # One-hot: Property Type
    for pt in ['Flat', 'House', 'Lower Portion', 'Penthouse', 'Room', 'Upper Portion']:
        features[f'property_type_{pt}'] = 1 if property_type == pt else 0

    # One-hot: Location
    for loc in ['DHA Defence', 'E-11', 'F-7', 'F-8', 'G-13', 'G-15', 'Other', 'Soan Garden']:
        features[f'location_{loc}'] = 1 if location == loc else 0

    # Extra numerical features
    total_rooms = bedrooms + bathrooms
    features['Total_Area_log'] = np.log1p(area)
    features['distance_to_center'] = 5  # Default
    features['bed_bath_ratio'] = bedrooms / bathrooms
    features['total_rooms'] = total_rooms
    features['season_winter'] = 0  # Default
    features['is_weekend'] = 0  # Default
    features['area_per_bedroom'] = area / bedrooms
    features['price_per_room'] = area / total_rooms
    features['location_cluster'] = 0  # Placeholder

    # Final ordered features
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

    # Predict actual price (no log)
    predicted_price = model.predict(input_df)[0]

    # Show result
    st.subheader("🔎 Model Input Preview")
    st.dataframe(input_df)

    st.success(f"🏷️ Estimated House Price: PKR {predicted_price:,.0f}")
