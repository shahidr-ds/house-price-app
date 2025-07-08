import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load the trained XGBoost model (trained on log1p(price))
with open("xgboost_house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# App title and config
st.set_page_config(page_title="Pakistan House Price Predictor")
st.title("🏠 House Price Prediction - Islamabad")

# User inputs
property_type = st.selectbox("Property Type", ['Flat', 'House', 'Lower Portion', 'Room', 'Upper Portion'])
location = st.selectbox("Location", ['DHA Defence', 'E-11', 'F-11', 'F-7', 'F-8', 'G-13', 'Other'])
area = st.number_input("Total Area (sq ft)", min_value=100, max_value=20000, value=1000)
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=2)
bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)
# Removed season input

# Predict button
if st.button("Predict Price"):

    # Feature engineering
    features = {}

    # One-hot encoding: property_type
    for pt in ['Flat', 'House', 'Lower Portion', 'Room', 'Upper Portion']:
        features[f'property_type_{pt}'] = 1 if property_type == pt else 0

    # One-hot encoding: location
    for loc in ['DHA Defence', 'E-11', 'F-11', 'F-7', 'F-8', 'G-13', 'Other']:
        features[f'location_{loc}'] = 1 if location == loc else 0

    # Numeric features
    features['Total_Area_log'] = np.log1p(area)
    features['bed_bath_ratio'] = bedrooms / bathrooms if bathrooms else 0
    features['total_rooms'] = bedrooms + bathrooms
    features['price_per_room'] = area / (bedrooms + bathrooms) if (bedrooms + bathrooms) else 0

    # Remove season; set default
    features['season_summer'] = 0
    features['season_winter'] = 0

    # Placeholder for cluster
    features['location_cluster'] = 0

    # Final feature order
    ordered_features = [
        'property_type_Flat', 'property_type_House', 'property_type_Lower Portion',
        'property_type_Room', 'property_type_Upper Portion', 'location_DHA Defence',
        'location_E-11', 'location_F-11', 'location_F-7', 'location_F-8',
        'location_G-13', 'location_Other', 'Total_Area_log', 'bed_bath_ratio',
        'total_rooms', 'season_summer', 'season_winter', 'price_per_room',
        'location_cluster'
    ]

    # Create input DataFrame
    input_df = pd.DataFrame([[features[feat] for feat in ordered_features]], columns=ordered_features)

    # Predict log(price)
    log_price = model.predict(input_df)[0]

    # Inverse log1p to get actual price
    predicted_price = np.expm1(log_price)

    # Show result
    st.success(f"🏷️ Estimated House Price: PKR {predicted_price:,.0f}")
