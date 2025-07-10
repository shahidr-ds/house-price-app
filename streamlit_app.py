import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load model and preprocessors
model = joblib.load("xgboost_house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
ohe = joblib.load("ohe.pkl")
final_features = joblib.load("final_features.pkl")

# Location → center distance
location_distance_dict = {
    'F-7': 0, 'F-8': 1, 'G-13': 12, 'G-15': 15,
    'E-11': 6, 'DHA Defence': 20, 'Soan Garden': 25, 'Bahria Town': 22
}

# --- UI ---
st.set_page_config(page_title="House Price App", layout="centered")
st.title("🏠 Pakistan House Price Predictor")

with st.form("predict_form"):
    area = st.number_input("Total Area (Marla)", min_value=1.0, value=5.0)
    bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)

    property_type = st.selectbox("Property Type", [
        "Flat", "House", "Lower Portion", "Upper Portion", "Penthouse", "Room"
    ])
    location = st.selectbox("Location", [
        "F-7", "F-8", "G-13", "G-15", "E-11", "DHA Defence", "Soan Garden", "Bahria Town"
    ])
    submit = st.form_submit_button("🔍 Predict Price")

# --- Predict ---
if submit:
    # Derived features
    raw_area = area
    total_area_log = np.log(area + 1)
    bed_bath_ratio = bedrooms / bathrooms if bathrooms else 0
    total_rooms = bedrooms + bathrooms
    season_winter = 1 if datetime.today().month in [12, 1, 2] else 0
    area_per_bedroom = area / bedrooms if bedrooms else 0
    price_per_room = area / total_rooms if total_rooms else 0
    is_weekend = 1 if datetime.today().weekday() >= 5 else 0
    location_cluster = 0
    distance_to_center = location_distance_dict.get(location, 15)

    # Categorical: property_type, location
    cat_df = pd.DataFrame([[property_type, location]], columns=["property_type", "location"])
    cat_encoded = ohe.transform(cat_df)
    cat_df_ohe = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out())

    # Numerical
    num_df = pd.DataFrame([{
        "raw_area": raw_area,
        "total_area_log": total_area_log,
        "bed_bath_ratio": bed_bath_ratio,
        "total_rooms": total_rooms,
        "season_winter": season_winter,
        "area_per_bedroom": area_per_bedroom,
        "price_per_room": price_per_room,
        "is_weekend": is_weekend,
        "distance_to_center": distance_to_center,
        "location_cluster": location_cluster
    }])

    # Combine
    input_df = pd.concat([cat_df_ohe, num_df], axis=1)

    # Ensure all expected columns are present
    for col in final_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[final_features]
    st.markdown("### 🔍 Final Features Sent to Model:")
    st.write(input_df)


    # Scale & Predict
    X_scaled = scaler.transform(input_df)
    log_price = model.predict(X_scaled)[0]
    predicted_price = np.exp(log_price)

    # Output
    st.subheader("💰 Estimated House Price:")
    st.success(f"PKR {predicted_price:,.0f}")
