import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessing files
model = joblib.load("xgboost_house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
ohe = joblib.load("ohe.pkl")
final_features = joblib.load("final_features.pkl")

# Location distance dictionary
location_distance_dict = {
    'F-7': 0, 'F-8': 1, 'G-13': 12, 'G-15': 15,
    'E-11': 6, 'DHA Defence': 20, 'Soan Garden': 25, 'Bahria Town': 22
}

# Streamlit UI
st.set_page_config(page_title="House Price App", layout="centered")
st.title("🏠 Pakistan House Price Estimator")

with st.form("predict_form"):
    area = st.number_input("Total Area (Marla)", min_value=1.0, value=5.0)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

    property_type = st.selectbox("Property Type", [
        "Flat", "House", "Lower Portion", "Upper Portion", "Penthouse", "Room"
    ])
    location = st.selectbox("Location", [
        "F-7", "F-8", "G-13", "G-15", "E-11", "DHA Defence", "Soan Garden", "Bahria Town"
    ])

    submit = st.form_submit_button("🔍 Predict Price")

if submit:
    # Basic numeric features
    raw_area = area
    distance_to_center = location_distance_dict.get(location, 15)

    # Categorical encoding
    cat_df = pd.DataFrame([[property_type, location]], columns=["property_type", "location"])
    cat_encoded = ohe.transform(cat_df)
    cat_df_ohe = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out())

    # Numeric DataFrame
    num_df = pd.DataFrame([{
        "raw_area": raw_area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "distance_to_center": distance_to_center
    }])

    # Combine
    input_df = pd.concat([cat_df_ohe, num_df], axis=1)

    # Add missing features (in case)
    for col in final_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[final_features]

    # Debug preview
    st.markdown("### 🔍 Model Input Preview")
    st.write(input_df)

    # Prediction
    X_scaled = scaler.transform(input_df)
    log_price = model.predict(X_scaled)[0]
    predicted_price = np.exp(log_price)

    # Result
    st.subheader("💰 Estimated House Price")
    st.success(f"PKR {predicted_price:,.0f}")
