import streamlit as st
import numpy as np
import pandas as pd
import pickle

# --- Load model and scaler ---
with open("models/xgboost_house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --- Streamlit UI ---
st.set_page_config(page_title="Pakistan House Price Prediction", layout="centered")
st.title("🏠 House Price Prediction (Pakistan)")

st.markdown("""
This app predicts house prices in Pakistan based on area, rooms, location and more. 
Make your selections below:
""")

# --- User inputs ---
area = st.number_input("Total Area (in Marla)", min_value=1.0, value=5.0)
bedrooms = st.number_input("Bedrooms", min_value=1, value=2)
bathrooms = st.number_input("Bathrooms", min_value=1, value=2)

location = st.selectbox("Location", [
    "DHA Defence", "E-11", "F-7", "F-8", "G-13", "G-15", "Soan Garden", "Other"
])

property_type = st.selectbox("Property Type", [
    "Flat", "House", "Lower Portion", "Upper Portion", "Room", "Penthouse"
])

if st.button("Predict Price"):
    # --- Prepare input ---
    feature_names = model.get_booster().feature_names
    input_data = dict.fromkeys(feature_names, 0)

    # Engineered features
    try:
        input_data["Total_Area_log"] = np.log(area)
        input_data["bed_bath_ratio"] = bedrooms / bathrooms
        input_data["total_rooms"] = bedrooms + bathrooms
        input_data["area_per_bedroom"] = area / bedrooms
    except ZeroDivisionError:
        st.error("Bedrooms and bathrooms must be greater than zero.")
        st.stop()

    input_data["is_weekend"] = 0
    input_data["season_winter"] = 0
    input_data["location_cluster"] = 0
    input_data["price_per_room"] = 0  # Not used at prediction time

    # One-hot encode location and property type
    loc_key = f"location_{location}"
    ptype_key = f"property_type_{property_type}"

    if loc_key in input_data:
        input_data[loc_key] = 1
    else:
        st.warning(f"Location '{location}' not found in training set.")

    if ptype_key in input_data:
        input_data[ptype_key] = 1
    else:
        st.warning(f"Property type '{property_type}' not found in training set.")

    # Convert to DataFrame and scale
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    # Predict
    log_price = model.predict(input_scaled)[0]
    predicted_price = np.exp(log_price)

    # Display result
    st.subheader("🏷️ Estimated House Price")
    st.success(f"PKR {predicted_price:,.0f}")
