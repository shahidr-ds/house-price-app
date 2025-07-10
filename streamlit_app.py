import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load model and scaler ---
with open("models/xgboost_house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --- Get feature names from scaler and model ---
scaler_features = scaler.feature_names_in_
model_features = model.get_booster().feature_names

# --- Streamlit UI ---
st.title("🏠 House Price Prediction App")
st.sidebar.header("Enter Property Details")

# --- User input ---
area = st.sidebar.number_input("Area (in marla)", min_value=1.0, max_value=100.0, value=5.0)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2)

location = st.sidebar.selectbox("Location", [
    "DHA Defence", "E-11", "F-7", "F-8", "G-13", "G-15", "Other", "Soan Garden"
])

property_type = st.sidebar.selectbox("Property Type", [
    "Flat", "House", "Lower Portion", "Penthouse", "Room", "Upper Portion"
])

if st.sidebar.button("Predict Price"):

    # --- Step 1: Initialize input_data with scaler features ---
    input_data = dict.fromkeys(scaler_features, 0)

    # --- Step 2: One-hot encoded fields ---
    location_col = f"location_{location}"
    prop_col = f"property_type_{property_type}"

    if location_col in input_data:
        input_data[location_col] = 1
    if prop_col in input_data:
        input_data[prop_col] = 1

    # --- Step 3: Engineered numerical features ---
    input_data["Total_Area_log"] = np.log(area) if area > 0 else 0
    input_data["bed_bath_ratio"] = bedrooms / bathrooms if bathrooms != 0 else 0
    input_data["total_rooms"] = bedrooms + bathrooms
    input_data["area_per_bedroom"] = area / bedrooms if bedrooms != 0 else 0
    input_data["price_per_room"] = 0  # price unknown at prediction time

    # --- Optional fields ---
    input_data["is_weekend"] = 0
    input_data["season_winter"] = 0
    input_data["location_cluster"] = 0

    # --- Step 4: Create DataFrame for scaler ---
    input_df = pd.DataFrame([input_data])[scaler_features]

    # --- Step 5: Scale the input ---
    input_scaled = scaler.transform(input_df)

    # --- Step 6: Prepare model input ---
    input_scaled_df = pd.DataFrame(input_scaled, columns=scaler_features)

    # Add any missing model features as zeros
    for col in model_features:
        if col not in input_scaled_df:
            input_scaled_df[col] = 0

    # Reorder columns
    input_scaled_df = input_scaled_df[model_features]

    # --- Step 7: Predict ---
    log_price = model.predict(input_scaled_df, validate_features=True)[0]
    predicted_price = np.exp(log_price)

    # --- Step 8: Show result ---
    st.subheader("🏷️ Predicted House Price")
    st.success(f"Estimated Price: **PKR {predicted_price:,.0f}**")

    # --- Optional: Debug Panel ---
    with st.expander("🔎 Debug Info"):
        st.write("Input Data (before scaling):", input_df)
        st.write("Scaled Input for Model:", input_scaled_df)
        st.write("Bedrooms:", bedrooms, "| Bathrooms:", bathrooms)
        st.write("Engineered Features:", {
            "bed_bath_ratio": input_data["bed_bath_ratio"],
            "total_rooms": input_data["total_rooms"],
            "area_per_bedroom": input_data["area_per_bedroom"]
        })
