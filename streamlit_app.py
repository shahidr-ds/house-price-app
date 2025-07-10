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
    try:
        input_data = {}

        # --- Step 1: Engineered numerical features ---
        input_data["Total_Area_log"] = np.log(area)
        input_data["bed_bath_ratio"] = bedrooms / bathrooms
        input_data["total_rooms"] = bedrooms + bathrooms
        input_data["area_per_bedroom"] = area / bedrooms
        input_data["price_per_room"] = 0
        input_data["is_weekend"] = 0
        input_data["season_winter"] = 0
        input_data["location_cluster"] = 0

        # --- Step 2: One-hot encode location and property type ---
        for col in scaler.feature_names_in_:
            if col.startswith("location_"):
                input_data[col] = 1 if col == f"location_{location}" else 0
            elif col.startswith("property_type_"):
                input_data[col] = 1 if col == f"property_type_{property_type}" else 0

        # --- Step 3: Convert to DataFrame and scale ---
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)

        # --- Step 4: Predict ---
        log_price = model.predict(input_scaled)[0]
        predicted_price = np.exp(log_price)

        # --- Step 5: Display result ---
        st.subheader("🏷️ Estimated House Price")
        st.success(f"PKR {predicted_price:,.0f}")

    except ZeroDivisionError:
        st.error("Bedrooms and bathrooms must be greater than zero.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
