import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

import os
import streamlit as st

# Show the current working directory
st.write("📂 Current working directory:", os.getcwd())

# List all files in that directory
st.write("📁 Files in this directory:")
st.write(os.listdir())


# Load trained model and preprocessors
model = joblib.load("xgboost_house_price_model.pkl")
scaler = joblib.load("scaler.pkl")               # StandardScaler used on X_train
ohe = joblib.load("ohe.pkl")                     # OneHotEncoder fitted on property_type and location
final_features = joblib.load("final_features.pkl")  # List of column names used in training

# Set title
st.title("🏠 Pakistan House Price Prediction App")

# User Inputs
area = st.number_input("Total Area (in Marla)", value=5.0)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
distance_to_center = st.slider("Distance to City Center (km)", min_value=0.0, max_value=50.0, value=10.0)

property_type = st.selectbox("Property Type", ["Flat", "House", "Lower Portion", "Upper Portion", "Penthouse", "Room"])
location = st.selectbox("Location", ohe.categories_[1].tolist())  # Assuming location is 2nd category in OHE

# Feature Engineering
total_area_log = np.log(area + 1)
bed_bath_ratio = bedrooms / bathrooms if bathrooms != 0 else 0
total_rooms = bedrooms + bathrooms
season = datetime.today().month
season_winter = 1 if season in [12, 1, 2] else 0
area_per_bedroom = area / bedrooms if bedrooms != 0 else 0
price_per_room = area / total_rooms if total_rooms != 0 else 0
is_weekend = 1 if datetime.today().weekday() >= 5 else 0
location_cluster = 0  # You can replace this if you had a KMeans cluster model

# Categorical features to OHE
cat_df = pd.DataFrame([[property_type, location]], columns=['property_type', 'location'])
cat_encoded = ohe.transform(cat_df).toarray()
cat_df_ohe = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out())

# Numerical features
num_data = pd.DataFrame([{
    'is_weekend': is_weekend,
    'Total_Area_log': total_area_log,
    'distance_to_center': distance_to_center,
    'bed_bath_ratio': bed_bath_ratio,
    'total_rooms': total_rooms,
    'season_winter': season_winter,
    'area_per_bedroom': area_per_bedroom,
    'price_per_room': price_per_room,
    'location_cluster': location_cluster
}])

# Combine all features
input_df = pd.concat([cat_df_ohe, num_data], axis=1)

# Align with training features
X = pd.DataFrame(columns=final_features)
X = pd.concat([X, input_df], ignore_index=True).fillna(0)

# Apply scaling
X_scaled = scaler.transform(X)

# Predict (model predicts log(price), we convert back)
log_price = model.predict(X_scaled)[0]
predicted_price = np.exp(log_price)

# Show result
st.subheader("🏷️ Estimated Price:")
st.success(f"PKR {predicted_price:,.0f}")
