import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('xgboost_house_price_model.pkl', 'rb'))

# Streamlit app layout
st.set_page_config(page_title="Pakistan House Price Predictor")
st.title("🏡 House Price Prediction (Pakistan)")
st.write("Enter the details below to predict house price:")

# User inputs
area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=500)
bedrooms = st.selectbox("Number of Bedrooms", [1, 2, 3, 4, 5])
bathrooms = st.selectbox("Number of Bathrooms", [1, 2, 3, 4])
city = st.selectbox("City", ['Lahore', 'Karachi', 'Islamabad'])  # Update if you have more cities

# Encode 'city' (example encoding: one-hot or label, depending on your model training)
city_encoded = 0  # Placeholder (adjust if you used encoding)

# Prepare input (update to match model input format exactly)
# Replace with correct order of features
input_data = np.array([[area, bedrooms, bathrooms]])

# Prediction
if st.button("Predict Price"):
    price = model.predict(input_data)[0]
    st.success(f"🏷️ Estimated Price: PKR {price:,.0f}")
