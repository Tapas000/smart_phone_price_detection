import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model and dataset
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Set the title of the app
st.title("Smart Phone Price Prediction")

# Collect user inputs
company = st.selectbox('Select Brand', df['brand_name'].unique())
processor_brand = st.selectbox('Select Processor Brand', df['processor_brand'].unique())
os = st.selectbox('Select Operating System', df['os'].unique())

processor_speed = st.number_input("Processor Speed (GHz)", min_value=0.1, max_value=5.0, step=0.1)
ram_capacity = st.selectbox('RAM Capacity (GB)', df['ram_capacity'].unique())
internal_memory = st.selectbox('Storage (GB)', df['internal_memory'].unique())
refresh_rate = st.selectbox('Refresh Rate (Hz)', df['refresh_rate'].unique())
num_rear_cameras = st.selectbox('Number of Rear Cameras', df['num_rear_cameras'].unique())
num_front_cameras = st.selectbox('Number of Front Cameras', df['num_front_cameras'].unique())
ppi = st.number_input("Pixel Density (PPI)", min_value=100, max_value=800, step=10)

# Convert categorical Yes/No to numerical (binary)
has_5g = st.radio("5G Connectivity", ['Yes', 'No'])
has_nfc = st.radio("NFC Support", ['Yes', 'No'])
has_5g = 1 if has_5g == 'Yes' else 0
has_nfc = 1 if has_nfc == 'Yes' else 0

# Predict price when button is clicked
if st.button("💰 Predict Price"):
    # Convert inputs into a NumPy array
    query = np.array([company, has_5g, has_nfc, processor_brand, processor_speed,
                      ram_capacity, internal_memory, refresh_rate, num_rear_cameras,
                      num_front_cameras, os, ppi]).reshape(1, -1)

    # Convert to DataFrame
    query_df = pd.DataFrame(query, columns=['brand_name', 'has_5g', 'has_nfc', 'processor_brand',
                                            'processor_speed', 'ram_capacity', 'internal_memory',
                                            'refresh_rate', 'num_rear_cameras', 'num_front_cameras',
                                            'os', 'ppi'])

    # Predict the price
    predicted_price = np.exp(pipe.predict(query_df)[0])  # Apply exponential if log-transformed

    # Display the result
    st.success(f"💲 The estimated price of this smartphone is **₹{int(predicted_price)}**")
