import streamlit as st
import pickle
import pandas as pd
import requests
import os
import requests
#import lightgbm

# Try installing lightgbm
try:
    import lightgbm
except ImportError:
    st.write("Installing lightgbm...")
    !pip install lightgbm
    import lightgbm


# URL of the raw file on GitHub
file_url = 'https://raw.githubusercontent.com/hajouasmae1/diabetes_dataset/lgbm_tuned_model.pkl'

# Send a GET request to the URL
response = requests.get(file_url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Open the downloaded file using a context manager
    with open('lgbm_tuned_model.pkl', 'wb') as f:
        f.write(response.content)

# Load the trained model
with open('lgbm_tuned_model.pkl', 'rb') as f:
    lgbm_model = pickle.load(f)

# Function to make predictions
def predict_outcome(model, data):
    prediction = model.predict(data)
    return prediction

import streamlit as st
def main():
    st.title("Diabetes Prediction App")

    # User input for features
    pregnancies = st.slider("Number of times pregnant", 0, 20, 0)
    glucose = st.slider("Plasma glucose concentration", 0, 200, 0)
    blood_pressure = st.slider("Diastolic blood pressure", 0, 150, 0)
    skin_thickness = st.slider("Triceps skin fold thickness", 0, 100, 0)
    insulin = st.slider("2-Hour serum insulin", 0, 1000, 0)
    bmi = st.slider("Body mass index", 0.0, 70.0, 0.0)
    pedigree = st.slider("Diabetes pedigree function", 0.0, 2.0, 0.0)
    age = st.slider("Age", 0, 150, 0)

    # Create a DataFrame from user input
    user_data = pd.DataFrame({
        "Number of times pregnant": [pregnancies],
        "Plasma glucose concentration": [glucose],
        "Diastolic blood pressure": [blood_pressure],
        "Triceps skin fold thickness": [skin_thickness],
        "2-Hour serum insulin": [insulin],
        "Body mass index": [bmi],
        "Diabetes pedigree function": [pedigree],
        "Age": [age]
    })

     # Use the trained model to make a prediction
    prediction = predict_outcome(lgbm_model, user_data)

    # Display the prediction
    if prediction[0] == 1:
        st.error('Patient is likely to have diabetes.')
    else:
        st.success('Patient is likely to be diabetes-free.')

if __name__ == "__main__":
    main()
