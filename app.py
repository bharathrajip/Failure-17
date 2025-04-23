
import streamlit as st
import pandas as pd
import pickle

# Load model
with open('disease_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load encoders
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

st.title("Disease Prediction App")

# Sample user input fields
def user_input_features():
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    fever = st.selectbox("Fever", ["Yes", "No"])
    input_data = {'Age': [age], 'Gender': [gender], 'Fever': [fever]}
    return pd.DataFrame(input_data)

input_df = user_input_features()

# Apply label encoding safely
for col in input_df.columns:
    if col in label_encoders:
        input_df[col] = input_df[col].apply(
            lambda val: val if val in label_encoders[col].classes_ else label_encoders[col].classes_[0]
        )
        input_df[col] = label_encoders[col].transform(input_df[col])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f"Predicted Disease: {prediction[0]}")
