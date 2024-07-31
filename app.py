import pandas as pd
import streamlit as st
import pickle

# User inputs
age = st.number_input('Age')
sex = st.selectbox('Sex', ['male', 'female'])
BMI = st.number_input('BMI')
children = st.number_input('Number of Children')
smoker = st.selectbox('Do you smoke?', ['yes', 'no'])
region = st.selectbox('Region', ['southwest', 'northwest', 'southeast', 'northeast'])

# Create DataFrame
X = pd.DataFrame([[age, sex, BMI, children, smoker, region]], columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

# Debug: Check the type and structure of X
# st.write(type(X))
# st.write(X.head())

# Ensure one-hot encoding (if required)
X_encoded = pd.get_dummies(X, columns=['sex', 'smoker', 'region'])

# Load the model
with open('insurance_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict
try:
    pred = model.predict(X_encoded)
    st.write(f'The model predicts an insurance cost of $ {pred[0]}')
except Exception as e:
    st.write(f"An error occurred: {e}")
